"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from si2ca.runtime import sandbox_lease

logger = logging.getLogger(__name__)

ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path

@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...

def _getenv(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``SLIME_AGENT_*`` name first, older names after."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default

class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    image_metadata_key_env = ("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")

    default_lifetime_sec = 3600
    default_rpc_retries = 3

    rpc_backoff_base_sec = 1.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @staticmethod
    def _is_transient_rpc_error(e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory):
        """Run coro_factory() with retries for transient E2B RPC failures."""
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    backoff = self.rpc_backoff_base_sec * (2**attempt)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def __aenter__(self) -> E2BSandbox:
        if self.image_metadata_key is None:
            raise RuntimeError(
                "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        from e2b import AsyncSandbox  

        md = {self.image_metadata_key: self.image}
        self._sb = await AsyncSandbox.create(timeout=self.timeout, metadata=md)
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""

class LocalDockerSandbox:
    """Async context manager around a local Docker container.

    Drop-in replacement for ``E2BSandbox`` that runs each rollout in a local
    ``docker`` container instead of a remote E2B microVM. Implements the same
    ``Sandbox`` contract (exec / write_file / read_file) so the harness, swe task
    layer, and ``generate`` orchestrator work unchanged. The ``image`` is a local
    docker image tag; nothing is pulled from a gateway.

    Networking: the agent CLI inside the container dials back to the adapter at
    ``ADAPTER_PUBLIC_HOST``. By default we add ``host.docker.internal`` mapped to
    the host gateway so a sandbox can reach an adapter bound on the host; set
    ``SLIME_AGENT_DOCKER_NETWORK=host`` to share the host network instead.
    """

    network_env = ("SLIME_AGENT_DOCKER_NETWORK",)
    extra_run_args_env = ("SLIME_AGENT_DOCKER_RUN_ARGS",)
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")

    docker_host_env = ("SLIME_AGENT_DOCKER_HOST",)

    default_lifetime_sec = 3600

    def __init__(self, image: str, *, timeout: int | None = None,
                 cpus: float | None = None, memory_mb: int | None = None) -> None:
        self.image = image

        self.cpus = cpus
        self.memory_mb = memory_mb
        self.timeout = timeout if timeout is not None else int(
            _getenv(*self.lifetime_sec_env, default=str(self.default_lifetime_sec))
        )
        self.sandbox_id = ""
        self._cid = ""
        self._docker_url = ""  

        self._uids: dict[str, str] = {"root": "0"}

    def register_user_uid(self, name: str, uid: str) -> None:
        self._uids[name] = uid

    @classmethod
    def _docker_hosts(cls) -> list[str]:
        raw = _getenv(*cls.docker_host_env)
        return [u.strip() for u in raw.split(",") if u.strip()] if raw else []

    def _docker_cmd(self) -> list[str]:
        if self._docker_url:
            return ["podman", "--remote", "--url", self._docker_url]
        hosts = self._docker_hosts()
        return ["podman", "--remote", "--url", hosts[0]] if hosts else ["docker"]

    @staticmethod
    async def _run(argv: list[str], *, stdin: bytes | None = None, timeout: int = 300) -> ExecResult:
        """Run a docker CLI command (argv, no shell) and return (rc, out, err).

        Drain both pipes while the process runs so large test logs cannot
        block its exit. After exit, bound the remaining drain: detached Podman
        children may inherit a pipe and keep it open for the container lifetime.
        """
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _feed_stdin() -> None:
            if stdin is None or proc.stdin is None:
                return
            try:
                proc.stdin.write(stdin)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        async def _drain(stream: asyncio.StreamReader | None, buffer: bytearray) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                buffer.extend(chunk)

        out, err = bytearray(), bytearray()
        readers = [asyncio.create_task(_drain(proc.stdout, out)),
                   asyncio.create_task(_drain(proc.stderr, err))]
        feeder = asyncio.ensure_future(_feed_stdin())

        async def _wait_for_exit() -> None:
            # Process.wait() can also wait for EOF on inherited pipes. The
            # transport sets returncode as soon as the actual child exits.
            while proc.returncode is None:
                await asyncio.sleep(0.05)

        try:
            try:
                await asyncio.wait_for(_wait_for_exit(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(_wait_for_exit(), timeout=30)
                except asyncio.TimeoutError:
                    logger.warning("[local docker] pid %s survived SIGKILL for 30s after a "
                                   "%ss timeout; abandoning the wait so the caller can proceed",
                                   getattr(proc, "pid", "?"), timeout)
                return 124, "", f"local docker command timed out after {timeout}s"
            try:
                await asyncio.wait_for(asyncio.gather(*readers), timeout=3)
            except asyncio.TimeoutError:
                pass
            return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        finally:
            for task in [feeder, *readers]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(feeder, *readers, return_exceptions=True)

    async def __aenter__(self) -> LocalDockerSandbox:
        boot_timeout = int(os.environ.get("SLIME_SANDBOX_BOOT_TIMEOUT", "420"))
        attempts = max(1, int(os.environ.get("SLIME_SANDBOX_BOOT_RETRIES", "3")))
        last_rc, last_err = 0, ""
        hosts = self._docker_hosts()
        lock_heal_done = False
        for i in range(attempts):
            
            seed = int.from_bytes(os.urandom(4), "big")
            name = f"slime-sb-{os.getpid()}-{seed:08x}"
            if hosts:

                self._docker_url = hosts[(seed + i) % len(hosts)]
            argv = [*self._docker_cmd(), "run", "-d", "--name", name, "--label", "slime-sandbox=1"]
            if self.cpus:
                argv += ["--cpus", str(self.cpus)]
            if self.memory_mb:
                argv += ["--memory", f"{int(self.memory_mb)}m"]

            rollout_key = sandbox_lease.current_key()
            if rollout_key:
                argv += ["--label", f"{sandbox_lease.LABEL}={rollout_key}"]
            network = _getenv(*self.network_env)
            if network:
                argv += ["--network", network]
            else:
                
                argv += ["--add-host", "host.docker.internal:host-gateway"]
            extra = _getenv(*self.extra_run_args_env)
            if extra:
                argv += shlex.split(extra)
            
            argv += ["--entrypoint", "/bin/sh", self.image, "-c", "sleep infinity"]
            rc, out, err = await self._run(argv, timeout=boot_timeout)
            if rc == 0:
                self._cid = out.strip()
                self.sandbox_id = self._cid[:12]
                ready_err = await self._await_exec_ready()
                if ready_err is None:
                    return self

                rc, err = 125, ready_err
                self._cid = ""

            last_rc, last_err = rc, err.strip()[:400]
            await self._run([*self._docker_cmd(), "rm", "-f", name], timeout=30)
            if not lock_heal_done and ("num_locks" in err or "allocating lock" in err):

                lock_heal_done = True
                logger.warning(
                    "[agent.sandbox] lock exhaustion on %s — emergency cleanup of exited containers",
                    self._docker_url or "local",
                )
                rc_ls, out_ls, _ = await self._run(
                    [*self._docker_cmd(), "ps", "-aq", "--filter", "status=exited"], timeout=60
                )
                if rc_ls == 0 and out_ls.strip():
                    ids = out_ls.split()
                    await self._run([*self._docker_cmd(), "rm", "-f", *ids[:200]], timeout=120)
                continue
            if i < attempts - 1:
                await asyncio.sleep(5 * (i + 1))
        raise RuntimeError(
            f"docker run failed (rc={last_rc}) after {attempts} attempts for image {self.image!r}: {last_err}"
        )

    async def _await_exec_ready(self) -> str | None:
        """Block until the container can actually run an exec; None once it can.

        ``run -d`` returning 0 only means the container was created: on rootless
        podman + fuse-overlayfs the rootfs can still be settling, and an exec
        issued in that window dies with crun's ``unable to find user root: no
        matching entries in passwd file`` — it cannot read /etc/passwd out of a
        rootfs that is not fully there yet. The window widens with image size and
        concurrent boots, so multi-GB images (SWE-bench Pro's sweap-images) at
        high concurrency hit it routinely while small images almost never do.
        Callers treat that as a hard sandbox failure, which is how a whole Pro
        grading pass can come back 0/3 while the same commands pass by hand.
        """
        deadline = time.monotonic() + float(os.environ.get("SLIME_SANDBOX_READY_TIMEOUT", "60"))
        last = ""
        delay = 0.5

        probe = [*self._docker_cmd(), "exec", "-u", "root", self._cid, "bash", "-c", "id root >/dev/null"]
        while True:
            rc, _, err = await self._run(probe, timeout=60)
            if rc == 0:
                return None
            last = err.strip()[:300]
            if time.monotonic() >= deadline:
                return f"container never became exec-ready: {last}"
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self._cid:
            return
        try:
            rc, _, _ = await self._run([*self._docker_cmd(), "rm", "-f", self._cid], timeout=60)
            if rc != 0:

                await self._run([*self._docker_cmd(), "kill", self._cid], timeout=30)
                await self._run([*self._docker_cmd(), "rm", "-f", self._cid], timeout=60)
        except Exception as e:
            logger.warning("[agent.sandbox] docker rm %s failed: %s", self.sandbox_id, e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        exec_id = uuid.uuid4().hex
        argv = [*self._docker_cmd(), "exec", "-u", self._uids.get(user, user),
                "-e", f"SLIME_EXEC_ID={exec_id}"]

        if not (env or {}).get("HOME"):
            uid = str(self._uids.get(user, user))
            argv += ["-e", f"HOME={'/root' if uid in ('0', 'root') else f'/home/{user}'}"]
        for k, v in (env or {}).items():
            argv += ["-e", f"{k}={v}"]

        stdin_payload: bytes | None = None
        if len(cmd.encode("utf-8", "surrogateescape")) > 100_000:
            argv += [self._cid, "bash", "-s"]
            stdin_payload = cmd.encode("utf-8", "surrogateescape")
        else:
            argv += [self._cid, "bash", "-c", cmd]
        rc, out, err = await self._run(argv, stdin=stdin_payload, timeout=timeout)
        if rc == 124:
            await self._reap_in_container(exec_id, timeout)

        for attempt in range(2):
            if rc == 0 or "no matching entries in passwd file" not in err:
                break
            await asyncio.sleep(2 * (attempt + 1))
            rc, out, err = await self._run(argv, stdin=stdin_payload, timeout=timeout)
        if check and rc != 0:
            raise RuntimeError(f"local exec failed (exit={rc}): {cmd[:120]}\n{err[:400]}")
        return rc, out, err

    async def _reap_in_container(self, exec_id: str, timeout: int) -> None:
        """Terminate the in-container process tree after a command times out.

        _run only terminates the host-side docker exec client. Container processes
        can survive and accumulate while the agent continues after rc=124. In a
        historical run, repeated recursive grep commands scanned /proc and consumed
        200% CPU per container; 48 such containers could saturate a 96-core host.

        Match the inherited SLIME_EXEC_ID environment variable via /proc rather
        than relying on process groups or pkill/procps, which images may lack.
        The reaper shell does not carry that ID and therefore cannot match itself.
        Cleanup is best-effort and must not change the original command result.
        """
        script = ('for p in /proc/[0-9]*; do '
                  f'grep -qa "SLIME_EXEC_ID={exec_id}" "$p/environ" 2>/dev/null '
                  '&& kill -9 "${p##*/}" 2>/dev/null; done; true')
        try:
            await self._run([*self._docker_cmd(), "exec", "-u", "0", self._cid,
                             "sh", "-c", script], timeout=30)
        except Exception as e:  
            logger.warning("[local docker] in-container reap after %ss timeout failed: %s",
                           timeout, type(e).__name__)

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            
            rc, _, err = await self._run(
                [*self._docker_cmd(), "cp", str(content), f"{self._cid}:{sandbox_path}"], timeout=600
            )
            if rc != 0:
                raise RuntimeError(f"docker cp -> {sandbox_path} failed (rc={rc}): {err.strip()[:300]}")
            if user != "root":
                await self.exec(f"chown {user}:{user} {shlex.quote(sandbox_path)}", user="root", timeout=60)
            return

        data = content.encode("utf-8") if isinstance(content, str) else content
        
        argv = [
            *self._docker_cmd(),
            "exec",
            "-i",
            "-u",
            self._uids.get(user, user),
            self._cid,
            "sh",
            "-c",
            f"cat > {shlex.quote(sandbox_path)}",
        ]
        rc, _, err = await self._run(argv, stdin=data, timeout=600)

        for attempt in range(4):
            if rc == 0 or "unlinkat" not in err or "directory not empty" not in err:
                break
            logger.warning(
                "[agent.sandbox] podman exec cleanup race writing %s (attempt %d/4); retrying",
                sandbox_path, attempt + 1,
            )
            await asyncio.sleep(2 * (attempt + 1))
            rc, _, err = await self._run(argv, stdin=data, timeout=600)
        if rc != 0:
            raise RuntimeError(f"write_file {sandbox_path} failed (rc={rc}): {err.strip()[:300]}")

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            host_p = os.path.join(td, "f")
            rc, _, _ = await self._run(
                [*self._docker_cmd(), "cp", f"{self._cid}:{sandbox_path}", host_p], timeout=300
            )
            if rc == 0:
                try:
                    return Path(host_p).read_text(errors="replace")
                except OSError:
                    pass
        rc, out, _ = await self.exec(f"cat {shlex.quote(sandbox_path)}", user=user, timeout=120, check=False)
        return out if rc == 0 else ""

    async def open_persistent_shell(
        self, *, user: str = "root", env: dict[str, str] | None = None, workdir: str | None = None
    ) -> "PersistentDockerShell":
        """Open a STATEFUL bash session in the container (cd / env / bg jobs persist).

        The default exec() is one-shot: every command is a fresh `docker exec bash -c`,
        so `cd`, exported vars, and background services do NOT survive between commands.
        That is correct for SWE-bench (each step is self-contained) but structurally caps
        Terminal-Bench, whose tasks assume a persistent terminal — start a service then
        poke it, cd around, accumulate state. The official Terminus agent gets this via a
        tmux pane; we get the same persistence with a long-lived `docker exec -i bash`
        reading from stdin. No tmux/script needed inside the image (only bash, which every
        TB3 image ships); all control stays host-side.
        """
        argv = [*self._docker_cmd(), "exec", "-i", "-u", self._uids.get(user, user), self._cid,
                "bash", "--noprofile", "--norc"]

        shell = PersistentDockerShell(
            argv, env=env or {}, workdir=workdir,
            kill_prefix=[*self._docker_cmd(), "exec", "-u", "0", self._cid])
        await shell.start()
        return shell

class PersistentDockerShell:
    """A stateful bash session inside a container: cd / env / background jobs persist.

    ``run()`` returns the same ``(returncode, combined_output, "")`` shape as
    ``LocalDockerSandbox.exec`` so the harness swaps it in transparently. stderr is
    merged into stdout — the agent sees what a human at the terminal sees. Non-interactive
    bash reading a pipe does not echo its input, so stdout carries only real command
    output (no command-text to strip back out).

    Command completion is detected by printing a per-command random marker plus ``$?``
    after the command; ``run`` blocks until that marker line appears. A command exceeding
    ``timeout`` is interrupted with Ctrl-C and reported rc=124, then the session is probed
    to confirm it survived. The one case one-shot exec handled implicitly and a persistent
    shell must handle explicitly: an unclosed heredoc / genuinely hung foreground process
    makes the marker never arrive — caught by the timeout + Ctrl-C + liveness probe.
    """

    _READY = "__MSWE_SHELL_READY__"

    def __init__(self, argv: list[str], *, env: dict[str, str], workdir: str | None,
                 kill_prefix: list[str] | None = None):
        self._argv = argv
        self._env = env
        self._workdir = workdir
        self._kill_prefix = kill_prefix or []
        self._shell_pid: str | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._pending = ""
        self._lock = asyncio.Lock()
        self.broken = False

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        init = ["export PS1=''", "export PS2=''"]
        for k, v in self._env.items():
            init.append(f"export {k}={shlex.quote(str(v))}")
        if self._workdir:
            init.append(f"cd {shlex.quote(self._workdir)} 2>/dev/null || true")

        init.append("printf '__MSWE_PID__%d\\n' $$")
        init.append(f"printf '\\n%s\\n' {self._READY}")
        self._proc.stdin.write(("\n".join(init) + "\n").encode())
        await self._proc.stdin.drain()
        
        await asyncio.wait_for(self._read_until(self._READY), timeout=30)
        buf = self._take_pending()
        for line in buf.splitlines():
            if line.startswith("__MSWE_PID__"):
                self._shell_pid = line[len("__MSWE_PID__"):].strip() or None

    def _take_pending(self) -> str:
        out, self._pending = self._pending, ""
        return out

    async def _read_until(self, marker: str) -> int | None:
        """Read stdout lines into ``self._pending`` until a line carries ``marker``.

        Returns the exit code parsed after the marker (None if the marker had no code,
        e.g. the liveness probe). Accumulating into an instance buffer — not a local — is
        deliberate: if the caller's ``wait_for`` cancels us on timeout, the bytes read so
        far are still salvageable as partial output.
        """
        assert self._proc and self._proc.stdout
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:  
                self.broken = True
                return 1
            line = raw.decode("utf-8", "replace")
            idx = line.find(marker)
            if idx == -1:
                self._pending += line
                continue
            if idx > 0:
                self._pending += line[:idx]
            rest = line[idx + len(marker):].strip()
            try:
                return int(rest)
            except ValueError:
                return None

    async def _interrupt(self) -> None:
        """Kill the shell's foreground children (the timed-out command) from outside.

        Ctrl-C down a pipe is inert (no TTY -> no SIGINT), so we pkill -P <shell_pid>
        via a separate root exec. TERM then KILL. Once the children die, the pending
        `printf marker $?` for the timed-out command finally runs, which the caller
        consumes to keep the buffer clean.
        """
        if self._kill_prefix and self._shell_pid:
            argv = [*self._kill_prefix, "bash", "-c",
                    f"pkill -TERM -P {self._shell_pid} 2>/dev/null; sleep 1; "
                    f"pkill -KILL -P {self._shell_pid} 2>/dev/null; true"]
            try:
                p = await asyncio.create_subprocess_exec(
                    *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await asyncio.wait_for(p.wait(), timeout=25)
            except Exception:
                pass
        else:  
            try:
                self._proc.stdin.write(b"\x03")
                await self._proc.stdin.drain()
            except Exception:
                pass

    async def _probe_alive(self) -> bool:
        m = f"__MSWE_PROBE_{uuid.uuid4().hex}__"
        try:
            self._proc.stdin.write(f"printf '\\n%s\\n' {m}\n".encode())
            await self._proc.stdin.drain()
            await asyncio.wait_for(self._read_until(m), timeout=15)
            self._take_pending()
            return True
        except Exception:
            return False

    async def run(self, cmd: str, timeout: int = 120) -> ExecResult:
        if self.broken or self._proc is None or self._proc.returncode is not None:
            return 1, "", "persistent shell is not running"
        async with self._lock:
            marker = f"__MSWE_SHELL_{uuid.uuid4().hex}__"

            payload = f"{cmd}\nprintf '\\n%s%d\\n' {marker} $?\n"
            try:
                self._proc.stdin.write(payload.encode("utf-8", "surrogateescape"))
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                self.broken = True
                return 1, "", "persistent shell stdin closed"
            try:
                rc = await asyncio.wait_for(self._read_until(marker), timeout=timeout)
                return (rc if rc is not None else 0), self._take_pending(), ""
            except asyncio.TimeoutError:
                partial = self._take_pending()
                await self._interrupt()  

                try:
                    await asyncio.wait_for(self._read_until(marker), timeout=20)
                    self._take_pending()
                except Exception:
                    if not await self._probe_alive():
                        self.broken = True
                return 124, partial, ""

    async def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

_SANDBOX_BACKENDS = {
    "e2b": E2BSandbox,
    "local_docker": LocalDockerSandbox,
}
sandbox_backend_env = ("SLIME_AGENT_SANDBOX_BACKEND", "SWE_SANDBOX_BACKEND")

def create_sandbox(image: str, **kwargs) -> Sandbox:
    """Instantiate the sandbox backend named by ``SLIME_AGENT_SANDBOX_BACKEND``.

    Defaults to ``e2b`` to preserve existing behavior; set ``local_docker`` to
    run rollouts in local docker containers. Unknown backend-specific kwargs are
    ignored by backends that don't accept them."""
    name = _getenv(*sandbox_backend_env, default="e2b").lower()
    cls = _SANDBOX_BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"{sandbox_backend_env[0]}={name!r} not in {sorted(_SANDBOX_BACKENDS)}")
    if cls is LocalDockerSandbox:
        return cls(image, timeout=kwargs.get("timeout"),
                   cpus=kwargs.get("cpus"), memory_mb=kwargs.get("memory_mb"))

    return cls(image, **{k: v for k, v in kwargs.items() if k not in ("cpus", "memory_mb")})

async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the 'agent' user that owns workdir + can git diff.

    Default: a real unprivileged user + ``chown -R workdir``. For images with a
    huge workdir (e.g. jefzda ``sweap-images`` openlibrary ``/app`` = 117k files)
    that recursive chown forces an overlayfs copy-up of every inode -> ~196s on a
    single container, which alone exceeds the 180s exec timeout and, under
    concurrency, snowballs into a podman zombie spiral (every failed setup leaves
    a container ``rm -f`` cannot reap, and each zombie slows the boltdb metadata
    lock until every ``docker exec`` crawls). Set ``SWE_AGENT_UID0=1`` to instead
    create ``agent`` with uid 0 (root-equivalent; ``-o`` permits the duplicate
    id): it then already owns every file so no chown is needed (setup ~1s), and a
    write to a root-owned file triggers the copy-up lazily. Running the agent as
    root is standard for SWE-bench grading (its test harness runs as root)."""
    if _getenv("SWE_AGENT_UID0") in ("1", "true", "yes"):

        cmd = (
            "id agent >/dev/null 2>&1 || useradd -o -u 0 -M -d /home/agent -s /bin/bash agent 2>/dev/null "
            "|| adduser -D -H -h /home/agent -s /bin/sh -u 0 agent 2>/dev/null "
            "|| printf 'agent:x:0:0:agent:/home/agent:/bin/sh\\n' >> /etc/passwd; "

            "grep -q '^agent:' /etc/group 2>/dev/null || printf 'agent:x:0:\\n' >> /etc/group; "

            "grep -q '^::1' /etc/hosts 2>/dev/null && { grep -v '^::1' /etc/hosts > /tmp/.hosts.v4 && cat /tmp/.hosts.v4 > /etc/hosts; }; "
            "id agent >/dev/null 2>&1 && "
            "mkdir -p /home/agent /workspace && chmod 777 /workspace && "
            "git config --system --add safe.directory '*' && id agent"
        )
    else:
        cmd = (
            f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent 2>/dev/null "
            f"|| adduser -D -h /home/agent -s /bin/sh agent 2>/dev/null; "
            f"id agent >/dev/null 2>&1 && "
            f"mkdir -p /workspace && chmod 777 /workspace && "
            f"chown -R agent:agent /home/agent {workdir} && "
            f"git config --system --add safe.directory '*' && id agent"
        )

    _ATTEMPTS = 10
    for attempt in range(_ATTEMPTS):
        rc, _, err = await sb.exec(cmd, user="root", check=False, timeout=180)
        if rc == 0:
            break
        if "unlinkat" not in err or "directory not empty" not in err:
            raise RuntimeError(f"ensure_agent_user failed (exit={rc}): {err[:400]}")
        logger.warning(
            "[agent.sandbox] podman exec cleanup race on %s (attempt %d/%d); retrying",
            getattr(sb, "sandbox_id", "?"), attempt + 1, _ATTEMPTS,
        )
        await asyncio.sleep(min(2 * (attempt + 1), 5))
    else:
        raise RuntimeError(f"ensure_agent_user still failing after {_ATTEMPTS} attempts: {err[:400]}")

    register = getattr(sb, "register_user_uid", None)
    if register is not None:
        uid = ""
        for attempt in range(5):
            _, out, _ = await sb.exec("id -u agent", user="root", timeout=60)
            uid = out.strip().splitlines()[-1].strip() if out.strip() else ""
            if uid.isdigit():
                break
            await asyncio.sleep(1 + attempt)
        if not uid.isdigit():
            raise RuntimeError(
                "could not resolve agent's numeric uid after useradd -- this container's "
                "passwd view is unusable, so every agent exec would fail"
            )
        register("agent", uid)
    await _await_user_visible(sb, "agent")

async def _await_user_visible(sb: Sandbox, user: str, timeout_sec: float = 120.0) -> None:
    """Block until ``exec -u <user>`` resolves, after the useradd that created it.

    The runtime resolves ``-u`` against /etc/passwd as seen through the host-side
    mount of the container rootfs, and on fuse-overlayfs that view can still be
    the pre-useradd one for a moment: the very next exec then dies with "unable
    to find user agent: no matching entries in passwd file" even though useradd
    returned 0. Callers read that as a dead sandbox, so a whole rollout is lost to
    a race that resolves itself in well under a second."""
    deadline = time.monotonic() + timeout_sec
    delay = 0.25
    while True:
        rc, _, err = await sb.exec("true", user=user, check=False, timeout=60)
        if rc == 0:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"user {user!r} never became usable by exec: {err.strip()[:200]}")
        await asyncio.sleep(delay)
        delay = min(delay * 2, 4.0)
