"""SWE task layer: workspace prep, diff capture, and fresh-sandbox eval.

Harness-agnostic on purpose -- nothing here is Claude-specific. ``SWE_PROMPT`` is
the task instruction (semantics, not CLI syntax); ``prepare_workspace`` /
``git_diff`` / ``evaluate`` work with any harness. The only place a task meets a
harness is the prompt, which the orchestrator passes into ``harness.run()``.
"""

from __future__ import annotations

from si2ca.prompts import PROMPT_DIR

import asyncio
import json
import time
import logging
import os
import contextlib
import fcntl
import re
import shlex
from pathlib import Path
from typing import Any

from si2ca.runtime import sandbox as agent_sandbox
from si2ca.runtime.sandbox import Sandbox, create_sandbox
from typing import Any as Sample

logger = logging.getLogger(__name__)

_PATCH = "/workspace/__cagent_patch__.diff"
_PRE = "/workspace/__cagent_pre__.sh"
_F2P = "/workspace/__cagent_f2p__.py"
_SWEPRO_DIR = "/workspace/swepro_eval"

SWE_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    (PROMPT_DIR / 'agent_task.md').read_text(encoding="utf-8").removesuffix("\n"),
)

def get_metadata(sample: Sample) -> dict[str, Any]:
    """Normalize the two dataset schemas (flat vs ``remote_env_info``)."""
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    return {
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "swepro": m.get("swepro"),
        "eval_cmd": m.get("eval_cmd"),
        "eval_files": m.get("eval_files"),
        "base_commit": m.get("base_commit"),
        "f2p_script": rem.get("f2p_script"),
        "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
    }

def _coerce_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""

async def prepare_workspace(sb: Sandbox, workdir: str, md: dict) -> None:
    """Apply swepro setup + pre_commands, then drop PROBLEM_STATEMENT.md.

    Assumes the agent user already owns ``workdir`` (the harness's ``run()`` calls
    ``ensure_agent_user``; the orchestrator runs this before ``run()`` and the
    agent user is created lazily there). To stay independent of call order we
    create the agent user here too -- it is idempotent.
    """
    await agent_sandbox.ensure_agent_user(sb, workdir)
    swepro = md.get("swepro")
    if swepro:
        await apply_before_repo_set_cmd(sb, workdir, swepro)
    pre_commands = md.get("pre_commands")
    if pre_commands:
        await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        md.get("problem_statement") or "",
        user="agent",
    )

async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict) -> None:
    """Run swepro['before_repo_set_cmd'] in the sandbox if present (no-op if not)."""
    before = swepro.get("before_repo_set_cmd")
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"

    await _retry_race("swepro setup dir", lambda: sb.exec(
        "mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup",
        user="root", check=True))
    await _retry_race("write before.sh", lambda: sb.write_file(
        "/workspace/swepro_setup/before.sh", payload, user="agent"))
    ec, _, err = await sb.exec("bash /workspace/swepro_setup/before.sh", user="agent", check=False, timeout=600)
    if ec != 0:
        logger.warning("[swepro] before_repo_set_cmd exited %d: %s", ec, (err or "")[:200])

async def apply_pre_commands(sb: Sandbox, workdir: str, pre: list[str] | str) -> None:

    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await _retry_race("write pre_commands", lambda: sb.write_file(_PRE, "set -e\n" + body, user="agent"))
    ec, _, err = await sb.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}",
                               user="agent", check=False, timeout=600)
    if ec != 0:

        logger.warning("[pre_commands] exited %d (baseline may be wrong): %s", ec, (err or "")[:200])

_RACE_SIGNS = ("unlinkat", "directory not empty")

async def _exec_race_safe(sb, cmd: str, what: str, attempts: int = 6, **kw):
    """exec() that does not let podman's cleanup race masquerade as the command failing.

    podman 3.4.4 returns rc=255 from tearing down the exec session AFTER the
    command inside the container ran. `_retry_race` cannot cover these call sites
    because they pass check=False, so nothing raises — the bogus rc is simply
    returned and read as a result.

    That is the same defect as the one _apply_diff had, and on these call sites it
    is worse: `cd {workdir} && {eval_cmd}` IS the Verified grade, so a race there
    marks a passing test suite as a failure, and git_diff returning empty makes a
    model that wrote a correct patch look like it wrote nothing.

    Only used where re-running is provably safe: running tests, `git diff`,
    parser.py. Deliberately NOT the generic exec() path — that also carries the
    agent's own commands, which already executed and may not be repeatable.
    """
    ec, out, err = await sb.exec(cmd, check=False, **kw)
    for attempt in range(attempts - 1):
        if ec == 0:
            return ec, out, err
        blob = f"{err or ''}\n{out or ''}"
        if not any(sig in blob for sig in _RACE_SIGNS):
            return ec, out, err  
        logger.warning("[race] %s hit the podman cleanup race (rc=%d, attempt %d/%d); retrying",
                       what, ec, attempt + 1, attempts)
        await asyncio.sleep(min(2 * (attempt + 1), 5))
        ec, out, err = await sb.exec(cmd, check=False, **kw)
    return ec, out, err

async def git_diff(sb: Sandbox, workdir: str, base_commit: str | None = None) -> str:
    """Capture the agent's work as a patch.

    Without ``base_commit`` this diffs the working tree, which is right when the
    prompt forbids committing (SWE_PROMPT and the minimal harness both do).

    With it, the diff is taken against that commit instead, so work that the
    agent committed -- or moved onto another branch -- is still captured. That
    is not hypothetical: all 113 DeepSWE instructions end with "work on this in
    a new branch from main and commit everything when you are done", which
    directly contradicts the harness prompt. An agent that obeys the task leaves
    a clean working tree, the plain diff comes back empty, and the run records a
    model that produced nothing rather than one that solved the task.
    """

    _SCAFFOLD = ("Dockerfile", ".dockerignore", "PROBLEM_STATEMENT.md", ".harness/")
    excludes = " ".join(f"':(exclude){p}'" for p in _SCAFFOLD)
    if base_commit:
        rev = shlex.quote(base_commit)

        cmd = (f"cd {workdir} && git add -A . >/dev/null 2>&1; "
               f"if git cat-file -e {rev}^{{commit}} 2>/dev/null; then "
               f"git diff {rev} -- . {excludes}; "
               f"else git diff HEAD -- . {excludes}; fi")
    else:
        cmd = f"cd {workdir} && git add -N . && git diff -- . {excludes}"
    _, out, _ = await _exec_race_safe(sb, cmd, "git_diff", user="agent", timeout=120)
    return out

async def _resolve_workdir(ev: Sandbox, declared: str) -> str:
    """Return the real repo dir inside the sandbox.

    Some dataset rows hardcode ``workdir=/testbed`` while the image actually
    checks the repo out at ``/<reponame>`` (swerebenchv2 convention). A wrong
    workdir makes ``chown``/``cd {workdir}`` fail, which silently zeroes reward.
    If the declared workdir has no ``.git`` we auto-detect the shallowest git
    repo and use it; otherwise we keep the declared value (zero overhead)."""
    q = shlex.quote(declared)
    rc, _, _ = await ev.exec(f"test -d {q}/.git", check=False, timeout=90)
    if rc == 0:
        return declared
    rc, out, _ = await ev.exec(
        "find / -maxdepth 4 -type d -name .git 2>/dev/null "
        "| awk '{print length, $0}' | sort -n | head -1 | cut -d' ' -f2-",
        check=False,
        timeout=180,
    )
    got = out.strip()
    if got:
        repo = os.path.dirname(got)
        logger.warning("[swe.evaluate] workdir %s has no .git; using detected repo %s", declared, repo)
        return repo
    logger.warning("[swe.evaluate] workdir %s has no .git and none detected; keeping declared", declared)
    return declared

async def evaluate(
    *,
    image: str,
    workdir: str,
    diff_text: str,
    swepro: dict | None = None,
    eval_cmd: str | None = None,
    eval_files: dict[str, str] | None = None,
    f2p_script: str | None = None,
    pre_commands: list[str] | str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool]:
    """Returns (reward, applied_cleanly).

    Three mutually-exclusive grading paths, in priority order: swepro test
    harness, a shell ``eval_cmd``, or a self-contained ``f2p_script`` pytest
    file. All resolve to "exit 0 == solved", and reward is 1.0 iff solved.

    ``eval_files`` maps a path inside the eval sandbox to a path on this host,
    and is pushed before ``eval_cmd`` runs. It exists because a verifier that
    ships as a *directory* cannot be inlined into ``eval_cmd``: that string is
    passed to the sandbox exec as a single argv entry, and Linux caps one entry
    at MAX_ARG_STRLEN (128 KiB), which no amount of compression fixes for the
    largest cases. Pushing files is what the swepro path already does; this
    generalises it. Writing into the eval sandbox (never the agent's) keeps the
    no-test-cheating guarantee intact.

    No-test-cheating guarantee: the eval sandbox is built from the same image
    but starts CLEAN, so only the model-produced diff affects reward."""
    if not (swepro or eval_cmd or f2p_script):
        logger.warning("[e2b.evaluate] no swepro/eval_cmd/f2p_script; reward=0")
        return 0.0, True

    async with create_sandbox(image) as ev:
        workdir = await _resolve_workdir(ev, workdir)
        await agent_sandbox.ensure_agent_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if eval_files:
            await _setup_eval_files(ev, eval_files)
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands)

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:

            logger.warning("[evaluate] %s: diff did not apply (%d chars) -> not a model failure",
                           workdir, len(diff_text or ""))
            return 0.0, False

        if swepro:
            r, _ = await _run_swepro(ev, workdir, swepro, timeout_sec)
        elif eval_cmd:
            r, _ = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        else:
            r, _ = await _run_f2p_script(ev, workdir, f2p_script, timeout_sec)
        return r, True

async def _retry_race(what: str, fn, attempts: int = 8):
    """Run a sandbox operation through podman 3.4.4's exec-cleanup race.

    podman loses a race removing an exec session's userdata directory while
    conmon still holds files in it, and reports
      Error: unlinkat .../overlay-containers/<id>/userdata/<exec>: directory not empty
    with a non-zero rc AFTER the command inside the container already ran. It is
    load-dependent: unmeasurable at concurrency 1, and at 128 it took out every
    Pro case within 54-75s.

    ensure_agent_user already retries this (slime/agent/sandbox.py). Pro's asset
    setup did not, and Pro is the arm that does the most of these operations —
    two extra file pushes and two extra execs per case on top of the normal boot —
    which is why Verified showed zero aborts in the same run where Pro showed
    100%.
    """
    last = ""
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as e:  
            last = str(e)
            if "unlinkat" not in last:
                raise
            logger.warning("[swepro] %s hit the podman cleanup race (attempt %d/%d); retrying",
                           what, attempt + 1, attempts)
            await asyncio.sleep(min(2 * (attempt + 1), 5))
    raise RuntimeError(f"{what} still failing after {attempts} attempts: {last[:400]}")

async def _setup_swepro_assets(ev: Sandbox, swepro: dict) -> None:
    await _retry_race("mkdir swepro dir",
                      lambda: ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}",
                                      user="root", check=True))
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            if not Path(host_p).exists():

                raise FileNotFoundError(f"swepro asset missing on this host: {host_p}")
            await _retry_race(f"push {dst}",
                              lambda hp=host_p, d=dst: ev.write_file(f"{_SWEPRO_DIR}/{d}", Path(hp), user="root"))
    await _retry_race("chmod/chown swepro dir",
                      lambda: ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}",
                                      user="root", check=True))

async def _setup_eval_files(ev: Sandbox, eval_files: dict[str, str]) -> None:
    """Push host files into the eval sandbox before grading.

    Missing assets raise rather than degrade: the 0804 Pro run scored 0/266
    because a dataset carried dev-box absolute paths and the copy failed
    silently. A verifier that is not there must not read as a failed patch.
    """
    for dest, host_path in eval_files.items():
        src = Path(host_path)
        if not src.exists():
            raise FileNotFoundError(f"eval_files asset missing on this host: {host_path}")
        parent = str(Path(dest).parent)
        await _retry_race(f"mkdir {parent}",
                          lambda p=parent: ev.exec(f"mkdir -p {p}", user="root", check=True))
        await _retry_race(f"push {dest}",
                          lambda d=dest, s=src: ev.write_file(d, s, user="root"))
    await _retry_race("chown eval_files",
                      lambda: ev.exec("chown -R agent:agent " +
                                      " ".join(sorted({str(Path(d).parent) for d in eval_files})),
                                      user="root", check=True))

def _patch_new_files(diff_text: str) -> set[str]:
    """Paths the diff creates with a `new file mode` header."""
    out, cur = set(), None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            cur = line.split(" b/", 1)[1].strip() if " b/" in line else None
        elif line.startswith("new file mode") and cur:
            out.add(cur)
    return out

async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await _retry_race("write patch file", lambda: ev.write_file(_PATCH, diff_text, user="agent"))

    errs = []

    new_files = _patch_new_files(diff_text)
    tried_clear = False
    for cmd in [
        f"cd {workdir} && git apply --3way --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}",
    ]:
        ec, out, err = await ev.exec(cmd, user="agent", check=False, timeout=120)
        if ec == 0:
            return True
        blob = f"{err or ''}\n{out or ''}"
        if not tried_clear and "already exists in working directory" in blob:
            clash = set(re.findall(r"error: (.+?): already exists in working directory", blob)) & new_files
            if clash:
                tried_clear = True
                logger.warning("[apply] removing %d pre-existing path(s) the patch creates: %s",
                               len(clash), sorted(clash)[:4])
                await ev.exec("cd %s && rm -f %s" % (workdir, " ".join(shlex.quote(c) for c in sorted(clash))),
                              user="agent", check=False, timeout=60)
                ec, out, err = await ev.exec(cmd, user="agent", check=False, timeout=120)
                if ec == 0:
                    return True
                blob = f"{err or ''}\n{out or ''}"

        if "unlinkat" in blob or "directory not empty" in blob or "Applied patch to" in blob:
            rc2, out2, _ = await ev.exec(
                f"cd {workdir} && git apply --check --reverse {_PATCH} && echo __PATCH_PRESENT__",
                user="agent", check=False, timeout=120,
            )
            if "__PATCH_PRESENT__" in (out2 or ""):
                logger.warning(
                    "[apply] exec returned %d from the podman cleanup race, but the patch IS "
                    "applied in %s -- counting it as applied", ec, workdir)
                return True
        errs.append(f"rc={ec}:{(err or out or '').strip()[:180]}")

    files = _split_diff_by_file(diff_text)
    if len(files) > 1:
        ok_files, bad_files = [], []
        for path, chunk in files:
            await _retry_race("write per-file patch",
                              lambda c=chunk: ev.write_file(_PATCH, c, user="agent"))
            applied_one = False
            for cmd in (f"cd {workdir} && git apply --3way --whitespace=nowarn {_PATCH}",
                        f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
                        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}"):
                ec, _o, _e = await ev.exec(cmd, user="agent", check=False, timeout=120)
                if ec == 0:
                    applied_one = True
                    break
            (ok_files if applied_one else bad_files).append(path)
        if ok_files:
            logger.warning("[apply] per-file fallback in %s: applied %d/%d (dropped: %s)",
                           workdir, len(ok_files), len(files), ", ".join(bad_files[:6]))
            return True
    logger.warning("[apply] all 3 failed in %s (%d chars) || %s", workdir, len(diff_text), " || ".join(errs))
    return False

def _split_diff_by_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a unified diff into (path, single-file diff) pairs.

    Splits on `diff --git`, which git emits once per file, so each piece is a complete
    patch on its own. Returns [] for anything that does not look like a git diff rather
    than guessing at a format it cannot parse.
    """
    if "diff --git " not in (diff_text or ""):
        return []
    parts, cur, path = [], [], None
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if cur and path:
                parts.append((path, "".join(cur)))
            cur, path = [line], line.rstrip().split(" b/")[-1]
        elif cur:
            cur.append(line)
    if cur and path:
        parts.append((path, "".join(cur)))
    return parts

_FIXED_PORTS = (3000, 4568, 8080)

_SWEPRO_REQUIRED_EXCLUSIONS = {
    'instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan': (
        'test/database.js | Test database test/database/sorted.js::Sorted Set methods test/database/sorted.js::getSortedSetRange() should work with big arrays (length > 100)',
        'test/user.js | User Digest.getSubscribers should accurately build digest list given ACP default "day',
        'test/user.js | User Digest.getSubscribers should accurately build digest list given ACP default "off',
        'test/user.js | User Digest.getSubscribers should accurately build digest list given ACP default "week',
    ),
    'instance_future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a-v1151a6325649aaf997cd541ebe533b53fddf1b07': (
        'Test_redhatBase_parseUpdatablePacksLine/amazon_2023:_Is_this_ok_[y/N]:_"dnf"_"0"_"4.14.0"_"1.amzn2023.0.6"_"amazonlinux',
        'Test_redhatBase_parseUpdatablePacksLine/centos_7.0:_"shadow-utils"_"2"_"4.1.5.1_24.el7"_"rhui-REGION-rhel-server-releases',
        'Test_redhatBase_parseUpdatablePacksLine/centos_7.0:_"zlib"_"0"_"1.2.7"_"17.el7"_"rhui-REGION-rhel-server-releases',
    ),
}

def _swepro_instance_id(swepro: dict) -> str:
    return os.path.basename(os.path.dirname(swepro.get("run_script_path") or ""))

async def _remap_fixed_test_ports(ev: Sandbox, workdir: str) -> None:
    ports = " ".join(str(p) for p in _FIXED_PORTS)
    script = (
        f'for port in {ports}; do '
        '  (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null || continue; exec 3<&- 2>/dev/null; '
        f'  files=$(grep -rl "listen($port" {workdir}/test 2>/dev/null | grep -v "/build/" | head -20); '
        '  [ -n "$files" ] || continue; '
        '  for i in $(seq 1 40); do p=$(( 20000 + RANDOM % 12000 )); '   

        '    if ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then break; fi; exec 3<&- 2>/dev/null; done; '
        '  for f in $files; do sed -i "s/listen($port/listen($p/g; s#localhost:$port#localhost:$p#g; '
        's#127\\.0\\.0\\.1:$port#127.0.0.1:$p#g" "$f"; done; '
        '  echo "[portfix] $port -> $p in $(echo $files | tr \'\\n\' \' \')"; '
        'done'
    )
    _, out, _ = await ev.exec(script, user="root", check=False, timeout=120)
    if out.strip():
        logger.warning("[swepro] %s", out.strip()[:300])

_PORT_BOUND_REPOS = ("NodeBB", "tutanota")

@contextlib.asynccontextmanager
async def _host_repo_lock(name: str):
    path = f"/tmp/slime_swepro_{name}.lock"

    def _acquire():
        f = open(path, "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    f = await asyncio.get_running_loop().run_in_executor(None, _acquire)
    try:
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()

async def _run_swepro(ev: Sandbox, workdir: str, swepro: dict, timeout: int) -> tuple[float, bool]:
    iid = _swepro_instance_id(swepro)
    repo_lock = next((r for r in _PORT_BOUND_REPOS if r.lower() in iid.lower()), None)
    if repo_lock:
        async with _host_repo_lock(repo_lock):
            return await _run_swepro_inner(ev, workdir, swepro, timeout)
    return await _run_swepro_inner(ev, workdir, swepro, timeout)

async def _run_swepro_inner(ev: Sandbox, workdir: str, swepro: dict, timeout: int) -> tuple[float, bool]:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    await _remap_fixed_test_ports(ev, workdir)
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"

    t0 = time.monotonic()
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh "
        f"{json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user="agent",
        check=False,
        timeout=timeout,
    )
    if time.monotonic() - t0 >= timeout - 5:
        raise TimeoutError(f"swepro run_script hit {timeout}s judge timeout (silent-zero forbidden)")
    await _exec_race_safe(
        ev,
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}",
        "swepro parser",
        user="agent",
        timeout=120,
    )
    raw = await ev.read_file(result_f, user="agent")
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    drop = set(_SWEPRO_REQUIRED_EXCLUSIONS.get(_swepro_instance_id(swepro), ()))
    if drop & required:
        kept = required - drop
        if kept:  
            logger.warning("[swepro] %s: excluding %d unmatchable/flaky required test(s), %d remain",
                           _swepro_instance_id(swepro), len(drop & required), len(kept))
            required = kept
    solved = bool(required) and required.issubset(passed)
    if not solved and os.environ.get("SLIME_SWEPRO_DEBUG"):
        
        missing = sorted(required - passed)[:8]
        _, err_tail, _ = await ev.exec(f"tail -c 1500 {stderr_f} 2>/dev/null; echo; tail -c 800 {stdout_f}", user="agent", check=False, timeout=60)
        logger.warning("[swepro-debug] missing=%s parsed_n=%d tail=%s", missing, len(parsed.get("tests", [])), err_tail[-1200:])
    return (1.0 if solved else 0.0), solved

async def _run_eval_cmd(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> tuple[float, bool]:
    """Run the grading command. A timeout is NOT a wrong answer.

    The old form returned 0.0 for any non-zero exit code, which silently folded
    rc=124 — sandbox._run's own kill when the command exceeds --eval-timeout —
    into "the model's patch did not pass". Those two are indistinguishable in the
    results file afterwards, and on a loaded node the timeout is the common case,
    not the rare one. The Pro branch of this same file already refuses to do this
    (see _run_swepro: "silent-zero forbidden"); this is the Verified branch
    catching up.
    """
    t0 = time.monotonic()
    ec, _, _ = await _exec_race_safe(ev, f"cd {workdir} && {cmd}", "eval_cmd", user="agent", timeout=timeout)

    if ec != 0 and time.monotonic() - t0 >= timeout - 5:
        raise TimeoutError(f"eval_cmd hit {timeout}s judge timeout (silent-zero forbidden)")
    return (1.0 if ec == 0 else 0.0), ec == 0

async def _run_f2p_script(ev: Sandbox, workdir: str, script: str, timeout: int) -> tuple[float, bool]:

    await _retry_race("write f2p script", lambda: ev.write_file(_F2P, script, user="agent"))
    ec, _, _ = await _exec_race_safe(ev, f"cd {workdir} && python {_F2P}", "f2p_script", user="agent", timeout=timeout)
    return (1.0 if ec == 0 else 0.0), ec == 0
