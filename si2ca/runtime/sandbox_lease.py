"""Ownership leases that let the sandbox host reclaim orphaned containers.

Every sandbox slime creates is stamped with the rollout that owns it

    --label slime-rollout-key=<run_id>.<rollout_id>

and every live rollout publishes a *lease file* it keeps touching for as long as
it runs. Put together, the sandbox host can answer the one question a state
filter cannot: does this container still have an owner?

    label present, lease fresh    -> a rollout is still using it, leave it alone
    label present, lease gone     -> the rollout finished, this one leaked
    label present, lease stale    -> the rollout process died, this one leaked
    no label                      -> not ours to judge, never touched

The lease is a file rather than a dict in this process because the most common
way sandboxes are orphaned is the process holding the dict dying — an in-memory
map disappears exactly when it is needed. Ownership rides on the container (a
label) and liveness rides on the filesystem (a lease), so both outlive us.

Enabled only for the ``local_docker`` backend; e2b sandboxes are reaped by the
gateway's own lifetime. Every entry point here is failure-tolerant: if the lease
directory is unwritable we publish nothing, containers go out unlabeled, and the
reaper ignores them — i.e. we degrade to exactly the old behavior rather than
risking a false kill.

Reaper side: ``configs/sandbox_reaper.py`` (keep ``LABEL`` in sync).
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL = "slime-rollout-key"

backend_env = ("SLIME_AGENT_SANDBOX_BACKEND", "SWE_SANDBOX_BACKEND")
lease_dir_env = "SLIME_SANDBOX_LEASE_DIR"

default_lease_dir = "/scratch/slime_sandbox_leases"

touch_interval_sec = 60

_ROLLOUT_KEY_ENV = "SLIME_SANDBOX_ROLLOUT_KEY"
_RUN_ID_ENV = "SLIME_SANDBOX_RUN_ID"

publisher_marker = ".publisher"

_lock = threading.Lock()
_held: set[str] = set()
_toucher: threading.Thread | None = None

def enabled() -> bool:
    for name in backend_env:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip().lower() == "local_docker"
    return False

def run_id() -> str:
    """Identifier for this training run, stable for the life of the process.

    Rollout ids restart from 0 on every job, so they cannot identify an owner on
    their own: yesterday's rollout 5 and today's rollout 5 would share a key and
    yesterday's leaked containers would look alive again. The run id makes a key
    unique to one process, which also means a resumed job never adopts the
    previous attempt's containers — it reaps them.
    """
    rid = os.environ.get(_RUN_ID_ENV)
    if not rid:
        rid = uuid.uuid4().hex[:8]
        os.environ[_RUN_ID_ENV] = rid
    return rid

def lease_dir() -> Path:
    return Path(os.environ.get(lease_dir_env) or default_lease_dir)

def current_key() -> str:
    """The rollout key to stamp on containers created right now ('' when idle).

    Read by ``LocalDockerSandbox.__aenter__``. This is an env var rather than a
    module global so it also reaches sandboxes created from a forked child.
    """
    return os.environ.get(_ROLLOUT_KEY_ENV, "")

def _lease_path(key: str) -> Path:
    return lease_dir() / f"{key}.lease"

def _touch(key: str) -> bool:
    """Publish/refresh the lease for ``key``. False if the lease dir is unusable."""
    path = _lease_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(f"{int(time.time())} pid={os.getpid()} key={key}\n")
        return True
    except OSError as e:
        logger.warning("[sandbox_lease] cannot publish lease %s: %s — sandboxes stay unlabeled", path, e)
        return False

def _mark_publisher() -> None:
    """Record that leases for this host's sandboxes are published *here*."""
    try:
        (lease_dir() / publisher_marker).write_text(f"{int(time.time())} pid={os.getpid()} run={run_id()}\n")
    except OSError as e:
        logger.warning("[sandbox_lease] cannot mark lease dir %s: %s", lease_dir(), e)

def _touch_loop() -> None:
    while True:
        time.sleep(touch_interval_sec)
        with _lock:
            keys = list(_held)
        for key in keys:
            _touch(key)

def _start_toucher() -> None:
    global _toucher
    if _toucher is None or not _toucher.is_alive():
        _toucher = threading.Thread(target=_touch_loop, name="sandbox-lease-toucher", daemon=True)
        _toucher.start()

def begin(rollout_id: int, *, evaluation: bool = False) -> str:
    """Open a lease for this rollout and return the key to label its sandboxes with.

    Returns '' when leasing is off or unavailable, in which case sandboxes are
    created unlabeled and the reaper leaves them alone.
    """
    if not enabled():
        return ""
    key = f"{run_id()}.{'e' if evaluation else ''}{rollout_id}"

    if not _touch(key):
        return ""
    _mark_publisher()
    with _lock:
        _held.add(key)
        _start_toucher()
    os.environ[_ROLLOUT_KEY_ENV] = key
    return key

def end(key: str) -> int:
    """Close the lease and reclaim any sandbox this rollout left behind.

    Returns the number of containers removed — normally 0, because a healthy
    rollout tears its own sandboxes down; anything else is a leak we just caught.
    """
    if not key:
        return 0
    if current_key() == key:
        os.environ.pop(_ROLLOUT_KEY_ENV, None)
    with _lock:
        _held.discard(key)

    with contextlib.suppress(OSError):
        _lease_path(key).unlink(missing_ok=True)
    return sweep(key)

def sweep(key: str) -> int:
    """kill + rm every container labeled with ``key``. Returns the number removed."""
    removed = 0
    for cmd in _engine_cmds():
        rc, out = _capture([*cmd, "ps", "-aq", "--filter", f"label={LABEL}={key}"], timeout=60)
        if rc != 0 or not out.strip():
            continue
        for cid in out.split():

            rc_rm, _ = _capture([*cmd, "rm", "-f", cid], timeout=60)
            if rc_rm != 0:
                _capture([*cmd, "kill", cid], timeout=30)
                rc_rm, _ = _capture([*cmd, "rm", "-f", cid], timeout=60)
            if rc_rm == 0:
                removed += 1
            else:
                logger.warning("[sandbox_lease] could not reclaim %s (rollout %s)", cid[:12], key)
    return removed

def _engine_cmds() -> list[list[str]]:

    from si2ca.runtime.sandbox import LocalDockerSandbox

    hosts = LocalDockerSandbox._docker_hosts()
    return [["podman", "--remote", "--url", h] for h in hosts] if hosts else [["docker"]]

def _capture(argv: list[str], *, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("[sandbox_lease] %s failed: %s", argv[0], e)
        return 1, ""
    return proc.returncode, proc.stdout

@contextlib.contextmanager
def rollout_lease(rollout_id: int, *, evaluation: bool = False):
    """Hold a lease for one rollout step, sweeping its sandboxes on the way out.

    Wraps the whole rollout so that the guarantee "a finished rollout owns no
    containers" holds even when the step raises. Bookkeeping failures are logged
    and swallowed — cleanup must never be the reason a training step dies.
    """
    key = ""
    try:
        key = begin(rollout_id, evaluation=evaluation)
    except Exception as e:
        logger.warning("[sandbox_lease] could not open lease for rollout %s: %s", rollout_id, e)
    try:
        yield key
    finally:
        try:
            leaked = end(key)
            if leaked:
                logger.warning("[sandbox_lease] rollout %s ended with %d sandbox(es) still up — reclaimed", key, leaked)
        except Exception as e:
            logger.warning("[sandbox_lease] could not close lease %s: %s", key, e)
