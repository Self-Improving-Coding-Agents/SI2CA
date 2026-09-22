#!/usr/bin/env python3
"""Run multilingual tests in a sandbox and parse their logs on the host.

Why parse on the host?
----------------------
The initial design copied a standalone swebench log parser into the sandbox
and used eval_cmd's exit code as the verdict. It assumed every image contained
python3. Some Java images did not, so rc=127 became a false zero reward on five
gold-patch checks: the grader had never actually started.

Follow swebench's host-side parsing design instead. The sandbox runs tests
using its own toolchain and produces logs; the host uses the installed swebench
package to parse them. This removes standalone parser packaging, AST gates,
and test_spec substitutes by using log_parsers and make_test_spec directly.

The grading sandbox starts clean from the same image. The official eval_script
resets test files before execution, so model edits to tests do not affect the
verdict.
"""
from __future__ import annotations

import logging
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)

START_MARK = ">>>>> Start Test Output"
END_MARK = ">>>>> End Test Output"
SANDBOX_DIR = "/tmp/_ml_eval"

def _as_list(v) -> list[str]:
    import json
    if isinstance(v, list):
        return list(v)
    if isinstance(v, str) and v.strip():
        s = v.strip()
        return list(json.loads(s)) if s.startswith("[") else [s]
    return []

def grade_log(raw: str, parser_name: str, instance_id: str,
              f2p: list[str], p2p: list[str]) -> tuple[float, dict]:
    """Require all fail-to-pass and pass-to-pass tests to pass.

    Ignore the test command's exit code: cargo, Maven, or go test may return
    nonzero for unrelated pre-existing failures, not failure on the target task.
    """
    from types import SimpleNamespace

    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    import swebench.harness.log_parsers as lp

    parser = None
    for mod in (lp.c, lp.go, lp.java, lp.javascript, lp.php, lp.python, lp.ruby, lp.rust):
        parser = getattr(mod, parser_name, None)
        if parser is not None:
            break
    if parser is None:
        parser = MAP_REPO_TO_PARSER.get(instance_id.split("__")[0])
    if parser is None:
        return 0.0, {"error": f"No parser named {parser_name}"}

    if START_MARK in raw and END_MARK in raw:
        body = raw.split(START_MARK, 1)[1].rsplit(END_MARK, 1)[0]
        truncated = False
    else:

        body, truncated = raw, True

    try:
        status = parser(body, SimpleNamespace(instance_id=instance_id))
    except Exception as exc:  
        return 0.0, {"error": f"Parser raised {type(exc).__name__}: {exc}",
                     "markers_missing": truncated}

    bad_f2p = [t for t in f2p if status.get(t) != "PASSED"]
    bad_p2p = [t for t in p2p if status.get(t) != "PASSED"]
    solved = bool(f2p) and not bad_f2p and not bad_p2p
    detail = {
        "n_parsed": len(status),
        "f2p_passed": len(f2p) - len(bad_f2p), "f2p_total": len(f2p),
        "p2p_passed": len(p2p) - len(bad_p2p), "p2p_total": len(p2p),
        "f2p_failing": [(t, status.get(t, "missing")) for t in bad_f2p[:5]],
        "p2p_failing": [(t, status.get(t, "missing")) for t in bad_p2p[:5]],
        "markers_missing": truncated,
    }
    return (1.0 if solved else 0.0), detail

async def evaluate_multilingual(*, image: str, workdir: str, diff_text: str,
                                metadata: dict, timeout_sec: int = 1800) -> tuple[float, bool, dict]:
    """Return (reward, applied_cleanly, grading_details).

    The first two values match swe.evaluate. Details distinguish unsolved
    tasks from grading failures so a zero reward remains interpretable.
    """
    from si2ca.runtime import swe
    from si2ca.runtime import sandbox as agent_sandbox

    iid = metadata["instance_id"]
    assets = metadata.get("eval_assets") or {}
    body_path = assets.get("eval_body")
    if not body_path or not Path(body_path).exists():
        raise FileNotFoundError(f"{iid}: eval_body.sh is missing on the host: {body_path}")

    async with swe.create_sandbox(image) as ev:
        wd = await swe._resolve_workdir(ev, workdir)
        await agent_sandbox.ensure_agent_user(ev, wd)

        await ev.exec(f"mkdir -p {SANDBOX_DIR}", user="root", check=True)
        await ev.write_file(f"{SANDBOX_DIR}/eval_body.sh", Path(body_path), user="root")
        await ev.exec(f"chown -R agent:agent {SANDBOX_DIR}", user="root", check=True)

        applied = await swe._apply_diff(ev, wd, diff_text)
        if not applied:
            return 0.0, False, {"error": "Patch could not be applied"}

        await swe._exec_race_safe(
            ev, f"cd {wd} && bash {SANDBOX_DIR}/eval_body.sh > {SANDBOX_DIR}/test.log 2>&1; true",
            "eval_body", user="agent", timeout=timeout_sec)

        _, raw, _ = await swe._exec_race_safe(
            ev, f"tail -c 4000000 {shlex.quote(SANDBOX_DIR)}/test.log 2>/dev/null || true",
            "read_log", user="agent", timeout=300)

    if not (raw or "").strip():
        return 0.0, True, {"error": "Test log is empty (tests did not start in the sandbox)"}

    reward, detail = grade_log(raw, metadata["log_parser"], iid,
                               _as_list(metadata.get("f2p")), _as_list(metadata.get("p2p")))
    return reward, True, detail
