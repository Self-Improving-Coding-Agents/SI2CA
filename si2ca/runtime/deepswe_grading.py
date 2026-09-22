"""DeepSWE in-place grading used by the unified execution loop."""

from __future__ import annotations

import json
import time
from pathlib import Path


async def grade(sb, meta: dict) -> tuple[float, str, dict]:
    """Grade in place: copy four eval_files, run eval_cmd, then read reward.json.

    Distinguish grading infrastructure failures from genuine zero rewards:
        OK           Grading completed; the reward is valid.
        GRADEFAIL:*  Missing assets, copy failures, or verifier failures.
        TIMEOUT      The verifier exhausted its budget (normally 1,800 seconds).
    """
    ds = meta.get("deepswe") or {}
    vt = int(ds.get("verifier_timeout_sec") or 1800)
    workdir = meta.get("workdir") or "/app"

    await sb.exec("mkdir -p /tests /logs/verifier /logs/artifacts", user="root", check=False, timeout=60)
    for dest, host_path in (meta.get("eval_files") or {}).items():
        src = Path(host_path)

        if not src.is_file():
            return 0.0, f"GRADEFAIL:ASSET_MISSING:{host_path}", {}
        await sb.exec(f"mkdir -p $(dirname {dest})", user="root", check=False, timeout=60)
        try:
            await sb.write_file(dest, src.read_bytes(), user="root")
        except Exception as e:
            return 0.0, f"GRADEFAIL:COPY:{dest}:{type(e).__name__}", {}

    eval_cmd = meta["eval_cmd"].replace("/logs/verifier/test_stdout.log",
                                        "/logs/verifier/test-stdout.txt")
    t0 = time.monotonic()
    rc, out, err = await sb.exec(f"cd {workdir} && {eval_cmd}",
                                 user="root", check=False, timeout=vt + 120)
    ran = time.monotonic() - t0

    if rc != 0 and ran >= vt - 5:
        return 0.0, f"TIMEOUT:{int(ran)}s", {}

    _, rjson, _ = await sb.exec("cat /logs/verifier/reward.json 2>/dev/null || true",
                                user="root", check=False, timeout=60)
    s = (rjson or "").strip()
    if not s:
        return 0.0, f"GRADEFAIL:NO_REWARD_JSON:rc={rc}:{(err or out or '')[-200:]}", {}
    try:
        d = json.loads(s)
    except Exception:
        return 0.0, f"GRADEFAIL:BAD_JSON:{s[:120]}", {}
    detail = {k: d.get(k) for k in ("f2p_total", "f2p_passed", "p2p_total", "p2p_passed") if k in d}
    return float(int(d.get("reward", 0))), "OK", detail
