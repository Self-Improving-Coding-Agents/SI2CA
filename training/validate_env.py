#!/usr/bin/env python3
"""Certify SWE task environments through the EXACT reward path (swe.evaluate).

A task row is VALID iff both hold in a fresh sandbox of its image:
  bug  : empty diff        -> reward 0   (the F2P tests really fail before the fix;
                                          otherwise an idle agent gets a free 1.0)
  gold : diff=gold_patch   -> reward 1   (the environment can still award 1.0;
                                          otherwise the task is env-rot, not "hard")

Both runs go through ``swe.evaluate`` (pre_commands + agent user + the 3way->plain->
patch apply chain + eval_cmd), i.e. the same code that grades rollouts, so a
verdict here transfers 1:1 to training/eval time. This merges the two one-sided
probes that already exist (clean_baseline_pass.py = bug side, probe_env_gold.py =
gold side) into one resumable pass; every row costs two container boots.

Verdicts:
  VALID            bug=0 and gold=1
  LEAKY            bug=1                (baseline already passes -> drop)
  GOLD_FAIL        bug=0 and gold=0     (env-rot / stale gold -> drop, or re-probe serially)
  GOLD_NOT_APPLIED gold patch did not apply
  ERROR            boot/pull/timeout exception (re-run; not a verdict on the task)

Usage:
  SLIME_AGENT_SANDBOX_BACKEND=local_docker \
  python examples/coding_agent_rl/validate_task_env.py \
      --input training_data/swe_selfimprove_pool_v0/tierB.jsonl \
      --results training_data/swe_selfimprove_pool_v0/validation_tierB.jsonl \
      --concurrency 8 --timeout-sec 900 [--limit N] [--candidates ids.txt] [--rmi]

Do NOT use SLIME_AGENT_DOCKER_NETWORK=host on a shared machine (host-port
collisions masquerade as flaky tests); the default bridge network is enough for
grading. Images are pulled on first ``docker run`` (needs the DockerHub Pro
login already present in ~/.docker/config.json). ``--rmi`` removes each image
after its verdict to keep disk bounded (re-pull later for generation).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from si2ca.runtime import swe


def _rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _done(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for r in _rows(path):
            # an ERROR row is not a verdict: let a re-run retry it
            if r.get("verdict") != "ERROR":
                out[r["instance_id"]] = r
    return out


async def _grade(md: dict, diff: str, timeout_sec: int) -> tuple[float | None, bool | None, str | None]:
    hard = timeout_sec + int(os.environ.get("SWE_PROBE_BOOT_BUDGET", "600"))
    try:
        r, applied = await asyncio.wait_for(
            swe.evaluate(image=md["image"], workdir=md["workdir"], diff_text=diff,
                         eval_cmd=md.get("eval_cmd"), pre_commands=md.get("pre_commands"),
                         swepro=md.get("swepro"), timeout_sec=timeout_sec),
            timeout=hard)
        return float(r), bool(applied), None
    except Exception as e:  # boot / pull / timeout -> ERROR (retryable)
        return None, None, f"{type(e).__name__}: {e}"[:400]


async def _one(row, *, sem, timeout_sec, rmi, fh, lock, counters, total):
    md = row["metadata"]
    iid = md["instance_id"]
    rec = {"instance_id": iid, "image": md["image"]}
    t0 = time.time()
    async with sem:
        bug_r, _, bug_err = await _grade(md, "", timeout_sec)
        rec.update(bug_reward=bug_r, bug_error=bug_err)
        if bug_err is None and bug_r == 0.0:
            gold_r, gold_applied, gold_err = await _grade(md, md.get("gold_patch") or "", timeout_sec)
            rec.update(gold_reward=gold_r, gold_applied=gold_applied, gold_error=gold_err)
        if rmi:
            p = await asyncio.create_subprocess_exec("docker", "rmi", "-f", md["image"],
                                                     stdout=asyncio.subprocess.DEVNULL,
                                                     stderr=asyncio.subprocess.DEVNULL)
            try:
                await asyncio.wait_for(p.wait(), timeout=120)
            except Exception:
                pass
    rec["elapsed"] = round(time.time() - t0, 1)
    if rec.get("bug_error"):
        v = "ERROR"
    elif rec["bug_reward"] == 1.0:
        v = "LEAKY"
    elif rec.get("gold_error"):
        v = "ERROR"
    elif rec.get("gold_applied") is False:
        v = "GOLD_NOT_APPLIED"
    elif rec.get("gold_reward") == 1.0:
        v = "VALID"
    else:
        v = "GOLD_FAIL"
    rec["verdict"] = v
    async with lock:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        counters["done"] += 1
        counters[v] = counters.get(v, 0) + 1
        n = counters["done"]
        if n % 10 == 0 or n == total:
            print(f"[{n}/{total}] " + " ".join(f"{k}={counters[k]}" for k in sorted(counters) if k != "done")
                  + f" (last {iid} {rec['elapsed']}s {v})", file=sys.stderr, flush=True)


async def _run(a):
    rows = list(_rows(Path(a.input)))
    if a.candidates:
        keep = {l.strip() for l in open(a.candidates) if l.strip()}
        rows = [r for r in rows if r["metadata"]["instance_id"] in keep]
    results = Path(a.results)
    done = _done(results)
    pending = [r for r in rows if r["metadata"]["instance_id"] not in done]
    if a.limit is not None:
        pending = pending[: a.limit]
    print(f"[validate] rows={len(rows)} done={len(done)} pending={len(pending)} conc={a.concurrency} "
          f"timeout={a.timeout_sec}s rmi={a.rmi}", file=sys.stderr, flush=True)
    sem, lock, counters = asyncio.Semaphore(a.concurrency), asyncio.Lock(), {"done": 0}
    with results.open("a", encoding="utf-8") as fh:
        await asyncio.gather(*[_one(r, sem=sem, timeout_sec=a.timeout_sec, rmi=a.rmi, fh=fh, lock=lock,
                                    counters=counters, total=len(pending)) for r in pending])
    print(f"[validate DONE] {counters}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--candidates", default=None, help="restrict to instance_ids listed in this file")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout-sec", type=int, default=900, help="per eval_cmd run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rmi", action="store_true", help="docker rmi the image after its verdict")
    a = ap.parse_args()
    os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "local_docker")
    asyncio.run(_run(a))


if __name__ == "__main__":
    main()
