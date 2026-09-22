#!/usr/bin/env python3
"""Turn each instance's golden patch into a FIXED-LENGTH hint block.

Why this exists. The head arm inserts the golden patch between the problem
statement and the trajectory prefix, and the patch's length is not a controlled
variable: 116 to 12,000+ characters across this pool, and strongly correlated
with which benchmark the instance came from (Verified median 882, Pro 6,169).
Measured over the head arm's 245,992 candidates, the scoring perturbation grows
with that length:

    gold block chars   mean delta      "answer helped" share
      0-500            +0.0084         34.5%
      500-1500         +0.0086         34.6%
      1500-4000        +0.0122         32.1%
      4000-12000       +0.0187         29.3%
      12000+           +0.0158         29.7%   (capped by truncation)

So "head helps on small patches and hurts on large ones" is confounded: large
patches are Pro instances AND they perturb the scoring twice as much. Replacing
the patch with a hint block of near-constant length removes the length axis and
leaves the information axis.

Target: 4 hints, 700-1100 characters, aiming at ~900 -- the floor of the
perturbation curve (0-500 and 500-1500 are indistinguishable, so going shorter
buys nothing and loses information).
"""
from __future__ import annotations

from si2ca.prompts import PROMPT_DIR, MESSAGES

import argparse, asyncio, json, os, re, sys, time
from pathlib import Path

import httpx

TEMPLATE = (PROMPT_DIR / "hint_generation.md").read_text(encoding="utf-8")
LO, HI, AIM = 700, 1100, 900
# One retry with an explicit correction, then a sentence-boundary trim. A block
# that is still out of band after that is recorded with its length so the
# achieved distribution can be reported instead of silently assumed.
RETRY_NOTE = (MESSAGES['hints']['retry'])


def trim_to_sentences(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    lines, out = text.splitlines(), []
    for ln in lines:
        if len("\n".join(out + [ln])) > cap:
            break
        out.append(ln)
    kept = "\n".join(out).rstrip()
    return kept if len(kept) >= LO else text[:cap].rsplit(".", 1)[0].rstrip() + "."


async def one(client, sem, md, out_fh, lock, stats):
    iid = md["instance_id"]
    prompt = (TEMPLATE
              .replace("{{problem_statement}}", (md.get("problem_statement") or "")[:20000])
              .replace("{{gold_patch}}", (md.get("gold_patch") or "")[:40000]))
    async with sem:
        text, attempts = "", 0
        for attempt in range(2):
            body = {"model": args.model, "messages": [{"role": "user", "content": prompt}]}
            for retry in range(4):
                try:
                    r = await client.post(args.url, json=body, timeout=900)
                    r.raise_for_status()
                    text = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                    break
                except Exception as e:  # noqa: BLE001 -- one instance must not kill the sweep
                    if retry == 3:
                        text = ""
                        stats["error"] = stats.get("error", 0) + 1
                    else:
                        await asyncio.sleep(20 * (retry + 1))
            attempts = attempt + 1
            if not text or LO <= len(text) <= HI:
                break
            prompt = prompt + RETRY_NOTE % (len(text), "above" if len(text) > HI else "below", LO, HI, AIM)
        if not text:
            stats["failed"] = stats.get("failed", 0) + 1
            return
        if len(text) > HI:
            text = trim_to_sentences(text.strip(), HI)
        rec = {"instance_id": iid, "hint": text.strip(), "hint_chars": len(text.strip()),
               "gold_chars": len(md.get("gold_patch") or ""), "attempts": attempts,
               "in_band": LO <= len(text.strip()) <= HI}
        async with lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_fh.flush()
            stats["done"] = stats.get("done", 0) + 1
            if stats["done"] % 25 == 0:
                print(f"[hints] {stats['done']} done, band={stats.get('done',0)-stats.get('oob',0)}", flush=True)
        if not rec["in_band"]:
            stats["oob"] = stats.get("oob", 0) + 1


async def main():
    rows = []
    for line in open(args.dataset):
        line = line.strip()
        if line:
            rows.append(json.loads(line)["metadata"])
    have = set()
    if args.resume and Path(args.out).exists():
        for line in open(args.out):
            try: have.add(json.loads(line)["instance_id"])
            except Exception: pass
    todo = [m for m in rows if m["instance_id"] not in have]
    print(f"[hints] pool={len(rows)} already={len(have)} todo={len(todo)} conc={args.concurrency}", flush=True)
    sem, lock, stats = asyncio.Semaphore(args.concurrency), asyncio.Lock(), {}
    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "a", encoding="utf-8")
    try:
        async with httpx.AsyncClient(headers={"Authorization": f"Bearer {os.environ.get('SI2CA_API_KEY', 'dummy')}"}) as client:
            await asyncio.gather(*[one(client, sem, m, fh, lock, stats) for m in todo])
    finally:
        fh.close()
    print(f"[hints] done={stats.get('done',0)} out_of_band={stats.get('oob',0)} "
          f"failed={stats.get('failed',0)} errors={stats.get('error',0)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.6-sol_2026-07-09")
    ap.add_argument("--url", default="http://127.0.0.1:8799/v1/chat/completions")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    args = ap.parse_args()
    asyncio.run(main())
