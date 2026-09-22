#!/usr/bin/env python3
"""Build the self-improvement data-construction pool (v0, 2026-09-01).

Goal: a ~10k-task subset of the python SWE pool whose rows are (a) format-complete
for the self-guided strategies (image + workdir + targeted eval_cmd + gold_patch +
pre_commands), (b) decontaminated against SWE-bench test/dev/Verified/Lite/Pro,
and (c) backed by execution evidence of solvability / gradability.

Sources (all restored from blob into training_data/pool_v1/):
  * swe_rl_9b_mix_v1.jsonl      6,650 rows  -- 9B RL mix; eval_cmd already rewritten
                                            to targeted node-ids (rebench) / official
                                            targeted test_cmd (swesmith)
  * swe_distill_5358.jsonl      5,358 rows  -- 27B teacher-distill pool, same rewrite,
                                            zero overlap with mix_v1
  * swe_rl_python_all_v1.jsonl 49,617 rows  -- the decontaminated + de-duplicated master
                                            pool; used here ONLY as the membership
                                            filter (a labeled row absent from it was
                                            dropped upstream as contaminated/duplicate)

Evidence (execution results, joined by instance_id):
  * Qwen3.5-35B-A3B naive rollouts on mix_v1 (49,339 unique, ~8 per task):
      ~/xiao/rollout_gen_35b/{global_done,reward_*}.jsonl, ~/xiao/cov_*/**, ...
  * Qwen3.5-35B-A3B self-guided (minimal_branch k=2) rollouts on 2,951 mix_v1 tasks:
      training_data/distill3k_selfguide_trajectories_20260803.jsonl (solved flag)
  * Qwen3.6-27B teacher rollouts on the 5,358 pool (5,669, k~1):
      training_data/pool_v1/distill_raw_dumps_backup.jsonl.gz

Tiers:
  A  proven  -- >=1 rollout scored reward 1.0 (env gradable AND task solvable)
  B  unproven -- every rollout scored 0 (hard, or env-rot); MUST pass
                validate_task_env.py (bug FAILs + gold PASSes) before use

Outputs (--out-dir):
  candidates_all.jsonl, tierA.jsonl, tierB.jsonl, images_tierB.txt, summary.json
Each row keeps the standard {prompt,label,metadata} shape and gains
metadata.si = {tier, src, evidence, difficulty_bin, needs_validation}.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re
from pathlib import Path

TD = None
REWARD_SOURCES = []
SELFGUIDE_TRAJ = TEACHER27_GZ = MIX_V1 = DISTILL_5358 = POOL = None


def _sid(i: str) -> str:
    return re.sub(r"#dup\d+$", "", i or "")


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_35b_naive():
    """Same dedup key as deficit_analysis/coverage_analysis.py: (iid, elapsed, diff_len)."""
    seen = {}
    for fp in sorted(set(REWARD_SOURCES)):
        try:
            for d in rows(fp):
                iid = _sid(d.get("instance_id", "?"))
                key = (iid, round(d.get("elapsed", 0.0) or 0.0, 1), d.get("diff_len", 0) or 0)
                seen[key] = d
        except FileNotFoundError:
            pass
    per = collections.defaultdict(lambda: {"n": 0, "pass": 0, "usable": 0})
    for (iid, _, _), d in seen.items():
        p = per[iid]
        p["n"] += 1
        if (d.get("reward") or 0) >= 0.999:
            p["pass"] += 1
        if d.get("applied") and (d.get("diff_len") or 0) > 0:
            p["usable"] += 1
    return per


def load_35b_selfguide():
    out = {}
    if SELFGUIDE_TRAJ.is_file():
        for d in rows(SELFGUIDE_TRAJ):
            out[d["label"]] = bool(d.get("solved"))
    return out


def load_27b_teacher():
    per = collections.defaultdict(lambda: {"n": 0, "pass": 0})
    if TEACHER27_GZ.is_file():
        for d in rows(TEACHER27_GZ):
            p = per[_sid(d.get("instance_id") or d.get("label") or "?")]
            p["n"] += 1
            if (d.get("reward") or 0) >= 0.999:
                p["pass"] += 1
    return per


def difficulty_bin(ev: dict) -> str:
    """Solve-rate bin for the 35B naive policy (n>=4), else coarse label."""
    if ev["n35"] >= 4:
        r = ev["pass35"] / ev["n35"]
        return "35b_0" if ev["pass35"] == 0 else "35b_lt25" if r < 0.25 else "35b_25_75" if r < 0.75 else "35b_ge75"
    if ev["n27"] > 0:
        return "27b_pass" if ev["pass27"] > 0 else "27b_0"
    return "no_rollout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--reward-files", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    global TD, REWARD_SOURCES, SELFGUIDE_TRAJ, TEACHER27_GZ, MIX_V1, DISTILL_5358, POOL
    TD = Path(a.input_root).resolve()
    REWARD_SOURCES = [p for pattern in a.reward_files for p in glob.glob(pattern, recursive=True)]
    SELFGUIDE_TRAJ = TD / "distill3k_selfguide_trajectories_20260803.jsonl"
    TEACHER27_GZ = TD / "pool_v1/distill_raw_dumps_backup.jsonl.gz"
    MIX_V1 = TD / "pool_v1/swe_rl_9b_mix_v1.jsonl"
    DISTILL_5358 = TD / "pool_v1/swe_distill_5358.jsonl"
    POOL = TD / "pool_v1/swe_rl_python_all_v1.jsonl"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    naive = load_35b_naive()
    sg = load_35b_selfguide()
    t27 = load_27b_teacher()
    pool_ids = {d["metadata"]["instance_id"] for d in rows(POOL)}
    print(f"[evidence] 35B naive instances={len(naive)} selfguide={len(sg)} 27B={len(t27)} pool_ids={len(pool_ids)}")

    cands, dropped = [], collections.Counter()
    for src_file, setname in ((MIX_V1, "mix_v1"), (DISTILL_5358, "distill_5358")):
        for d in rows(src_file):
            m = d["metadata"]
            iid = m["instance_id"]
            if iid not in pool_ids:
                dropped["not_in_decontaminated_pool"] += 1
                continue
            if not (m.get("gold_patch") or "").strip():
                dropped["no_gold_patch"] += 1
                continue
            if not (m.get("eval_cmd") and m.get("image") and m.get("workdir")):
                dropped["incomplete_row"] += 1
                continue
            src = "swesmith" if "/swesmith." in m["image"] else "rebench_v2"
            n35 = naive.get(iid, {"n": 0, "pass": 0, "usable": 0})
            n27 = t27.get(iid, {"n": 0, "pass": 0})
            ev = {"n35": n35["n"], "pass35": n35["pass"], "usable35": n35["usable"],
                  "sg35_solved": sg.get(iid), "n27": n27["n"], "pass27": n27["pass"]}
            proven = ev["pass35"] > 0 or ev["pass27"] > 0 or bool(ev["sg35_solved"])
            m["si"] = {
                "tier": "A" if proven else "B",
                "src": src,
                "from_set": setname,
                "block": m.get("mix_block") or m.get("distill_block"),
                "evidence": ev,
                "difficulty_bin": difficulty_bin(ev),
                "needs_validation": not proven,
            }
            cands.append(d)

    tierA = [d for d in cands if d["metadata"]["si"]["tier"] == "A"]
    tierB = [d for d in cands if d["metadata"]["si"]["tier"] == "B"]
    for name, sel in (("candidates_all", cands), ("tierA", tierA), ("tierB", tierB)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for d in sel:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    imgs = sorted({d["metadata"]["image"] for d in tierB})
    (out / "images_tierB.txt").write_text("\n".join(imgs) + "\n")

    def stats(sel):
        c = collections.Counter()
        for d in sel:
            si = d["metadata"]["si"]
            c[f"{si['src']}/{si['difficulty_bin']}"] += 1
        return {"n": len(sel), "repos": len({d["metadata"].get("repo") for d in sel}),
                "images": len({d["metadata"]["image"] for d in sel}), "by_src_difficulty": dict(sorted(c.items()))}

    summary = {"dropped": dict(dropped), "tierA": stats(tierA), "tierB": stats(tierB), "all": stats(cands)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] wrote {len(cands)} candidates -> {out}")


if __name__ == "__main__":
    main()
