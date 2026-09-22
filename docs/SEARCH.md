# 🔄 Recursive Search: one-cycle guide

SI2CA provides two agent skills, a strategy evaluator and a promotion comparator.
The operator connects them as **proposer → evaluation → recorder**, then repeats
with the selected incumbent. There is no bundled autonomous agent scheduler.

## 🧩 Find and invoke the skills

```bash
si2ca ls skills
```

This prints readable file paths in the checkout or installed package; it performs
no model requests. Give the relevant `skills/*.md` file to your coding agent explicitly,
with a task and the staged inputs below. Installation does not register or launch
agents in a particular agent platform.

These are standalone Markdown files. If your agent platform requires a
`<skill-name>/SKILL.md` installation layout, copy the chosen file into that layout
in the platform's skill directory; no such subdirectories are kept in this repo.

| Role | Skill | Inputs | Outputs |
|---|---|---|---|
| Proposer | [Proposal.md](../skills/Proposal.md) | Cycle task, incumbent code/config, anonymized traces and history | One `candidates/<name>/strategy.py`, offline tests and `pending_eval.json` |
| Recorder | [Recorder.md](../skills/Recorder.md) | Proposal, measured comparison, operator decision and anonymized contrastive traces | New `history/entries/cycle_NNN.md` and an appended `history/POOL.md` summary |

`pending_eval.json` is a handoff declaration, not a file automatically consumed by
the evaluator. Neither skill runs evaluation or promotes an incumbent itself.

## 📋 1. Fix the benchmark and incumbent

Search uses **192 SWE-bench Multilingual tasks**.
The bundled manifest records each task's gold-validation status.
Complete gold-validation checks before a promotable full cycle; retain the
verified status in both runs. A `--validated-only` subset run is a diagnostic.
Explicit `--allow-unvalidated` runs are exploratory, not validated search results.

Evaluate an SJ incumbent:

```bash
si2ca run \
  --method SJ \
  --benchmark validation \
  --model Qwen/Qwen3.5-122B-A10B \
  --base-url http://127.0.0.1:8151 \
  --out runs/search/incumbent
```

If validation was completed in a new manifest, supply it with `--dataset`;
do not merely flip validation flags.
Execution writes `config.json`, normalized `tasks.jsonl`, `results.json` and
`traj/` in the output directory.
Standard, SJ and candidates use the same [versioned strategy runner](HARNESS.md).
Rerun the incumbent in a new output directory after a harness-version change;
historical results cannot be resumed or promoted against a new-version candidate.

The operator keeps the dataset, references and raw results private. Stage a
separate agent workspace containing `TASK.md`, `INCUMBENT.md`, parent code,
anonymized evidence and history (empty history is valid for the first cycle).
State the permitted change, runtime settings, available signals and cycle number.
Remove identifying/privileged content from traces before sharing them. Restrict
agent access to those staged files if the original checkout is otherwise visible.

## 💡 2. Propose and review one strategy

Give the proposer skill and staged workspace to the coding agent. It returns one
candidate and an evidence-backed declaration. Review the code and run its offline
tests with synthetic contexts and a stub judge before loading it: Python strategy
files are executable code, and the loader is not a sandbox.

Only `Strategy.plan(ctx)` and `Strategy.select(ctx, candidates, judge)` may change
candidate allocation/selection. Policy messages must remain unprivileged; gold
belongs in SJ judge requests or SL rescoring requests, never in generation.
The current planner acts before candidate generation, with no post-draft expansion
hook. Keep submitted candidates immutable and give revisions a new name.

## 🧪 3. Evaluate with the same configuration

Use the lower-level runner for a reviewed external strategy. The following is an
SJ example matching the default incumbent's generation limits and sampling; the
candidate should inherit `SelfGuideRubrics` with K=2, SCORE_SAMPLES=3 and
GOLD_WEIGHT=0.30 unless its declared mechanism changes these.

```bash
SI2CA_BACKEND=self-hosted SI2CA_JUDGE_BACKEND=self-hosted python -m si2ca.runtime.strategy --dataset runs/search/incumbent/tasks.jsonl --strategy runs/search/candidates/c001/strategy.py --scaffold SJ --model Qwen/Qwen3.5-122B-A10B --base-url http://127.0.0.1:8151/v1 --seed 42 --n-samples 1 --concurrency 8 --task-timeout 100000000 --eval-timeout 3600 --max-turns 250 --temperature 1.0 --top-p 0.95 --top-k 10 --max-gen-tokens 4096 --k 2 --score-samples 3 --gold-weight 0.3 --judge-temperature 1.0 --judge-top-p 0.95 --judge-max-tokens 4096 --judge-reasoning-effort medium --out runs/search/evaluations/cycle_001/results.json --traj-dir runs/search/evaluations/cycle_001/traj
```

Replace `c001` with the submitted candidate name and change the explicit selector
overrides only when declared by the proposal. Like `si2ca run`, this command
executes immediately; it expects an already served model and normalized dataset,
and takes a **result file** for `--out`. It does not apply the public CLI's
validation-status guard. Do not use it to bypass validation or report pending
grading as valid. Match the saved incumbent configuration and environment,
including any reasoning/API options, rather than assuming the example matches
every preset. Do not inherit unrelated experiment overrides.

`--scaffold SJ` enables gold in the judge; `none` is the no-PI variant. SL uses
`--scaffold SL`, a compatible self-hosted SGLang endpoint and `--tokenizer-path`, with explicit
`--inject-source gold|hint` and `--inject-position head|tail`. Gold covers all 192
tasks; hint-based search additionally needs a complete `--hints-file`. API SJ
does not imply that logprobs or SL are supported. See [parameters](PARAMETERS.md).

## ⚖️ 4. Compare and choose the incumbent

After checking matching settings, complete grading and trajectory purity:

```bash
python -m si2ca.search \
  --incumbent runs/search/incumbent/results.json \
  --candidate runs/search/evaluations/cycle_001/results.json \
  --context-audit-passed
```

Only pass the audit flag after the review. The command compares exactly 192
matching task/sample keys and prints a JSON decision; it does not run the audit
or update any incumbent file. Defaults accept at least seven additional solves,
or an absolute solve difference below seven with at most two fewer solves and
retained turns or retained tokens ≤0.9× the incumbent. This is a heuristic, not
statistical significance. Retained-path cost excludes discarded candidate and
judge generation; trajectory quality is not an implemented promotion gate.

## 📝 5. Record and hand off the next cycle

Give the recorder skill the proposal, measured comparison, audit, actual operator
decision and two to four anonymized contrastive traces. It writes one cycle entry
and appends the history index without overwriting previous entries. Missing
measurements remain explicitly missing; the recorder does not propose the next
candidate. The operator selects the next incumbent and supplies the updated
history to the next proposer invocation.

Keep proposals, trajectories, results and generated history under the operator's
run workspace. The release includes the reusable skills, not historical campaign
state, searched trajectories or fleet-management scripts.
