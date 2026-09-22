---
name: proposal
description: Propose exactly one SI2CA decoding strategy for a recursive search cycle from anonymized history and offline replay evidence. Use for WHERE, HOW-MANY or GUIDANCE changes to Strategy.plan/select, not for running benchmarks or recording completed cycles.
---

# Proposal

Develop one candidate strategy per cycle. The operator owns benchmark execution,
grading and promotion; the recorder owns the completed-cycle history. Do not run
the benchmark, promote yourself or claim an unmeasured improvement.

These instructions are the portable SI2CA adaptation of the original decoding
proposer skill. They do not start an autonomous agent or evaluation process.

## Inputs and workspace

Read the operator's task, incumbent description/code, `history/POOL.md`, relevant
cycle entries and supplied anonymized trajectories before designing a candidate.
The operator must state the cycle number, parent, permitted scaffold, fixed runtime
configuration and available signals. Request missing essentials rather than
inventing them. Follow explicit user constraints; treat historical suggestions as
hypotheses, not evidence that a change works.

Work only with the staged evidence. Do not inspect benchmark manifests, gold
patches, hints, issue text, original task IDs, repositories or raw campaign output.
Use anonymous case handles and turn indices in proposals. A working-directory
convention is not an access-control boundary: the operator must restrict tools or
file access if the surrounding checkout exposes privileged inputs.

The fixed search split is **192 SWE-bench Multilingual tasks**. The operator must
supply the current gold-validation status. A subset diagnostic is not a full
search cycle. Missing or failed grading is not evidence that the policy failed
to solve a task.

## Preserve the experiment

- Keep the model/checkpoint, endpoint configuration, sampling, task split, seeds,
  grader, observation processing and context limits fixed against the incumbent.
- Change only candidate allocation in `plan` and winner selection in `select`.
  Never rewrite the policy history, observation, system prompt, candidate text or
  tool call. Do not insert feedback, hints, gold, judge rationales or a nudge into
  the policy's next generation request. Only the selected policy continuation is
  retained by the existing runner.
- Do not condition on task identity, benchmark position, repository or known
  answers. Do not read external data, make independent model/network calls, use
  fixed ports or keep cross-task state. Use `ctx.rng` for strategy randomness.
- Do not branch on elapsed time or accumulated token-budget counters. Describe
  candidate allocation explicitly and account for its cost separately.
- Do not modify the runner, grader or scoring implementation to make a proposal
  work. A missing interface requires a separately scoped extension, not an
  undocumented change to the comparison.

## Choose the scaffold and one mechanism

**Self-Judgement (SJ).** Generate ordinary policy candidates, then select using
the same model as a judge. `SelfGuideRubrics` provides the seven-rubric baseline;
with gold enabled, G1 receives weight 0.30 and the other rubrics share 0.70. The
reference recipe uses two candidates and three judge samples. The runner injects
gold into the judge request only. Never retrieve it into candidate code or policy
messages. Record any allowed change to judge sampling or weights explicitly.

**Self-Likelihood (SL).** Generate ordinary policy candidates, then score them
with privileged information in a separate zero-new-token request. Selection may
minimize privileged NLL or maximize plain NLL minus privileged NLL (likelihood
gain). Record hint/gold source and head/tail placement. The runner owns injection;
the strategy uses the resulting scores, not the reference content.

SL requires a compatible self-hosted SGLang endpoint with native input logprobs
and a matching tokenizer. API SJ does not imply logprob availability. Gold is
bundled for all 192 tasks, but hints for the additional 128 are not bundled: the
operator must provide complete hints before hint-based search. `None`/missing
logprobs are unavailable, not zero. Use distinct decision labels for unavailable
signals and ordinary non-firing gates. Sampled-token surprisal is not full
distribution entropy.

Declare one primary axis, or explain one inseparable composition:

- **WHERE:** a prefix-time gate choosing when to branch.
- **HOW-MANY:** how many candidates or judge samples to allocate at a decision.
- **GUIDANCE:** a selection rule using the allowed judge or likelihood signals.

The current runner calls `plan` **before** drawing the candidates. It has no
post-first-draft expansion hook. Drawing all candidates first and selecting one
does not save candidate-generation cost. Do not propose draft-conditioned
allocation as if this interface already implements it.

## Interface and offline prototype

Read the live contract in `si2ca/runtime/strategy.py` from the source checkout or
the installed `si2ca.runtime.strategy` module. Useful existing implementations are
`si2ca/strategies/discovered.py`, `sl.py` and `sl_gain.py`. Use SI2CA imports, not
imports from the original slime repository.

Write `candidates/<new_snake_case_name>/strategy.py` with a `Strategy` class or a
`STRATEGY` object exposing:

```python
from si2ca.runtime.strategy import SelfGuideRubrics


class Strategy(SelfGuideRubrics):
    name = "example_prefix_gate"
    K = 2
    SCORE_SAMPLES = 3
    GOLD_WEIGHT = 0.30

    def plan(self, ctx):
        # Illustrative contract only; not an empirically validated proposal.
        return self.K if ctx.last_returncode not in (None, 0) else 1
```

Inherit `select` when changing only the gate. For a custom selector the contract is
`async select(ctx, candidates, judge) -> (winner_index, decision_label, usage_dict)`.
`plan(ctx)` returns an integer at least one; the selected index must be valid.
`judge(prompt, system=None)` returns `(text, usage)` and retains the runner's
scaffold injection. Preserve judge usage accounting when making judge calls.

Use the supplied parent as the comparison, not the example gate above. Test with
synthetic contexts/candidates and a stub judge; no endpoint, GPU or container is
needed for these tests. Check non-firing and firing cases, the single-candidate
case, missing signals, deterministic behavior under the same `ctx.rng` seed and
that input histories/candidates remain unchanged. Audit before loading: the
strategy loader imports Python code and is **not a security sandbox**.

Replay the gate/selection on staged trajectories where the recorded fields allow
it. Report how often it fires, which picks differ from the parent, candidate/judge
cost and representative anonymous turns. A gate that never fires and a selector
that never changes the winner have not tested the hypothesis. If a counterfactual
candidate or signal was not recorded, mark that comparison **not tested**. Do not
infer task success, quality, entropy or statistical significance from unavailable
data. Historical noise estimates do not automatically apply to this split.

## Deliver one candidate

Keep submitted candidate directories immutable; put revisions in a new directory.
Write local tests and a `pending_eval.json` declaration such as:

```json
{
  "cycle": 1,
  "candidates": [{
    "name": "new_snake_case_name",
    "path": "candidates/new_snake_case_name",
    "entrypoint": "candidates/new_snake_case_name/strategy.py",
    "scaffold": "SJ",
    "question": "WHERE",
    "scaffold_config": {"k": 2, "judge_samples": 3, "gold_weight": 0.3},
    "parent": "operator_supplied_incumbent",
    "hypothesis": "A falsifiable claim grounded in staged evidence",
    "trigger_or_budget": "Exact rule and when its inputs are available",
    "signal": "Required prefix fields and behavior when unavailable",
    "expected_firing_rate": null,
    "expected_cost": "Prediction, with candidate and judge cost separated",
    "expected_solve_effect": "Prediction, not a benchmark result",
    "expected_traj_quality": "Not measured unless quality evidence is supplied",
    "prototype": "Measured replay counts, changed picks and limitations",
    "evidence": "Anonymous case handles and turn indices"
  }]
}
```

Replace placeholders with actual proposal details. `null` means not estimated,
not zero. This declaration is an operator handoff format; SI2CA does **not**
automatically read it, schedule evaluation or validate the proposal's claims.

Promotion is evaluated by the operator with `si2ca.search`: compare the same 192
task/sample keys, complete and valid grading, and an audited policy context. The
numeric rule is at least seven additional solves, or a solve difference with
absolute value below seven and at most two fewer solves, together with retained
turns or retained tokens at most 0.9 times the incumbent. This is a promotion
heuristic, not a statistical-significance test or a full-generation-cost bound.
Trajectory quality is not an implemented promotion gate. Only measured results
can establish whether the candidate meets these conditions.

End with `CANDIDATE: <name>` and a concise summary of the hypothesis, offline
evidence, changed files and remaining evaluation needs.
