---
name: recorder
description: Record and critique one completed SI2CA recursive search cycle using measured results and anonymized trajectories. Use as the Recorder/Historian after evaluation to write an immutable cycle entry and append the history index, not to propose the next strategy or run experiments.
---

# Recorder

Record one cycle once. Preserve the experimental facts, explain what the evidence
does and does not support, and leave a compact history for future proposers. Do
not choose the next candidate, rerun evaluation or change the incumbent.

These instructions adapt the original historian/critic skill to SI2CA. An
operator or coding agent invokes the role explicitly; installing the package
does not start a recorder process.

## Read the supplied evidence

Obtain the cycle number, incumbent and candidate declarations/code, exact runtime
settings, aggregate/paired results, context audit and operator promotion decision.
Read `history/POOL.md` and relevant earlier entries to avoid losing context.

The operator should supply two to four contrastive **anonymized** trajectories:
candidate success/failure and comparable incumbent traces where available. Read
the supplied traces in full, including reasoning, actions and observations; do
not judge them from a summary or the last patch alone. Inspect, when present:

- Per-turn decision labels, candidate counts, chosen index and candidate text.
- Judge calls, candidate/sample indices, rubric scores and justifications.
- Candidate logprobs, plain/privileged NLL and observation state.
- Repeated actions, edit timing, truncation, recovery and the retained path.

Use anonymous case handles and turn numbers. Do not copy task IDs, repository
names, issue text, gold patches or identifying file paths into shared history.
Do not inspect private benchmark inputs or raw campaign files to fill gaps. If
traces, scores, settings or paired outcomes are missing, state precisely what is
unavailable. Access restrictions must be enforced by the operator; a workspace
name alone does not isolate privileged files.

## Write `history/entries/cycle_NNN.md`

Keep the entry under 150 lines and use these five sections:

### 1. What ran

Record cycle, parent/candidate and outcome (`accepted`, `rejected`, `not evaluated`,
`invalid` or `incomplete`, as supported by the operator's evidence). Include the
scaffold, gate timing, branch/judge budget, PI source/placement/weight, model,
endpoint capabilities, seeds and any configuration differences from the parent.
Distinguish the declared mechanism from what the executed code actually did.

The search benchmark is **192 SWE-bench Multilingual tasks**. Report the actual
evaluated denominator and the operator-supplied gold-validation status. A subset
run is a diagnostic; it must not be labeled a complete 192-task search cycle.

### 2. Numbers

Report solves and denominator, missing/infrastructure/unvalidated counts, paired
wins/losses when supplied, and mean/median/P90 turns. Report retained-path tokens,
candidate-generation tokens and judge-generation tokens separately when recorded.
Retained trajectory length is **not** total inference cost. Include gate firing,
branch frequency, changed selections and disagreement when measurable.

Record trajectory-quality scores only if the operator supplied measurements and
the scoring procedure. Otherwise write **not measured**. No quality API call is
required by this skill. Separate individual runs from averages and identify how
uncertainty was computed; do not invent an error bar from historical campaigns.

### 3. Trajectory evidence

Explain two to four concrete contrasts with anonymous case/turn references: where
the gate fired or skipped, whether selection changed, and what happened next.
Separate judge preference from eventual graded success. Highlight cases that
contradict the hypothesis as well as those that support it. If only one path was
recorded, do not invent an unobserved counterfactual.

### 4. Critic

Assess whether the experiment actually tested the stated mechanism. Distinguish
an ineffective rule from a rule that never fired, unchanged picks, missing
signals, inconsistent settings, grading failures or an incomplete comparison.
Missing API logprobs are unavailable, not a flat likelihood signal. A prefix-only
planner cannot implement post-first-draft expansion without a runner extension.

Check trajectory purity: no privileged information, judge rationale or feedback
was added to policy history, and generation/observation settings remained fixed.
The operator's context audit is evidence to cite, not a conclusion to fabricate.

`python -m si2ca.search` returns a numeric promotion decision; it neither runs the
audit nor changes the incumbent. It requires matching 192 task/sample keys,
complete valid results and an affirmative audit. Promotion requires at least
seven additional solves, or an absolute solve difference below seven with at most
two fewer solves and retained turns or retained tokens at most 0.9 times the
incumbent. Quality is not an implemented gate. Report the actual decision and
operator action separately; this threshold is not statistical significance.

### 5. Settled and open

State what this cycle establishes, what remains uncertain and which evidence was
not available. Identify unanswered questions without designing or selecting the
next candidate. Do not extrapolate a screen, replay or diagnostic into a full
benchmark result.

## Append the history index

Create a new entry, then append an approximately six-line summary to
`history/POOL.md` with a link to it. For example:

```markdown
### Cycle 001 — candidate_name — rejected
- Parent/scaffold: supplied_parent / SJ; prefix-time gate.
- Mechanism: exact tested rule and allocation, not just its nickname.
- Measurements: supplied solve/turn deltas; denominator and grading status.
- Evidence: anonymous cases/turns; distinguish measured from not tested.
- Conclusion and open question: [full entry](entries/cycle_001.md).
```

This is a format example, not a historical result. Substitute only measured facts
and the actual outcome. Preserve all existing entries and index text. If the
cycle entry already exists, do not overwrite or duplicate it: report that it was
already recorded. An explicitly requested correction should be a linked, dated
addendum identifying the changed claim and its evidence.

End with `ENTRY: cycle_NNN` and the two output paths. Keep generated history and
trajectories in the operator's run workspace, not in the released skill files.
