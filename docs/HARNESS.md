# Unified evaluation harness

All new public experiments use `si2ca.runtime.strategy` with protocol
`si2ca-unified-v1`. Standard, Self-Judgement (SJ), Self-Likelihood (SL),
Discovered and student evaluation share one execution loop, including DeepSWE's
Standard and SJ presets. `si2ca.evaluate.evaluate` is a compatibility import for
the same dispatcher, not another engine.

## Shared execution contract

| Area | Unified behavior |
|---|---|
| Workspace | Prepare the dataset's `metadata.workdir`; use it in both the prompt and every agent command. Pro tasks no longer receive an inconsistent `/testbed` prompt when their workspace is `/app`. |
| Candidate actions | Validate the whole bash-tool batch. Execute every call in order, stopping only after a successful submission. Malformed batches are rejected as a whole, not silently reduced to their first call. |
| Submission | The executed command must return zero and its first nonblank output line must equal `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Merely mentioning the marker in a command does not submit. |
| Policy history | Keep the selected assistant content, reasoning and executed tool calls by default. The `keep_reasoning: false` override applies to the shared loop. Every executed tool call has its own result message. |
| Identical candidates | Compare the complete sequence of commands, choose with the seeded RNG and skip SJ judging. No 15% null-control judge calls in public SJ/Discovered. |
| Score ties | SJ, SL and likelihood-gain selectors choose among tied scores with the seeded RNG. Remote sampling itself is not guaranteed deterministic. |
| SJ judge | Full SJ and Discovered use the same `SelfGuideRubrics` selector and pure prompt renderer. Candidate rendering includes all tool calls; previous tool-calling reasoning is omitted from the judge prefix. |
| Judge sampling | Explicit temperature 1.0 and top-p 0.95 by default; one independent request per score sample. Endpoint, model, effort, token cap and prefix budgets are configurable. API adapters omit unsupported sampling fields unless `--api-sampling` is set. |
| Rubric precision | The prompt and numerical scoring use the same weights, including `0.105` at gold weight 0.3. No two-decimal rounding of the prompt formula. |
| Privileged information | Gold is used only in SJ judge requests; gold/hints enter only SL rescoring. Policy generation history remains unprivileged. |

Standard and student evaluation use the baseline selector. SL retains its
likelihood scoring objective, and gain retains its plain-minus-privileged
approximation. Unavailable/invalid scores retain explicit fallback labels;
these are visible in the decision logs, not reported as successful scoring.

The Discovered Early-Commit Window Strategy changes only **when to branch**;
branched turns use the same SJ procedure. The first turn is unbranched. With
`--branch-until-edit 3`, branching continues through the third mutation-like
command, then commits to one candidate. This is the retained command-text
heuristic, not a filesystem edit counter. `--method Discovered` activates it
directly; inherited `SLIME_BRANCH_GATE` overrides remain cleared. The window
helper lives in `si2ca.runtime.protocol`; there is no separate legacy gate adapter.

## Deliberate benchmark differences

The execution loop is shared; grading protocols and declared experimental
budgets are not forced to be identical across different benchmarks.

- Verified and full Pro (731 tasks) apply the exported patch in a fresh grading sandbox.
- Multilingual uses its grading adapter on a fresh sandbox.
- DeepSWE grades in the solving sandbox using `si2ca.runtime.deepswe_grading`, loading verifier assets only for grading.
- DeepSWE presets retain their turn, token, judge-sample and independent-run budgets. Use the same effective configuration when comparing methods on a benchmark.

## Failures, results and resume

Every new result includes `harness_protocol`; task records also include
`execution_status` and, once grading is reached, `grading_protocol`.
All presets write `results.json`, `traj/` and exported `patches/` under the public `--out` directory.
Repeated trials carry a `sample` index; their trajectory filenames have `_sN`
suffixes when `n_samples > 1`.

Transport failures, exhausted candidate requests, task timeouts and reported
grading infrastructure failures are **incomplete trials**, not completed model
failures. Historical driver `exit_code=1` or `-1` records are also flagged by the
summary tool. HTTP transport/5xx/429 failures have bounded retries; permanent
4xx failures do not retry indefinitely.

The denominator remains the declared tasks × independent samples. A partial
resolve rate is provisional: `final` is false until all trials are present,
infrastructure failures are resolved and required grading validation is complete.
Incomplete trials do not count as solves or enter turn statistics; `turns_trials`
reports their statistical coverage. This does not inflate accuracy by deleting
failed API calls from the denominator. Max-turn and format-error model exits
remain completed attempts when execution/grading otherwise succeeds.

Resume retains completed trials and retries infrastructure failures. Use a new
output directory for the unified protocol: old or mixed-version files are
rejected. Public runs additionally compare the saved effective configuration.
Scaffolds use only `none`, `SJ`, `SL`, or `SJ+SL`, with no alias conversion.
Resume requires matching names; use a new output directory for older labels.
The lower-level strategy-file runner requires the operator to match all sampling,
budget and dataset settings explicitly; its version check is not a complete
configuration audit. Search promotion rejects mismatched protocol versions.

For repeated DeepSWE trials, report per-sample resolve rates and their mean/sample
standard deviation, not whether any of the samples solved each task (pass@N).

## Historical paper results

This refactor does **not** rerun or retroactively standardize the paper results.
Archived experiments used both the branch harness and strategy runner; historical
DeepSWE Standard/SJ also used different MinimalHarness versions. Differences in
tool handling, workspace prompts, history, judging and submission can affect
outcomes. The unused historical drivers and their private helpers have been
removed from main; they remain recoverable from Git history. Their removal does
not change the shared runner or the DeepSWE in-place grading procedure.

Paper values and raw records remain unchanged. Claims that all historical
experiments used one harness, or differed only in decoding, need qualification
in the paper; code unification alone is not evidence for those claims. A fair
new comparison requires rerunning both baseline and improved methods with this
protocol and matched configurations. See [archived data notes](https://github.com/Self-Improving-Coding-Agents/SI2CA-Visualization/blob/main/paper_data/README.md)
for known source gaps and paper-versus-record differences.
