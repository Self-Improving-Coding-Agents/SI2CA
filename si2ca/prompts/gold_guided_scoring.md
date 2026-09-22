<!--
Privileged-context block for golden-patch guided decoding.

WHAT THIS IS. At turn t the agent has produced a prefix
    Prev = [problem statement, a_1, o_1, ..., o_{t-1}]
and k candidate next turns O_t^1..O_t^k drawn from p(. | Prev). This block is
appended to Prev as ONE user message, and each candidate is then scored under
    p(O_t^i | Prev, G)
where G is the golden patch. The candidate with the lowest mean NLL is executed.
Nothing is ever generated from this block; it exists only to condition the scorer.

WHY IT IS THIS SHORT. The score is a difference between four candidates under the
same context, so everything the block adds beyond the patch itself is a constant
that shifts all four and a perturbation that moves the model off the trajectory
manifold it was sampling on. An earlier draft re-introduced the agent to itself
("you are the same engineer, in the same session..."), which is both confusing --
the model has no notion of a version of itself that did not receive this message
-- and expensive: several hundred tokens of new framing at the decision point.
The block is now the smallest thing that carries the information.

SHAPE. Copied from OPSD's teacher prompt (arXiv 2601.18734, quoted verbatim in
reports/2026-07-17_privileged_context_teacher_prompt_survey.md), which is the
in-context self-distillation recipe this experiment is built on:
  1. one lead-in line naming what follows;
  2. the reference solution inside === Begin/End === delimiters;
  3. an anti-copy instruction. In OPSD it keeps the privileged distribution
     shaped like independent solving rather than transcription; here it does that
     AND stops the block from making "paste the patch" the likeliest next turn;
  4. a closing line that repeats the SAME generation instruction the agent
     already has ("write your next command(s)", the harness's own wording), so
     the surface form being scored is unchanged.
"do not mention it" is the one addition: a candidate that does not refer to the
patch must not be scored as surprising for the omission.

The file list appears ONLY when the patch had to be truncated (8.2% of the pool
at 12k chars), so the common case stays byte-minimal while a truncated one still
says where the fix lives. Which case applied is recorded per turn as
`gold_truncated`.

PLACEHOLDER (plain textual substitution, not jinja -- diffs contain braces):
    {{reference_patch}}
Everything outside the block below is comment and is NOT sent to the model. The
message begins at the "Here is" line.
-->
Here is a reference solution to this issue:
=== Reference Solution Begin ===
{{reference_patch}}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand what it changes and why — do not copy or paraphrase it, and do not mention it. Now, using your own reasoning, continue working toward the same outcome: write your next command(s) as you otherwise would.
