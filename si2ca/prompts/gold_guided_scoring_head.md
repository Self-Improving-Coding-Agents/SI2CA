<!--
HEAD-position variant of the privileged-context block.

Same information, different place: this block is inserted right after the
problem statement and BEFORE the trajectory, so the scored context reads

    [system] [problem statement] [REFERENCE SOLUTION] [a_1 o_1 ... o_{t-1}] -> candidate

instead of the tail arrangement, where it sits between o_{t-1} and the candidate.

WHY THE TAIL VERSION NEEDED REPLACING. Measured over 264,865 candidates on
2026-08-16: appending ANY diff-shaped block at the tail raises the candidate's
mean NLL by +0.221, while the answer's own content contributes only +0.066 on
top of that. In ranking terms the damage is worse -- argmin(gold) and
argmin(some other instance's patch) picked the SAME candidate 81.8% of the time
(random baseline for k=4 is 25%). A block dropped in immediately before the
continuation point interrupts a trajectory that was flowing, and the model's
rise in uncertainty is mostly a reaction to the interruption, not to the answer.

Placing it early makes it part of the task statement the whole trajectory was
written under, which is the condition we actually wanted to measure:
p(O_t | problem, G, a_1..o_{t-1}) rather than p(O_t | problem, a_1..o_{t-1}, G).

TEXT DELTA FROM THE TAIL VARIANT: one clause. The tail version ends with the
harness's own "write your next command(s)", which is right when the candidate
follows immediately and wrong here (the trajectory follows). Everything else --
lead-in, === delimiters, anti-copy instruction, "do not mention it" -- is
byte-identical, so the two arms differ in placement and nothing else.

COST. The gold context now forks from the plain one at the problem statement, so
each trajectory holds TWO prefixes in the radix cache for its whole life instead
of sharing one up to the tail. Measured pool is 1.64-1.85M tokens per replica
against ~16 live trajectories, so this fits at concurrency 64; it would not have
fit at the 958k pool the 2026-08-07 run had.

PLACEHOLDER: {{reference_patch}} (plain textual substitution, not jinja).
-->
Here is a reference solution to this issue:
=== Reference Solution Begin ===
{{reference_patch}}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand what it changes and why — do not copy or paraphrase it, and do not mention it. Work through the task with your own reasoning, as you otherwise would.
