<!--
HINT variant of the head-position privileged block.

Same placement as gold_guided_scoring_head.md -- right after the problem
statement, before the trajectory -- and the same four-part shape. The ONLY
change is what sits between the delimiters: a fixed-length hint block derived
from the golden patch instead of the patch itself.

WHY. The patch's length is not a controlled variable: 116 to 12,000+ characters
across this pool, and correlated with the benchmark (Verified median 882, Pro
6,169). Measured over the head arm's 245,992 candidates, the scoring
perturbation grows with it -- mean delta +0.0084 for blocks under 500 chars vs
+0.0187 for 4,000-12,000, with the "answer helped" share falling 34.6% -> 29.3%.
So "head helps on small patches, hurts on large ones" confounds two things:
large patches are Pro instances AND they perturb the scoring twice as hard.
A block of near-constant length (700-1100 chars, target 900) removes the length
axis and leaves the information axis.

TEXT DELTA FROM THE HEAD VARIANT: the noun only. "reference solution" -> "hints",
singular -> plural agreement. Lead-in, delimiter shape, anti-copy instruction,
"do not mention" and the closing generation instruction are otherwise identical,
so the two arms differ in the block's content and length, not in its framing.

PLACEHOLDER: {{reference_patch}} (plain textual substitution, not jinja).
-->
Here are some hints for this issue:
=== Hints Begin ===
{{reference_patch}}
=== Hints End ===

After reading the hints above, make sure you truly understand what they point to and why — do not copy or paraphrase them, and do not mention them. Work through the task with your own reasoning, as you otherwise would.
