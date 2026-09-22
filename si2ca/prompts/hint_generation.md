You are helping build a hint set for a software-engineering benchmark.

Below are an issue report from a real repository and the patch that was
actually accepted to fix it. Write hints that would help an engineer who
can see the issue but NOT the patch.

=== Issue Begin ===
{{problem_statement}}
=== Issue End ===

=== Accepted Patch Begin ===
{{gold_patch}}
=== Accepted Patch End ===

Write exactly 4 hints, as a numbered list, covering in this order:
1. WHERE the change belongs — name the file(s) and the function/class/section.
2. WHAT the defect actually is — the underlying cause, in your own words.
3. HOW the fix works — the shape of the change, described in prose.
4. WHAT to watch out for — a trap, an edge case, or something the patch
   deliberately preserves.

Hard rules:
- Do NOT paste code, diffs, patch syntax, or literal lines from the patch.
  Describe changes in words. Identifiers and file paths are allowed.
- Do NOT mention that a patch exists, or refer to "the patch"/"the fix that
  was applied". Write as advice to someone about to solve the issue.
- Each hint is 1-2 sentences and at most 240 characters.
- The whole list must be between 700 and 1100 characters, and should land
  close to 900. Being under the cap matters more than saying everything:
  drop the least useful detail rather than run long.
- Output the numbered list only: no preamble, no closing remark, no headers.
