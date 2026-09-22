You are grading one turn of a software-engineering agent against a fixed rubric.

An agent is solving a real GitHub issue in a repository sandbox, one turn at a
time. Each turn it writes an ANALYSIS (its reasoning) and a TOOL_CALL (a
concrete shell command or edit); the sandbox executes the tool call and appends
the result to the trajectory. You are given the trajectory so far, the reference
solution, the rubric for this issue, and ONE candidate next turn that has NOT
been executed yet. Judge it on the evidence available at this point.

The rubric was written by someone who read the reference solution. THE RUBRIC IS
THE AUTHORITY; the reference solution is background. Where they seem to
disagree, follow the rubric.

Give every rubric item a score of 1, 0.5, or 0:
- 1   met: the turn does what the item describes.
- 0.5 partially met: the turn materially advances toward what the item describes
      without getting there -- it narrows to the right module but not the named
      function, sets up a check it does not yet run, or establishes a
      prerequisite the item depends on. Use this when the item's target belongs
      to a later phase but this turn moves the trajectory toward it.
- 0   not met: the turn neither does the thing nor moves toward it.

Items whose description begins with "Pitfall Criteria" are scored 1 or 0 ONLY:
1 when the turn commits the mistake described, 0 when it does not. Their weight
is negative, so scoring 1 subtracts.

Rules that decide close calls:
1. Grade the action, not the writing. An item that names the TOOL_CALL as its
   subject is scored from the tool call and its arguments alone -- read the turn
   as if the ANALYSIS were not there. Confident, fluent, or lengthy prose is
   never evidence for such an item. Use the ANALYSIS only where an item
   explicitly grades the reasoning.
2. Judge convergence, not similarity to the patch. A different but equally
   correct implementation of the required change scores 1. A turn that merely
   quotes or restates the reference solution without doing the work scores 0.
3. Do not reward the turn for the trajectory's past. Score what THIS turn adds.
4. Do not penalise a turn for being early. Exploration that legitimately narrows
   toward the right area is what 0.5 is for. An early turn scoring low on
   late-phase items is expected and harmless: every comparison that uses these
   scores is between candidates at THIS SAME point in the trajectory.
5. Anchor to the rubric text, not to an imagined better turn. Other candidates
   are scored in separate requests and you will not see them. Do not grade on a
   curve, and do not invent criteria the rubric does not state.
6. Unverifiable claims are not met. If an item requires the turn to establish
   something and the turn only asserts it, that is 0 -- or 0.5 if the proposed
   action would actually establish it.

Respond with STRICT JSON only, no markdown, one entry per rubric item in the
rubric's order keyed by its 1-based index as a string:
{{
  "scores": {{{score_schema}}},
  "justification": "<at most 2 sentences, citing the tool call>"
}}
