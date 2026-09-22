"""Self-Likelihood selector, Δ-max variant (user request 2026-09-06).

Same machinery as ``SLArgminNLL`` (k=4 plain draws per turn; the runner forwards every candidate once
inside the privileged context, head or tail, and attaches ``mean_nll_privileged``; the sampling
logprobs already give ``mean_nll`` = plain NLL for free). The only change is the choice rule:

    argmin  (sl_argmin_nll): pick the candidate with the smallest NLL *given* the hint/gold
    Δ-max   (this file):     pick the candidate whose NLL *drops the most* once the hint/gold is
                             shown, Δ = mean_nll − mean_nll_privileged  (largest confidence gain)

Decision labels are kept distinct so the fire rate can be read off the turn logs:
  only_valid_toolcall / both_invalid / identical_command  (no choice to make)
  delta_unavailable_first                                 (a signal is missing -> first candidate,
                                                           never silently a coin flip)
  delta_argmax                                            (rule fired)
"""
from si2ca.runtime.strategy import SLArgminNLL
from si2ca.runtime.protocol import bash_actions

class Strategy(SLArgminNLL):
    name = "sl_argmax_delta"
    scaffold = "SL"
    K = 4          

    async def select(self, ctx, cands, judge):
        indices = [i for i, c in enumerate(cands) if c.get("command")]
        valid = [cands[i] for i in indices]
        if len(valid) < 2:
            return (indices[0] if valid else 0,
                    "only_valid_toolcall" if valid else "both_invalid", {})
        if len({tuple(cmd for _, cmd in bash_actions(c['msg'])) for c in valid}) == 1:
            return ctx.rng.choice(indices), "identical_command", {}
        scored = []
        for i, c in enumerate(valid):
            plain, priv = c.get("mean_nll"), c.get("mean_nll_privileged")
            if isinstance(plain, (int, float)) and isinstance(priv, (int, float)):
                scored.append((plain - priv, i))
        if not scored:
            return indices[0], "delta_unavailable_first", {}
        best = max(t for t, _ in scored)
        ties = [i for t, i in scored if t == best]
        best_i = ctx.rng.choice(ties) if len(ties) > 1 else ties[0]
        return indices[best_i], "delta_tie_random" if len(ties) > 1 else "delta_argmax", {}
