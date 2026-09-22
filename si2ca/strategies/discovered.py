"""Branch through the first three mutation-like actions."""

from __future__ import annotations

from si2ca.runtime.strategy import SelfGuideRubrics, TurnContext

from si2ca.runtime.protocol import early_commit_window

class Strategy(SelfGuideRubrics):
    """Use the oracle-rubric judge across the early commitment window."""

    name = "c006_r0"
    scaffold = "SJ"
    BRANCH_K = 2
    MAX_PRIOR_MUTATIONS = 2

    def gate(self, ctx: TurnContext) -> tuple[int, str]:
        return early_commit_window(getattr(ctx, "prior_commands", None),
                                   k=self.BRANCH_K, max_prior_mutations=self.MAX_PRIOR_MUTATIONS)

    def plan(self, ctx: TurnContext) -> int:
        return self.gate(ctx)[0]

    async def select(self, ctx, cands, judge) -> tuple[int, str, dict]:
        winner, parent_label, usage = await super().select(ctx, cands, judge)
        return winner, f"{self.gate(ctx)[1]}:{parent_label}", usage
