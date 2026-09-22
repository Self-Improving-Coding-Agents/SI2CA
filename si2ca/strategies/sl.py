"""Self-Likelihood selector, argmin variant with k=4 (user 2026-09-06: match the earlier 35B experiments, which drew 4).

Identical to the builtin ``sl_argmin_nll`` (pick the candidate with the smallest NLL inside the privileged
context) except for the candidate pool size.
"""
from si2ca.runtime.strategy import SLArgminNLL

class Strategy(SLArgminNLL):
    name = "sl_argmin_nll_k4"
    scaffold = "SL"
    K = 4
