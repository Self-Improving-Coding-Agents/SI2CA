"""Compatibility import for the shared evaluation lifecycle.

All new Standard, SJ, SL, Discovered and DeepSWE runs use the same strategy
runner. Benchmark-specific grading is selected from task metadata.
"""
from si2ca.run import run_strategy as evaluate
