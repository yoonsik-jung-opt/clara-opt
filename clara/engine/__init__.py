"""Solver engine.

CLARA uses HiGHS as its single solver backend. ``solve(problem)`` is
the convenience entry point; ``HiGHSBackend`` exposes warm-start
options (advanced basis, forced primal/dual simplex strategy).
"""

from clara.engine.highs_backend import (
    HiGHSBackend,
    SIMPLEX_STRATEGY_CHOOSE,
    SIMPLEX_STRATEGY_DUAL,
    SIMPLEX_STRATEGY_PRIMAL,
    solve,
)

__all__ = [
    "HiGHSBackend",
    "SIMPLEX_STRATEGY_CHOOSE",
    "SIMPLEX_STRATEGY_DUAL",
    "SIMPLEX_STRATEGY_PRIMAL",
    "solve",
]
