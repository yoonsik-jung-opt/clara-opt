"""Compare internal simplex engine with HiGHS.

Shows that both engines produce the same optimal value and solution,
but the internal engine additionally provides B^-1 and iteration history.

Usage:
  python examples/03_engine_comparison.py
"""

from clara.io import read_lp
from clara.engine.simplex import RevisedSimplex
from clara.engine.highs_backend import HiGHSBackend

problem = read_lp("tests/fixtures/albici_base.lp")

internal = RevisedSimplex(problem).solve()
highs = HiGHSBackend().solve(problem)

print("=== Internal Simplex ===")
print(f"Optimal: {internal.optimal_value:.4f}")
print(f"Iterations: {internal.iteration_count}")
print(f"Has B^-1: {internal.has_basis_inverse}")
print(f"Has trace: {internal.has_internal_trace}")

print()

print("=== HiGHS ===")
print(f"Optimal: {highs.optimal_value:.4f}")
print(f"Iterations: {highs.iteration_count}")
print(f"Has B^-1: {highs.has_basis_inverse}")
print(f"Has trace: {highs.has_internal_trace}")

print()

diff = abs(internal.optimal_value - highs.optimal_value)
print(f"Optimal value difference: {diff:.2e}")
print("Match!" if diff < 1e-6 else "MISMATCH — investigate!")
