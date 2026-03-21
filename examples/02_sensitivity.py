"""Explore sensitivity analysis results.

Demonstrates how to programmatically access sensitivity ranges
to answer questions like:
  - How much can a product's profit change before the solution changes?
  - Which constraint is the bottleneck?
  - Where should we invest to improve the objective?

Usage:
  python examples/02_sensitivity.py
"""

from clara.io import read_lp
from clara.engine.simplex import RevisedSimplex

problem = read_lp("tests/fixtures/albici_base.lp")
state = RevisedSimplex(problem).solve()

print(f"Optimal value: {state.optimal_value:.4f}")
print()

# Find bottleneck
bottleneck = max(state.binding_constraints, key=lambda c: abs(c.dual_value))
print(f"Bottleneck: {bottleneck.name}")
print(f"  Shadow price: {bottleneck.dual_value:.4f}")
print(f"  Expanding {bottleneck.name} by 1 unit improves objective by {bottleneck.dual_value:.4f}")
print()

# Sensitivity ranges for each variable
print("Objective coefficient sensitivity:")
for name, (lo, hi) in state.sensitivity.obj_coeff_ranges.items():
    width = hi - lo if hi != float("inf") else float("inf")
    print(f"  {name}: allowable range [{lo:.4f}, {hi:.4f}], width = {width:.4f}")
