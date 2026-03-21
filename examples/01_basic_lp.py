"""Solve and explain a simple production planning problem.

This example demonstrates CLARA's core workflow:
  1. Parse an LP file
  2. Solve with the internal simplex engine
  3. Generate a human-readable explanation

Usage:
  python examples/01_basic_lp.py
  # or equivalently:
  clara explain tests/fixtures/albici_base.lp
"""

from clara.io import read_lp
from clara.engine.simplex import RevisedSimplex
from clara.explain import Explainer, DetailLevel

problem = read_lp("tests/fixtures/albici_base.lp")
state = RevisedSimplex(problem).solve()
report = Explainer().explain(state, level=DetailLevel.DETAILED, problem=problem)
print(report.to_text())
