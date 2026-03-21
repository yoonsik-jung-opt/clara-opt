"""Demonstrate JSON output for programmatic use.

JSON output can be piped to other tools (jq, Python scripts, dashboards).
Equivalent to: clara explain file.lp --format json

Usage:
  python examples/04_json_output.py
"""

import json

from clara.io import read_lp
from clara.engine.simplex import RevisedSimplex
from clara.explain import Explainer, DetailLevel

problem = read_lp("tests/fixtures/albici_base.lp")
state = RevisedSimplex(problem).solve()
report = Explainer().explain(state, level=DetailLevel.BRIEF, problem=problem)

data = report.to_dict()

# Programmatic access
print(f"Bottleneck: {data['sensitivity_report']['bottleneck']}")
print(f"Number of binding constraints: {len(data['binding_report']['binding'])}")

# Full JSON
print()
print(json.dumps(data, indent=2))
