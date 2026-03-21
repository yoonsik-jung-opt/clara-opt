# CLARA — Classical LP Analysis for Reoptimization and Attribution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

A white-box optimization solver that explains why solutions are optimal,
when reoptimization is needed, and what changed.

## Features

- **Solver-intrinsic explanations** — not post-hoc ML surrogate, but direct
  interpretation of basis, dual values, and sensitivity ranges
- **Dual backend** — Internal Revised Simplex (full transparency, step-by-step trace)
  or HiGHS (performance for large problems)
- **Three explanation levels** mapped to the XAIOR framework (De Bock et al., 2024):
  - *Understandability*: Which constraints are binding? What resources are fully used?
  - *Justifiability*: Why is each variable at this value? What's the reduced cost?
  - *Actionability*: What can change without breaking the solution? Where to invest?
- **CLI and Python API** — `clara explain problem.lp` or import as library

## Installation

```bash
pip install clara-opt
```

For development:

```bash
git clone https://github.com/yoonsik-jung-opt/clara-opt.git
cd clara-opt
pip install -e ".[dev]"
```

## Quick Start

```bash
# Explain a production planning problem
clara explain examples/production.lp

# Brief summary
clara explain problem.lp --level brief

# JSON output
clara explain problem.lp --format json

# Use HiGHS engine for larger problems
clara explain problem.lp --engine highs

# Solve only (no explanation)
clara solve problem.lp

# Problem info
clara info problem.lp
```

### Python API

```python
from clara.io import read_lp
from clara.engine.simplex import RevisedSimplex
from clara.explain import Explainer

problem = read_lp("problem.lp")
state = RevisedSimplex(problem).solve()
report = Explainer().explain(state, problem=problem)
print(report.to_text())
```

## Roadmap

- [x] v0.1.0 — LP solver + explainer + CLI (you are here)
- [ ] v0.5.0 — MIP (Branch-and-Bound), counterfactual analysis, TUI
- [ ] v1.0.0 — Incremental reoptimization, parametric LP, streaming
- [ ] v2.0.0 — LLM hybrid explanations, web UI

## Academic Use

CLARA fills a gap identified in the XAIOR framework (De Bock et al., 2024, EJOR):
explainable AI for mathematical optimization solvers. If you use CLARA in research,
please cite:

```
(citation TBD — paper in preparation for Mathematical Programming Computation)
```

## License

MIT © 2026 Yoonsik
