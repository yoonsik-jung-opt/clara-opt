# CLARA — Classical LP Analysis for Reoptimization and Attribution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

A solver-intrinsic explanation framework for linear programming. CLARA extracts
explanations directly from the simplex basis inverse $B^{-1}$ — without machine
learning surrogates, without inverse formulations — and unifies factual and
counterfactual perspectives on LP explainability within a single geometric framework.

## What CLARA does

CLARA contributes two central capabilities, complemented by two supporting modules:

- **Factual–counterfactual duality** — a formal bound showing that the basis
  robustness $d_0$, computable in $O(mn)$ time from $B^{-1}$ alone, lower-bounds
  the cost of any basis-changing counterfactual explanation. Connects classical
  sensitivity analysis with counterfactual explanations for optimization.
- **Reoptimization pipeline** — automatic change detection, Oguz-bound impact
  analysis, warm-start method selection (primal simplex, dual simplex, or
  parametric LP), and structured diff reports.
- **Simultaneous sensitivity regions** (supporting) — Chebyshev center of the
  basis-preserving polyhedron, the geometric foundation of the duality bound.
- **Objective-change attribution** (supporting) — first-order and Shapley
  decomposition that annotates diff reports during reoptimization.

CLARA also offers two solver backends:

- **Internal Revised Simplex** — full transparency, retains $B^{-1}$ at every
  pivot for downstream analysis (practical for $n \le 100$).
- **HiGHS** — production-grade reference solver; CLARA reconstructs $B^{-1}$
  from the reported basis for larger problems.

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

## Academic Use

CLARA contributes to the explainability of mathematical optimization solvers,
a research area surveyed by De Bock et al. (2024) in the XAIOR framework.

A manuscript describing the framework is in preparation. Citation information
will be added here when a preprint or peer-reviewed version is available.

## License

MIT © 2026 Yoonsik