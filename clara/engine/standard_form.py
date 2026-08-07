"""Shared standard-form utilities.

CLARA's analysis modules work on the augmented system

    [A_aug | I] x_aug = b_aug,   x_aug >= 0,

where A_aug stacks the original ``Ax <= b`` rows with one explicit
``x_j <= ub_j`` row per finite upper bound, and the identity block
holds one slack per augmented row. Columns 0..n-1 are the decision
variables and columns n..n+m_aug-1 are slacks.

These helpers were previously static methods of the removed internal
``RevisedSimplex`` engine; they are now shared by the HiGHS backend,
the reoptimizer, the parametric tracer, and the region analyzer.
"""

from __future__ import annotations

import numpy as np

from clara.model.problem import LPProblem


def add_upper_bound_rows(problem: LPProblem) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Append ``x_j <= ub_j`` rows for every finite upper bound.

    Returns:
        (A_aug, b_aug, constraint_names_aug) where A_aug has shape
        (m + U, n) with U the number of finite upper bounds, in the
        order the variables appear.
    """
    n = problem.A.shape[1]
    extra_rows = []
    extra_rhs = []
    extra_names = []
    if problem.upper_bounds is not None:
        for j in range(n):
            ub = problem.upper_bounds[j]
            if np.isfinite(ub):
                row = np.zeros(n)
                row[j] = 1.0
                extra_rows.append(row)
                extra_rhs.append(ub)
                extra_names.append(f"ub_{problem.var_names[j]}")

    if extra_rows:
        A = np.vstack([problem.A] + extra_rows)
        b = np.concatenate([problem.b, extra_rhs])
        names = list(problem.constraint_names) + extra_names
    else:
        A = problem.A
        b = problem.b
        names = list(problem.constraint_names)

    return A, b, names


def finite_ub_indices(problem: LPProblem) -> list[int]:
    """Indices of variables with a finite upper bound, in row order."""
    if problem.upper_bounds is None:
        return []
    return [j for j in range(problem.A.shape[1])
            if np.isfinite(problem.upper_bounds[j])]


def augmented_var_names(problem: LPProblem, m_aug: int) -> list[str]:
    """Names for the augmented column space: decision vars then slacks."""
    return list(problem.var_names) + [f"y{i+1}" for i in range(m_aug)]
