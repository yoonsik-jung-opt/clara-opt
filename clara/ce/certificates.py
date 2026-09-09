"""Lower-bound certificates for counterfactual cost (Track A, A1).

All bounds are for the l1 cost by default (dual norm inf); pass p to
change. Provable bounds:

  * ``d0_restricted``      — min row facet distance over mutable rows;
    returns inf when no mutable coordinate can reach any primal facet
    (CE-nonexistence certificate for basis-changing row targets).
  * ``sandwich_lb``        — min over all reachable facet distances; valid
    lower bound for any target requiring a basis change (theory note 4.2,
    proved via P1 + crossing argument).

Conjectured (verified 46/46 on enumeration ground truth, proof obligation
P6 — treat as heuristic until proved):

  * ``refined_lb`` — for MakeVariableActive(j): min(t_j, min over trigger
    columns' facets), where a trigger column's min-ratio pivot removes an
    alpha-negative row (alpha = B^-1 a_j). Median 2.7x tighter than
    ``sandwich_lb`` in experiments.
"""

from __future__ import annotations

import numpy as np

from clara.ce.geometry import CEGeometry, TOL


def d0_restricted(geom: CEGeometry, S_b: np.ndarray, p: float = 1) -> float:
    """Restricted-subspace basis robustness radius (rhs side)."""
    best = np.inf
    for i in range(geom.m_aug):
        if geom.x_B[i] <= 1e-9:
            return 0.0
        nrm = CEGeometry._dual_norm(geom.Binv[i, S_b], p)
        if nrm > TOL:
            best = min(best, geom.x_B[i] / nrm)
    return float(best)


def sandwich_lb(geom: CEGeometry, S_b: np.ndarray | None,
                S_c: np.ndarray | None, p: float = 1) -> float:
    """min over reachable facets; inf if no facet reachable (certificate)."""
    vals = []
    if S_b is not None and len(S_b):
        vals += list(geom.row_facets(S_b, p).values())
    if S_c is not None and len(S_c):
        vals += list(geom.col_facets(S_c, p).values())
    return float(min(vals)) if vals else np.inf


def refined_lb(geom: CEGeometry, j: int, S_c: np.ndarray, p: float = 1) -> float:
    """CONJECTURED lower bound for MakeVariableActive(j), c-only case.

    Verified on all enumeration ground-truth cases; not yet proved (P6).
    """
    facets = geom.col_facets(S_c, p)
    if j not in facets:
        return np.inf
    alpha = geom.tableau[:, j]
    best = facets[j]
    for l, t_l in facets.items():
        if l == j:
            continue
        al = geom.tableau[:, l]
        pos_rows = np.nonzero(al > TOL)[0]
        if len(pos_rows) == 0:
            continue
        ratios = geom.x_B[pos_rows] / al[pos_rows]
        rmin = ratios.min()
        arg = pos_rows[ratios <= rmin + 1e-9 * max(1.0, rmin)]
        if np.any(alpha[arg] < -1e-9):
            best = min(best, t_l)
    return float(best)
