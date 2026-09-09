"""Exact counterfactuals by full basis enumeration (small instances).

Ground-truth reference for CLARA-CE (E1 experiments): enumerate every
basis of the augmented system, keep the target-compatible ones, solve the
per-basis subproblem, take the minimum. Exponential in size — intended
for validation on instances with C(n_tot, m_aug) up to ~10^5.

Completeness rests on Theorem A2-DEC (theory note v0.4, section 4.1).
"""

from __future__ import annotations

import itertools

import numpy as np

from clara.ce.geometry import CEGeometry
from clara.ce.search import CEReport, basis_subproblem, make_context
from clara.ce.targets import MutableSpec, Target

MAX_BASES = 200_000


def exact_counterfactual(geom: CEGeometry, target: Target,
                         mutable: MutableSpec | None = None) -> CEReport:
    mutable = mutable or MutableSpec()
    S_b = mutable.row_indices(geom.m)
    S_c = mutable.col_indices(geom.n)
    joint = len(S_b) > 0
    base = tuple(sorted(geom.Bidx.tolist()))

    import math
    n_bases = math.comb(geom.n_tot, geom.m_aug)
    if n_bases > MAX_BASES:
        raise ValueError(f"instance too large for enumeration ({n_bases} bases)")

    rep = CEReport(found=False)
    best = np.inf
    for idx in itertools.combinations(range(geom.n_tot), geom.m_aug):
        ctx = make_context(geom, idx, S_b, S_c)
        if ctx is None:
            continue
        if not joint and np.min(ctx.x_B) < -1e-9:
            continue
        if not target.compatible(ctx):
            continue
        rep.bases_examined += 1
        val, db, dc = basis_subproblem(ctx, target)
        if val is not None and val < best - 1e-12:
            best = val
            rep.found = True
            rep.cost = val
            rep.db, rep.dc = db, dc
            rep.achieving_basis = tuple(idx)
            rep.pivot_distance = len(set(idx) ^ set(base)) // 2
    if rep.found:
        rep.exact, rep.exact_note = True, "full basis enumeration"
        rep.lower_bound = rep.cost
    return rep
