"""CLARA-CE: certificate-guided counterfactual search (Track A, Phase 2).

Pipeline (theory note v0.4, section 4.4):

  1. Closed form + zone check (O(mn)) — if the direct minimizer stays in
     the basis-stable zone, return it (provably optimal when t equals the
     sandwich bound; conditionally exact under conjecture C1' otherwise).
  2. Certificates: sandwich lower bound (proved) and, for column targets,
     the refined trigger bound (conjectured, P6).
  3. Depth-limited basis search: BFS over pivot neighbors (depth 2 by
     default — enumeration ground truth put 8/9 shortcuts at depth 2 and
     1/9 at depth 3), solving one small LP per compatible basis.
  4. Anytime bracket [lower_bound, upper_bound]; exact when they meet.

Costs are l1 by default (LP subproblems); l-inf also LP; l2 not yet wired
into the per-basis subproblem (closed forms support it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import highspy
import numpy as np

from clara.ce import certificates as cert
from clara.ce.geometry import CEGeometry, TOL
from clara.ce.targets import (
    DriveBasicToZero,
    MakeVariableActive,
    MutableSpec,
    SetShadowPrice,
    Target,
)

COND_MAX = 1e10


# ------------------------------------------------------------------
# Small-LP helper (min c'v, A v <= b, v >= 0)
# ------------------------------------------------------------------

def _solve_lp(c, A_ub, b_ub):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    nv = len(c)
    for _ in range(nv):
        h.addVar(0.0, highspy.kHighsInf)
    for k in range(nv):
        h.changeColCost(k, float(c[k]))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    for row, rhs in zip(A_ub, b_ub):
        nz = np.nonzero(np.abs(row) > 1e-14)[0]
        h.addRow(-highspy.kHighsInf, float(rhs),
                 len(nz), [int(t) for t in nz], [float(row[t]) for t in nz])
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, None
    return float(h.getObjectiveValue()), np.array(h.getSolution().col_value)


# ------------------------------------------------------------------
# Basis context: everything a target/subproblem needs for one basis
# ------------------------------------------------------------------

@dataclass
class BasisContext:
    geom: CEGeometry
    basis: tuple[int, ...]
    Binv: np.ndarray
    x_B: np.ndarray            # B''^-1 b  (unperturbed)
    y: np.ndarray
    cbar: np.ndarray
    S_b: np.ndarray
    S_c: np.ndarray

    @property
    def basic_set(self) -> set[int]:
        return set(self.basis)

    @property
    def n_tot(self) -> int:
        return self.geom.n_tot

    def position_of(self, v: int) -> int:
        return self.basis.index(v)

    # LP variable layout: [db+ | db- | dc+ | dc-]
    @property
    def nvars(self) -> int:
        return 2 * len(self.S_b) + 2 * len(self.S_c)

    def b_row(self, w_aug: np.ndarray) -> np.ndarray:
        """LP row encoding <w_aug, db_aug> (db supported on S_b)."""
        sb, sc = len(self.S_b), len(self.S_c)
        row = np.zeros(2 * sb + 2 * sc)
        wS = np.asarray(w_aug)[self.S_b]
        row[:sb] = wS
        row[sb:2 * sb] = -wS
        return row

    def c_row(self, v_tot: np.ndarray) -> np.ndarray:
        """LP row encoding <v_tot, dc> (dc supported on S_c)."""
        sb, sc = len(self.S_b), len(self.S_c)
        row = np.zeros(2 * sb + 2 * sc)
        vS = np.asarray(v_tot)[self.S_c]
        row[2 * sb:2 * sb + sc] = vS
        row[2 * sb + sc:] = -vS
        return row


def make_context(geom: CEGeometry, basis, S_b, S_c) -> BasisContext | None:
    idx = tuple(sorted(basis))
    B = geom.A_eq[:, idx]
    try:
        if np.linalg.cond(B) > COND_MAX:
            return None
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return None
    y = geom.c_min[list(idx)] @ Binv
    cbar = geom.c_min - y @ geom.A_eq
    return BasisContext(geom=geom, basis=idx, Binv=Binv,
                        x_B=Binv @ geom.b_eq, y=y, cbar=cbar,
                        S_b=S_b, S_c=S_c)


# ------------------------------------------------------------------
# Per-basis subproblem  V_p(B'', target)
# ------------------------------------------------------------------

def basis_subproblem(ctx: BasisContext, target: Target):
    """min ||db||_1 + ||dc||_1 s.t. B'' optimal for the perturbed problem
    and the target's extra rows. Returns (value, db, dc) or (None, ..)."""
    geom = ctx.geom
    rows, rhs = [], []
    # primal feasibility of B'' under b + db
    if len(ctx.S_b):
        for p in range(geom.m_aug):
            rows.append(ctx.b_row(-ctx.Binv[p]))
            rhs.append(ctx.x_B[p])
    elif np.min(ctx.x_B) < -1e-9:
        return None, None, None                      # infeasible, b fixed
    # dual feasibility of B'' under c + dc
    for l in range(geom.n_tot):
        if l in ctx.basic_set:
            continue
        h = np.zeros(geom.n_tot)
        h[l] = 1.0
        alpha = ctx.Binv @ geom.A_eq[:, l]
        for pos, k in enumerate(ctx.basis):
            h[k] -= alpha[pos]
        rows.append(ctx.c_row(-h))
        rhs.append(ctx.cbar[l])
    # target rows
    trows, trhs = target.extra_rows(ctx)
    rows += list(trows)
    rhs += list(trhs)

    val, sol = _solve_lp(np.ones(ctx.nvars), np.array(rows), np.array(rhs))
    if val is None:
        return None, None, None
    sb, sc = len(ctx.S_b), len(ctx.S_c)
    db = np.zeros(geom.m)
    if sb:
        db[ctx.S_b] = sol[:sb] - sol[sb:2 * sb]
    dc = np.zeros(geom.n_tot)
    if sc:
        dc[ctx.S_c] = sol[2 * sb:2 * sb + sc] - sol[2 * sb + sc:]
    return val, db, dc


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

@dataclass
class CEReport:
    found: bool
    cost: float | None = None
    db: np.ndarray | None = None
    dc: np.ndarray | None = None
    achieving_basis: tuple[int, ...] | None = None
    pivot_distance: int | None = None
    lower_bound: float = 0.0
    exact: bool = False
    exact_note: str = ""
    bases_examined: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float | None:
        if self.cost is None or not np.isfinite(self.lower_bound):
            return None
        return max(0.0, self.cost - self.lower_bound)


# ------------------------------------------------------------------
# The search
# ------------------------------------------------------------------

def find_counterfactual(geom: CEGeometry, target: Target,
                        mutable: MutableSpec | None = None,
                        p: float = 1, depth: int = 2,
                        max_enter: int = 12, max_leave: int = 4) -> CEReport:
    mutable = mutable or MutableSpec()
    S_b = mutable.row_indices(geom.m)
    S_c = mutable.col_indices(geom.n)
    rep = CEReport(found=False)

    # ---- certificates ------------------------------------------------
    lb = cert.sandwich_lb(geom, S_b if len(S_b) else None,
                          S_c if len(S_c) else None, p)
    if isinstance(target, MakeVariableActive) and len(S_c) and not len(S_b):
        rlb = cert.refined_lb(geom, target.j, S_c, p)
        rep.notes.append(f"refined_lb(conjectured)={rlb:.6g}")
    rep.lower_bound = 0.0 if not np.isfinite(lb) else lb

    # ---- fast path: closed form + zone check -------------------------
    t_direct, db_direct, dc_direct = None, None, None
    if isinstance(target, MakeVariableActive) and len(S_c):
        t_direct, dc_direct = geom.closed_form_col(target.j, S_c, p)
    elif isinstance(target, DriveBasicToZero) and len(S_b) and target.v in geom.basic_set:
        pos = list(geom.Bidx).index(target.v)
        t_direct, db_direct = geom.closed_form_row(pos, S_b, p)
    elif isinstance(target, SetShadowPrice) and len(S_c):
        t_direct, dc_direct = geom.closed_form_dual(target.i, target.p, S_c, p)

    if t_direct is not None:
        db_aug = geom.db_to_aug(db_direct) if db_direct is not None else None
        if geom.in_zone(db_aug, dc_direct):
            rep.found = True
            rep.cost = t_direct
            rep.db, rep.dc = db_direct, dc_direct
            rep.achieving_basis = tuple(sorted(geom.Bidx.tolist()))
            rep.pivot_distance = 0
            if np.isfinite(lb) and t_direct <= lb * (1 + 1e-9):
                rep.exact, rep.exact_note = True, "t equals sandwich bound (proved)"
                rep.lower_bound = rep.cost
                return rep
            rep.exact, rep.exact_note = True, "zone-exact (conditional on C1')"
            if not (isinstance(target, MakeVariableActive) and not len(S_b)):
                rep.exact, rep.exact_note = False, "zone-optimal; exterior not excluded"
        else:
            rep.notes.append(f"direct t={t_direct:.6g} leaves zone; searching")

    # ---- basis search ------------------------------------------------
    base = tuple(sorted(geom.Bidx.tolist()))
    joint = len(S_b) > 0
    facets = geom.col_facets(S_c, p) if len(S_c) else {}
    incumbent = rep.cost if rep.found else np.inf

    visited = {frozenset(base)}
    frontier = [base]
    for d in range(1, depth + 1):
        nxt = []
        for bas in frontier:
            ctx0 = make_context(geom, bas, S_b, S_c)
            if ctx0 is None:
                continue
            tab = ctx0.Binv @ geom.A_eq
            xB0 = ctx0.x_B
            enters = [l for l in range(geom.n_tot) if l not in ctx0.basic_set]
            enters.sort(key=lambda l: facets.get(l, np.inf))
            for l in enters[:max_enter]:
                alpha = tab[:, l]
                pos_rows = np.nonzero(alpha > TOL)[0]
                leave_rows = []
                if len(pos_rows):
                    ratios = xB0[pos_rows] / alpha[pos_rows]
                    rmin = ratios.min()
                    leave_rows += list(pos_rows[ratios <= rmin + 1e-9 * max(1.0, rmin)])
                if joint:
                    neg = list(np.nonzero(alpha < -TOL)[0])
                    neg.sort(key=lambda r: xB0[r])
                    leave_rows += neg[:max_leave]
                for r in leave_rows[:max_leave + 2]:
                    leaving = ctx0.basis[r]
                    cand = tuple(sorted((set(bas) - {leaving}) | {l}))
                    key = frozenset(cand)
                    if key in visited:
                        continue
                    visited.add(key)
                    nxt.append(cand)
                    ctx = make_context(geom, cand, S_b, S_c)
                    if ctx is None:
                        continue
                    rep.bases_examined += 1
                    if not joint and np.min(ctx.x_B) < -1e-9:
                        continue
                    if not target.compatible(ctx):
                        continue
                    val, db, dc = basis_subproblem(ctx, target)
                    if val is not None and val < incumbent - 1e-12:
                        incumbent = val
                        rep.found = True
                        rep.cost = val
                        rep.db, rep.dc = db, dc
                        rep.achieving_basis = cand
                        rep.pivot_distance = len(set(cand) ^ set(base)) // 2
                        rep.exact, rep.exact_note = False, ""
        frontier = nxt

    if rep.found and np.isfinite(lb) and rep.cost <= lb * (1 + 1e-9):
        rep.exact, rep.exact_note = True, "meets sandwich bound (proved)"
        rep.lower_bound = rep.cost
    elif rep.found and not rep.exact:
        rep.exact_note = rep.exact_note or f"depth-{depth} search bracket"
    return rep
