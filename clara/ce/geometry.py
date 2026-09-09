"""Basis geometry for counterfactual analysis.

Builds, from an ``LPProblem`` and its solved ``SolveState`` (HiGHS backend
with reconstructed basis inverse), the min-form augmented quantities that
Track A's theory operates on:

    [A_aug | I] x = b_aug,  x >= 0,   min c_min^T x

with  x_B = B^-1 b,  y = c_B B^-1,  cbar = c_min - y [A_aug | I],
and the linear responses to perturbations delta = (db, dc):

    x_B(db)   = x_B + B^-1 db_aug
    cbar_l(dc)= cbar_l + h_l' dc            (h_l the transfer row)
    y_i(dc)   = y_i + <B^-1[:, i], dc_B>

All CE-side modules consume this object instead of touching the backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clara.engine.standard_form import add_upper_bound_rows
from clara.model.problem import LPProblem
from clara.model.solve_state import SolveState

TOL = 1e-9


@dataclass
class CEGeometry:
    problem: LPProblem
    A_eq: np.ndarray          # (m_aug, n_tot) augmented equality matrix
    b_eq: np.ndarray          # (m_aug,)
    c_min: np.ndarray         # (n_tot,) min-form costs (decision cols only nonzero)
    m: int                    # original row count
    n: int                    # decision variable count
    m_aug: int
    Binv: np.ndarray
    Bidx: np.ndarray
    x_B: np.ndarray
    y: np.ndarray             # min-form duals over augmented rows
    cbar: np.ndarray

    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, problem: LPProblem, state: SolveState) -> "CEGeometry":
        if state.basis_inverse is None or state.basis_indices is None:
            raise ValueError("SolveState has no reconstructed basis inverse")
        A_aug, b_aug, _ = add_upper_bound_rows(problem)
        m_aug, n = A_aug.shape
        A_eq = np.hstack([A_aug, np.eye(m_aug)])
        c_min = np.zeros(n + m_aug)
        c_min[:n] = -problem.c if problem.sense == "maximize" else problem.c
        Binv = np.asarray(state.basis_inverse, dtype=float)
        Bidx = np.asarray(state.basis_indices, dtype=int)
        x_B = Binv @ b_aug
        y = c_min[Bidx] @ Binv
        cbar = c_min - y @ A_eq
        return cls(problem=problem, A_eq=A_eq, b_eq=b_aug, c_min=c_min,
                   m=problem.A.shape[0], n=n, m_aug=m_aug, Binv=Binv,
                   Bidx=Bidx, x_B=x_B, y=y, cbar=cbar)

    # ------------------------------------------------------------------

    @property
    def n_tot(self) -> int:
        return self.n + self.m_aug

    @property
    def basic_set(self) -> set[int]:
        if not hasattr(self, "_basic_set"):
            self._basic_set = set(self.Bidx.tolist())
        return self._basic_set

    @property
    def tableau(self) -> np.ndarray:
        """B^-1 [A_aug | I], cached."""
        if not hasattr(self, "_tab"):
            self._tab = self.Binv @ self.A_eq
        return self._tab

    def h_col(self, l: int) -> np.ndarray:
        """Transfer row of column l: cbar_l(dc) = cbar_l + h_l' dc."""
        h = np.zeros(self.n_tot)
        h[l] = 1.0
        alpha = self.tableau[:, l]
        for pos, k in enumerate(self.Bidx):
            h[k] -= alpha[pos]
        return h

    # ------------------------------------------------------------------
    # Facet distances (restricted subspace, lp-norm CE with dual norm q)
    # ------------------------------------------------------------------

    @staticmethod
    def _dual_norm(v: np.ndarray, p: float) -> float:
        if p == 2:
            return float(np.linalg.norm(v, 2))
        if p == 1:
            return float(np.linalg.norm(v, np.inf))
        if p == np.inf:
            return float(np.linalg.norm(v, 1))
        raise ValueError(f"unsupported norm p={p}")

    def row_facets(self, S_b: np.ndarray, p: float = 1) -> dict[int, float]:
        """t_row(i) = x_B[i] / ||Binv[i, S_b]||_q for nondegenerate rows.

        Rows with ||.|| = 0 but x_B > 0 are unreachable within S (they are
        the infeasibility-certificate rows) and are omitted here.
        """
        out = {}
        for i in range(self.m_aug):
            if self.x_B[i] <= 1e-9:
                continue
            nrm = self._dual_norm(self.Binv[i, S_b], p)
            if nrm > TOL:
                out[i] = float(self.x_B[i] / nrm)
        return out

    def col_facets(self, S_c: np.ndarray, p: float = 1) -> dict[int, float]:
        """t_col(l) = cbar_l / ||h_l[S_c]||_q for nonbasic l with cbar_l > 0."""
        out = {}
        for l in range(self.n_tot):
            if l in self.basic_set or self.cbar[l] <= 1e-9:
                continue
            nrm = self._dual_norm(self.h_col(l)[S_c], p)
            if nrm > TOL:
                out[l] = float(self.cbar[l] / nrm)
        return out

    # ------------------------------------------------------------------
    # Zone membership
    # ------------------------------------------------------------------

    def in_zone(self, db_aug: np.ndarray | None = None,
                dc: np.ndarray | None = None, tol: float = 1e-9) -> bool:
        """Is delta inside cl S+ (current basis remains optimal)?"""
        if db_aug is not None:
            if np.min(self.x_B + self.Binv @ db_aug) < -tol * max(
                    1.0, float(np.max(np.abs(self.x_B)))):
                return False
        if dc is not None:
            scale = max(1.0, float(np.max(np.abs(self.cbar))))
            for l in range(self.n_tot):
                if l in self.basic_set:
                    continue
                if self.cbar[l] + self.h_col(l) @ dc < -tol * scale:
                    return False
        return True

    # ------------------------------------------------------------------
    # Closed forms for elementary targets (theory note section 3)
    # ------------------------------------------------------------------

    def closed_form_col(self, j: int, S_c: np.ndarray, p: float = 1):
        """(t, dc) making column j optimal-eligible: cbar_j -> 0.

        Returns (None, None) when unreachable within S_c (certificate).
        """
        hS = self.h_col(j)[S_c]
        nrm = self._dual_norm(hS, p)
        if nrm <= TOL:
            return None, None
        t = float(self.cbar[j] / nrm)
        dc = np.zeros(self.n_tot)
        if p == 2:
            dc[S_c] = -self.cbar[j] * hS / float(hS @ hS)
        elif p == 1:
            k = int(np.argmax(np.abs(hS)))
            dc[S_c[k]] = -self.cbar[j] / hS[k]
        else:
            dc[S_c] = -self.cbar[j] * np.sign(hS) / np.sum(np.abs(hS))
        return t, dc

    def closed_form_row(self, i: int, S_b: np.ndarray, p: float = 1):
        """(t, db) driving basic value of row i to zero."""
        hb = self.Binv[i, S_b]
        nrm = self._dual_norm(hb, p)
        if nrm <= TOL:
            return None, None
        t = float(self.x_B[i] / nrm)
        db = np.zeros(self.m)
        r = -self.x_B[i]
        if p == 2:
            db[S_b] = r * hb / float(hb @ hb)
        elif p == 1:
            k = int(np.argmax(np.abs(hb)))
            db[S_b[k]] = r / hb[k]
        else:
            db[S_b] = r * np.sign(hb) / np.sum(np.abs(hb))
        return t, db

    def closed_form_dual(self, i: int, p_target: float,
                         S_c: np.ndarray, p: float = 1):
        """(t, dc) moving the min-form dual of row i to p_target."""
        coeff = np.zeros(self.n_tot)
        for pos, k in enumerate(self.Bidx):
            coeff[k] = self.Binv[pos, i]
        cS = coeff[S_c]
        nrm = self._dual_norm(cS, p)
        if nrm <= TOL:
            return None, None
        r = p_target - float(self.y[i])
        t = float(abs(r) / nrm)
        dc = np.zeros(self.n_tot)
        if p == 2:
            dc[S_c] = r * cS / float(cS @ cS)
        elif p == 1:
            k = int(np.argmax(np.abs(cS)))
            dc[S_c[k]] = r / cS[k]
        else:
            dc[S_c] = r * np.sign(cS) / np.sum(np.abs(cS))
        return t, dc

    # ------------------------------------------------------------------

    def db_to_aug(self, db: np.ndarray) -> np.ndarray:
        out = np.zeros(self.m_aug)
        out[: self.m] = db
        return out

    def apply(self, db: np.ndarray | None = None,
              dc: np.ndarray | None = None) -> LPProblem:
        """New LPProblem with b <- b + db, c_min <- c_min + dc (native sense)."""
        b2 = self.problem.b.copy()
        if db is not None:
            b2 = b2 + np.asarray(db)[: self.m]
        c2 = self.problem.c.copy()
        if dc is not None:
            d = np.asarray(dc)[: self.n]
            c2 = c2 - d if self.problem.sense == "maximize" else c2 + d
        ub = self.problem.upper_bounds
        return LPProblem(c=c2, A=self.problem.A.copy(), b=b2,
                         sense=self.problem.sense,
                         upper_bounds=None if ub is None else ub.copy(),
                         name=self.problem.name)
