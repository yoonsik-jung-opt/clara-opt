"""Counterfactual target specifications (Track A, T-classes).

A target is a property the perturbed problem's optimal solution set must
satisfy (weak semantics: SOME optimal solution satisfies it). Each target
knows, for a candidate basis of the augmented system [A_aug | I]:

  * ``compatible(ctx)``  — can this basis witness the target at all
    (e.g. the column must be basic for a value target);
  * ``extra_rows(ctx)``  — additional linear constraints on
    (delta_b, delta_c) beyond zone feasibility of the basis.

``ctx`` is a ``BasisContext`` from :mod:`clara.ce.search`.

Implemented target classes (theory note v0.4):
  T2' value form  — MakeVariableActive(j):  x_j > 0 in some optimal solution
  E_row           — DriveBasicToZero(v):    variable/slack v equal to 0 in
                    some optimal solution (constraint-binding when v is a
                    slack)
  T3 (E_dual)     — SetShadowPrice(i, p):   dual price of row i equal to p
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MutableSpec:
    """Which coordinates may move.

    rows: indices of original constraints whose rhs may change (None = all;
        empty tuple = rhs fixed).
    cols: indices of decision variables whose cost may change (None = all;
        empty tuple = costs fixed).
    """
    rows: tuple[int, ...] | None = None
    cols: tuple[int, ...] | None = None

    def row_indices(self, m: int) -> np.ndarray:
        return np.arange(m) if self.rows is None else np.asarray(self.rows, dtype=int)

    def col_indices(self, n: int) -> np.ndarray:
        return np.arange(n) if self.cols is None else np.asarray(self.cols, dtype=int)


@dataclass(frozen=True)
class Target:
    """Base class. Subclasses define compatibility and extra constraints."""

    def compatible(self, ctx) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def extra_rows(self, ctx):  # pragma: no cover - abstract
        """Return (rows, rhs) with rows over ctx's (db+, db-, dc+, dc-)."""
        raise NotImplementedError


@dataclass(frozen=True)
class MakeVariableActive(Target):
    """x_j > 0 in some optimal solution (value form of T2')."""
    j: int
    eps: float = 1e-6

    def compatible(self, ctx) -> bool:
        return self.j in ctx.basic_set

    def extra_rows(self, ctx):
        # x_j(db) = (B''^-1 (b + db))_{pos_j} >= eps
        pos = ctx.position_of(self.j)
        row = ctx.b_row(-ctx.Binv[pos])          # -x_j(db) <= -eps + xB
        return [row], [ctx.x_B[pos] - self.eps]


@dataclass(frozen=True)
class DriveBasicToZero(Target):
    """Augmented variable v equal to 0 in some optimal solution (E_row).

    v < n: decision variable forced to zero;
    v >= n: slack of row v - n zero, i.e. that constraint binding.
    """
    v: int

    def compatible(self, ctx) -> bool:
        # witnessed either with v nonbasic (identically 0) or v basic at 0
        return True

    def extra_rows(self, ctx):
        if self.v not in ctx.basic_set:
            return [], []                        # nonbasic => x_v = 0 already
        pos = ctx.position_of(self.v)
        r1 = ctx.b_row(ctx.Binv[pos])            # x_v(db) <= 0
        r2 = ctx.b_row(-ctx.Binv[pos])           # -x_v(db) <= 0
        return [r1, r2], [-ctx.x_B[pos], ctx.x_B[pos]]


@dataclass(frozen=True)
class SetShadowPrice(Target):
    """Dual price of original row i equal to p (E_dual, T3).

    Uses the min-form convention of the augmented system; callers should
    produce p via the geometry helper so signs match the native sense.
    """
    i: int
    p: float
    tol: float = 0.0

    def compatible(self, ctx) -> bool:
        return True

    def extra_rows(self, ctx):
        # y''_i(dc) = y''_i + <Binv[:, i], dc_B''>; force equal to p (+- tol)
        coeff = np.zeros(ctx.n_tot)
        for pos, k in enumerate(ctx.basis):
            coeff[k] = ctx.Binv[pos, self.i]
        gap = self.p - ctx.y[self.i]
        r1 = ctx.c_row(coeff)                    # <coeff, dc> <= gap + tol
        r2 = ctx.c_row(-coeff)                   # -<coeff, dc> <= -gap + tol
        return [r1, r2], [gap + self.tol, -gap + self.tol]

