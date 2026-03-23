"""LP problem representation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LPProblem:
    """Standard-form LP problem: max c^T x  s.t.  Ax <= b,  x >= 0.

    Args:
        c: Objective coefficients (n,).
        A: Constraint matrix (m, n).
        b: Right-hand side (m,).
        var_names: Variable names. Defaults to x1, x2, ...
        constraint_names: Constraint names. Defaults to C1, C2, ...
        name: Problem name for display.
    """
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    var_names: list[str] = field(default_factory=list)
    constraint_names: list[str] = field(default_factory=list)
    name: str = ""
    sense: str = "maximize"  # "maximize" or "minimize"
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    integer_vars: set[int] = field(default_factory=set)
    binary_vars: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)

        m, n = self.A.shape
        if not self.var_names:
            self.var_names = [f"x{j+1}" for j in range(n)]
        if not self.constraint_names:
            self.constraint_names = [f"C{i+1}" for i in range(m)]
        if self.lower_bounds is None:
            self.lower_bounds = np.zeros(n)
        else:
            self.lower_bounds = np.asarray(self.lower_bounds, dtype=float)
        if self.upper_bounds is None:
            self.upper_bounds = np.full(n, np.inf)
        else:
            self.upper_bounds = np.asarray(self.upper_bounds, dtype=float)

    @property
    def num_variables(self) -> int:
        return self.A.shape[1]

    @property
    def num_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def has_integers(self) -> bool:
        return bool(self.integer_vars or self.binary_vars)

    def as_lp(self) -> "LPProblem":
        """Return a copy with integrality constraints removed (LP relaxation)."""
        return LPProblem(
            c=self.c.copy(), A=self.A.copy(), b=self.b.copy(),
            var_names=list(self.var_names),
            constraint_names=list(self.constraint_names),
            name=self.name, sense=self.sense,
            lower_bounds=self.lower_bounds.copy() if self.lower_bounds is not None else None,
            upper_bounds=self.upper_bounds.copy() if self.upper_bounds is not None else None,
            integer_vars=set(), binary_vars=set(),
        )
