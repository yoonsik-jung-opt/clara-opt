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

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)

        m, n = self.A.shape
        if not self.var_names:
            self.var_names = [f"x{j+1}" for j in range(n)]
        if not self.constraint_names:
            self.constraint_names = [f"C{i+1}" for i in range(m)]

    @property
    def num_variables(self) -> int:
        return self.A.shape[1]

    @property
    def num_constraints(self) -> int:
        return self.A.shape[0]
