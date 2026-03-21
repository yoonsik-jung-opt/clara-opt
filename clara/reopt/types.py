"""Reoptimization types — change detection, impact analysis, and diffing.

These structures support Phase 2 of CLARA:
    ParameterChange  — detected change between two problem instances
    ReoptDecision    — should we reoptimize? (Oguz bound + sensitivity check)
    SolutionDiff     — what changed in the solution and why
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from clara.model.solve_state import SolveState


class IncompatibleProblemsError(Exception):
    """Raised when two problems cannot be meaningfully compared."""
    pass


# ============================================================
# Change classification (Albici 2010)
# ============================================================

class ChangeType(Enum):
    """Albici-based classification of parameter changes.

    Type R  — RHS (b-vector) change only
    Type C  — Objective coefficient (c-vector) change only
    Type V  — New variables (column addition)
    Type X  — New constraints (row addition)
    Type M  — Constraint removal (row deletion)
    Type RC — Compound b + c change (parametric LP required)
    """
    TYPE_R = auto()
    TYPE_C = auto()
    TYPE_V = auto()
    TYPE_X = auto()
    TYPE_M = auto()
    TYPE_RC = auto()
    TYPE_A = auto()      # constraint matrix A changed (not just b or c)
    TYPE_MULTI = auto()  # multiple structural changes


# ============================================================
# Change detection output
# ============================================================

@dataclass(frozen=True)
class ParameterChange:
    """Detected change between two problem instances.

    The Change Detector (Phase 2) produces this by comparing
    the old problem and new problem parameter by parameter.
    """
    change_type: ChangeType
    delta_b: Optional[np.ndarray] = None
    delta_c: Optional[np.ndarray] = None
    new_columns: Optional[dict] = None
    new_rows: Optional[dict] = None
    removed_rows: Optional[list[str]] = None

    @property
    def max_rhs_change_ratio(self) -> Optional[float]:
        """Max relative RHS change. Used for sensitivity range comparison."""
        if self.delta_b is None:
            return None
        return float(np.max(np.abs(self.delta_b)))

    @property
    def max_obj_change_ratio(self) -> Optional[float]:
        """Max relative obj coeff change (δ in Oguz bound)."""
        if self.delta_c is None:
            return None
        return float(np.max(np.abs(self.delta_c)))

    @property
    def summary(self) -> str:
        """Human-readable one-line summary of the change."""
        if self.change_type == ChangeType.TYPE_R:
            n = int(np.sum(np.abs(self.delta_b) > 1e-10)) if self.delta_b is not None else 0
            return f"RHS changed in {n} constraint(s)"
        elif self.change_type == ChangeType.TYPE_C:
            n = int(np.sum(np.abs(self.delta_c) > 1e-10)) if self.delta_c is not None else 0
            return f"Objective coefficients changed for {n} variable(s)"
        elif self.change_type == ChangeType.TYPE_RC:
            return "Both RHS and objective coefficients changed (compound)"
        elif self.change_type == ChangeType.TYPE_V:
            return f"{len(self.new_columns)} new variable(s) added"
        elif self.change_type == ChangeType.TYPE_X:
            return f"{len(self.new_rows)} new constraint(s) added"
        elif self.change_type == ChangeType.TYPE_M:
            return f"{len(self.removed_rows)} constraint(s) removed"
        elif self.change_type == ChangeType.TYPE_A:
            return "Constraint matrix coefficients changed"
        elif self.change_type == ChangeType.TYPE_MULTI:
            return "Multiple structural changes"
        return "Unknown change"


# ============================================================
# Impact analysis output
# ============================================================

@dataclass(frozen=True)
class ReoptDecision:
    """Output of the Impact Analyzer (Phase 2).

    Three-step decision process:
        Step 1 (Oguz bound, Type C only):
            Quick upper bound on opportunity cost of NOT reoptimizing.
            If bound < threshold → skip reoptimization.

        Step 2 (Sensitivity range check):
            For each changed parameter, check if it falls within
            the sensitivity range stored in SolveState.
            All within range → same basis optimal, just recompute x_B = B⁻¹b'.

        Step 3 (Similarity heuristic, inspired by Witzig 2014):
            Estimate whether warm-start will beat scratch solve.
            Low similarity → scratch may be faster.
    """
    should_reoptimize: bool
    reason: str
    oguz_bound: Optional[float] = None
    within_sensitivity: bool = False
    estimated_pivots: Optional[int] = None
    recommended_method: Optional[str] = None


# ============================================================
# Solution diff output
# ============================================================

@dataclass(frozen=True)
class SolutionDiff:
    """Comparison of two SolveStates — the "what changed and why" report.

    Maps to XAIOR Actionability: tells the decision-maker what happened
    to their solution and what they should do next.
    """
    state_old: SolveState
    state_new: SolveState
    change: ParameterChange

    objective_delta: float = 0.0
    objective_delta_pct: float = 0.0

    basis_changed: bool = False
    variables_entered: tuple[str, ...] = ()
    variables_left: tuple[str, ...] = ()
    binding_added: tuple[str, ...] = ()
    binding_removed: tuple[str, ...] = ()

    variable_changes: tuple[tuple[str, float, float], ...] = ()
