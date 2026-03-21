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
# Reoptimization result
# ============================================================

@dataclass(frozen=True)
class ReoptResult:
    """Output of the Reoptimizer."""
    new_state: SolveState
    method_used: str        # "none", "recompute", "warm_start", "parametric_lp", "scratch"
    pivots: int
    scratch_estimate: Optional[int]
    basis_preserved: bool
    reopt_time_seconds: float

    @property
    def speedup(self) -> Optional[float]:
        """Estimated speedup vs scratch solve."""
        if self.scratch_estimate and self.scratch_estimate > 0:
            return self.scratch_estimate / max(self.pivots, 1)
        return None

    @property
    def summary(self) -> str:
        """One-line summary."""
        if self.basis_preserved:
            return (
                f"Basis preserved — recomputed in {self.reopt_time_seconds:.4f}s "
                f"(0 pivots)."
            )
        sp = f" (est. {self.speedup:.1f}x speedup)" if self.speedup else ""
        return (
            f"Reoptimized via {self.method_used}: {self.pivots} pivots "
            f"in {self.reopt_time_seconds:.4f}s{sp}."
        )


# ============================================================
# Diff report structures
# ============================================================

@dataclass(frozen=True)
class VariableChange:
    """Change in a single variable between old and new solution."""
    name: str
    old_value: float
    new_value: float
    delta: float
    delta_pct: float
    old_basis: "BasisStatus"
    new_basis: "BasisStatus"
    basis_changed: bool


@dataclass(frozen=True)
class ConstraintChange:
    """Change in a single constraint between old and new solution."""
    name: str
    old_rhs: float
    new_rhs: float
    old_slack: float
    new_slack: float
    old_dual: float
    new_dual: float
    old_binding: bool
    new_binding: bool
    became_binding: bool
    became_nonbinding: bool


@dataclass
class DiffReport:
    """Complete comparison of two solutions."""
    change: ParameterChange
    reopt_result: ReoptResult

    old_objective: float
    new_objective: float
    objective_delta: float
    objective_delta_pct: float

    variable_changes: tuple[VariableChange, ...]
    variables_entered_basis: tuple[str, ...]
    variables_left_basis: tuple[str, ...]

    constraint_changes: tuple[ConstraintChange, ...]
    became_binding: tuple[str, ...]
    became_nonbinding: tuple[str, ...]

    old_bottleneck: Optional[str]
    new_bottleneck: Optional[str]
    bottleneck_shifted: bool

    summary: str

    def to_text(self) -> str:
        """Full text diff report."""
        sep = "\u2550" * 60
        lines = [sep, "CLARA \u2014 Solution Change Report", sep, ""]

        # Parameter change
        lines.append("PARAMETER CHANGE")
        lines.append(f"  Type: {self.change.change_type.name}")
        lines.append(f"  {self.change.summary}")
        if self.change.delta_b is not None:
            for i, d in enumerate(self.change.delta_b):
                if abs(d) > 1e-10:
                    lines.append(f"    delta_b[{i}] = {d:+.4f}")
        if self.change.delta_c is not None:
            for i, d in enumerate(self.change.delta_c):
                if abs(d) > 1e-10:
                    lines.append(f"    delta_c[{i}] = {d:+.4f}")
        lines.append("")

        # Reoptimization
        lines.append("REOPTIMIZATION")
        lines.append(f"  {self.reopt_result.summary}")
        lines.append("")

        # Objective
        lines.append("OBJECTIVE")
        sign = "+" if self.objective_delta >= 0 else ""
        lines.append(
            f"  Old: {self.old_objective:.4f} \u2192 New: {self.new_objective:.4f} "
            f"({sign}{self.objective_delta:.4f}, {sign}{self.objective_delta_pct:.1f}%)"
        )
        lines.append("")

        # Variable changes
        lines.append("VARIABLE CHANGES")
        if not self.variable_changes:
            lines.append("  No significant variable changes.")
        else:
            for vc in self.variable_changes:
                s = "+" if vc.delta >= 0 else ""
                basis_note = ""
                if vc.basis_changed:
                    if vc.new_basis.name == "BASIC":
                        basis_note = " \u2014 ENTERED basis"
                    else:
                        basis_note = " \u2014 LEFT basis"
                lines.append(
                    f"  {vc.name}: {vc.old_value:.4f} \u2192 {vc.new_value:.4f} "
                    f"({s}{vc.delta:.4f}, {s}{vc.delta_pct:.1f}%){basis_note}"
                )
        lines.append("")

        # Constraint changes
        if self.became_binding or self.became_nonbinding:
            lines.append("CONSTRAINT CHANGES")
            for cc in self.constraint_changes:
                if cc.became_binding:
                    lines.append(f"  {cc.name}: non-binding \u2192 binding")
                elif cc.became_nonbinding:
                    lines.append(f"  {cc.name}: binding \u2192 non-binding")
            lines.append("")

        # Bottleneck
        lines.append("BOTTLENECK SHIFT")
        if self.bottleneck_shifted:
            lines.append(f"  Old: {self.old_bottleneck} \u2192 New: {self.new_bottleneck}")
        else:
            lines.append(f"  Unchanged: {self.old_bottleneck or 'none'}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        return {
            "change": {
                "type": self.change.change_type.name,
                "summary": self.change.summary,
            },
            "reoptimization": {
                "method": self.reopt_result.method_used,
                "pivots": self.reopt_result.pivots,
                "basis_preserved": self.reopt_result.basis_preserved,
                "time_seconds": self.reopt_result.reopt_time_seconds,
            },
            "objective": {
                "old": self.old_objective,
                "new": self.new_objective,
                "delta": self.objective_delta,
                "delta_pct": self.objective_delta_pct,
            },
            "variable_changes": [
                {
                    "name": vc.name, "old": vc.old_value, "new": vc.new_value,
                    "delta": vc.delta, "delta_pct": vc.delta_pct,
                    "basis_changed": vc.basis_changed,
                }
                for vc in self.variable_changes
            ],
            "constraint_changes": [
                {
                    "name": cc.name, "old_binding": cc.old_binding,
                    "new_binding": cc.new_binding,
                    "became_binding": cc.became_binding,
                    "became_nonbinding": cc.became_nonbinding,
                    "old_dual": cc.old_dual, "new_dual": cc.new_dual,
                }
                for cc in self.constraint_changes
            ],
            "bottleneck": {
                "old": self.old_bottleneck,
                "new": self.new_bottleneck,
                "shifted": self.bottleneck_shifted,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)
