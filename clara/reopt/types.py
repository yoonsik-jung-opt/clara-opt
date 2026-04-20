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
# Attribution result
# ============================================================

@dataclass(frozen=True)
class AttributionResult:
    """Decomposition of objective change into parameter contributions."""
    delta_z: float
    rhs_effect: float       # yᵀΔb (first-order)
    obj_effect: float       # Δcᵀx (first-order)
    interaction_effect: float  # Δc_Bᵀ B⁻¹ Δb
    first_order_residual: float

    rhs_contributions: dict[str, float]   # per-constraint: y_i * Δb_i
    obj_contributions: dict[str, float]   # per-variable: Δc_j * x_j

    shapley_b: Optional[float] = None
    shapley_c: Optional[float] = None
    z_b_only: Optional[float] = None
    z_c_only: Optional[float] = None

    basis_preserved: bool = False
    nonlinearity: float = 0.0

    @property
    def summary(self) -> str:
        parts = []
        if abs(self.delta_z) > 1e-10:
            rhs_pct = self.rhs_effect / self.delta_z * 100
            obj_pct = self.obj_effect / self.delta_z * 100
            int_pct = self.interaction_effect / self.delta_z * 100
            parts.append(f"RHS changes contributed {rhs_pct:.1f}%.")
            parts.append(f"Objective changes contributed {obj_pct:.1f}%.")
            if abs(self.interaction_effect) > 1e-10:
                parts.append(f"Interaction term: {int_pct:.1f}%.")
        if self.shapley_b is not None and self.shapley_c is not None:
            sb = self.shapley_b / self.delta_z * 100 if abs(self.delta_z) > 1e-10 else 0
            sc = self.shapley_c / self.delta_z * 100 if abs(self.delta_z) > 1e-10 else 0
            parts.append(f"Shapley: Δb = {sb:.1f}%, Δc = {sc:.1f}%.")
        if self.basis_preserved:
            parts.append("First-order decomposition is exact (basis preserved).")
        return " ".join(parts)


# ============================================================
# Sensitivity region
# ============================================================

@dataclass(frozen=True)
class SensitivityRegion:
    """Simultaneous sensitivity region analysis."""
    chebyshev_radius: float
    chebyshev_center: Optional[tuple]
    min_oat_tolerance: float
    simultaneity_ratio: float  # chebyshev / min_oat

    oat_rhs_tolerances: dict[str, float]
    oat_obj_tolerances: dict[str, float]

    projections: Optional[dict] = None  # {(i, j): [(x, y), ...]}

    @property
    def summary(self) -> str:
        r = self.chebyshev_radius
        oat = self.min_oat_tolerance
        ratio = self.simultaneity_ratio
        return (
            f"Chebyshev radius: {r:.4f}. "
            f"Min one-at-a-time tolerance: {oat:.4f}. "
            f"Simultaneity ratio: {ratio:.4f}. "
            f"One-at-a-time analysis overestimates the safe region by "
            f"{1/max(ratio, 1e-10):.1f}x."
        )


# ============================================================
# MIP bound result
# ============================================================

@dataclass(frozen=True)
class MIPBoundResult:
    """Opportunity cost bound for MIP reoptimization decision.

    B = z'*_LP - c'^T x*  (LP relaxation bound, Theorem 2)
    """
    decision: str          # "skip" | "reoptimize" | "infeasible"
    reason: str

    bound: float           # B = z'*_LP - c'^T x*
    bound_ratio: float     # B / |z'*_LP|

    bound_corrected: Optional[float]        # gap-corrected estimate
    bound_corrected_ratio: Optional[float]

    original_integrality_gap: float  # z*_LP - z*_MIP
    gap_ratio: float                 # gap / |z*_LP|

    z_mip_old: float
    z_lp_old: float
    z_lp_new: Optional[float]
    c_new_x_old: float       # c'^T x*

    old_solution_feasible: bool
    max_violation: float

    within_lp_sensitivity: bool
    lp_solve_needed: bool

    oguz_relative_bound: Optional[float]   # 2δ/(1+δ)
    delta: Optional[float]                 # max |Δc_j/c_j|

    @property
    def summary(self) -> str:
        lines = [f"Decision: {self.decision}"]
        lines.append(f"Bound B = {self.bound:.4f} ({self.bound_ratio:.1%} of LP optimal)")
        if self.bound_corrected is not None:
            lines.append(f"Gap-corrected: B_c = {self.bound_corrected:.4f}")
        lines.append(f"Integrality gap: {self.original_integrality_gap:.4f} ({self.gap_ratio:.1%})")
        if not self.old_solution_feasible:
            lines.append(f"Old solution INFEASIBLE (max violation: {self.max_violation:.4f})")
        if self.within_lp_sensitivity:
            lines.append("Within LP sensitivity — no LP re-solve needed")
        if self.oguz_relative_bound is not None:
            lines.append(f"Oguz bound: {self.oguz_relative_bound:.4f} (δ = {self.delta:.4f})")
        lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


# ============================================================
# Parametric LP structures
# ============================================================

@dataclass(frozen=True)
class ParametricBreakpoint:
    """A point where the basis changes along the parametric path."""
    theta: float
    breakpoint_type: str  # "primal" or "dual"
    leaving_var: Optional[str]
    entering_var: Optional[str]
    objective_value: float
    basic_variables: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class ParametricResult:
    """Output of parametric LP solver."""
    new_state: SolveState
    breakpoints: tuple[ParametricBreakpoint, ...]
    theta_start: float
    theta_end: float
    num_pivots: int
    path_monotone: bool
    solve_time_seconds: float

    @property
    def num_breakpoints(self) -> int:
        return len(self.breakpoints)

    @property
    def summary(self) -> str:
        if not self.breakpoints:
            return ("No breakpoints — current basis is optimal for the entire "
                    "parameter range [0, 1].")
        return (
            f"{self.num_breakpoints} breakpoint(s) found along θ ∈ [0, 1]. "
            f"{self.num_pivots} pivot(s) performed. "
            f"First structural change at θ = {self.breakpoints[0].theta:.4f}."
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


@dataclass(frozen=True)
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
    old_var_names: tuple[str, ...] = ()
    attribution: Optional["AttributionResult"] = None

    def to_text(self) -> str:
        """Full text diff report."""
        sep = "\u2550" * 60
        lines = [sep, "CLARA \u2014 Solution Change Report", sep, ""]

        # Parameter change
        lines.append("PARAMETER CHANGE")
        lines.append(f"  Type: {self.change.change_type.name}")
        lines.append(f"  {self.change.summary}")

        # RHS changes — use constraint names from constraint_changes
        rhs_changed = [cc for cc in self.constraint_changes
                       if abs(cc.new_rhs - cc.old_rhs) > 1e-10]
        if rhs_changed:
            lines.append("  RHS changes:")
            for cc in rhs_changed:
                d = cc.new_rhs - cc.old_rhs
                lines.append(f"    {cc.name}: {cc.old_rhs:.0f} \u2192 {cc.new_rhs:.0f} ({d:+.0f})")

        # Obj coeff changes — use old_var_names for delta_c display
        if self.change.delta_c is not None:
            changed = [(i, d) for i, d in enumerate(self.change.delta_c) if abs(d) > 1e-10]
            if changed:
                lines.append("  Objective coefficient changes:")
                for idx, d in changed:
                    name = self.old_var_names[idx] if idx < len(self.old_var_names) else f"var[{idx}]"
                    old_c = d  # delta only; we don't have original c here
                    lines.append(f"    {name}: {d:+.4f}")

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

        # Attribution
        if self.attribution is not None:
            a = self.attribution
            lines.append("ATTRIBUTION")
            if abs(a.delta_z) > 1e-10:
                rhs_pct = a.rhs_effect / a.delta_z * 100
                obj_pct = a.obj_effect / a.delta_z * 100
                int_pct = a.interaction_effect / a.delta_z * 100
                lines.append(f"  RHS changes: {a.rhs_effect:+.4f} ({rhs_pct:.1f}%)")
                for name, val in sorted(a.rhs_contributions.items(), key=lambda x: -abs(x[1])):
                    if abs(val) > 1e-6:
                        lines.append(f"    {name}: {val:+.4f}")
                lines.append(f"  Objective changes: {a.obj_effect:+.4f} ({obj_pct:.1f}%)")
                for name, val in sorted(a.obj_contributions.items(), key=lambda x: -abs(x[1])):
                    if abs(val) > 1e-6:
                        lines.append(f"    {name}: {val:+.4f}")
                if abs(a.interaction_effect) > 1e-6:
                    lines.append(f"  Interaction: {a.interaction_effect:+.4f} ({int_pct:.1f}%)")
            if a.shapley_b is not None and a.shapley_c is not None:
                sb = a.shapley_b / a.delta_z * 100 if abs(a.delta_z) > 1e-10 else 0
                sc = a.shapley_c / a.delta_z * 100 if abs(a.delta_z) > 1e-10 else 0
                lines.append(f"  Shapley: \u0394b = {a.shapley_b:.4f} ({sb:.1f}%), "
                             f"\u0394c = {a.shapley_c:.4f} ({sc:.1f}%)")
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
        d = {
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
        if self.attribution is not None:
            a = self.attribution
            d["attribution"] = {
                "delta_z": a.delta_z,
                "rhs_effect": a.rhs_effect,
                "obj_effect": a.obj_effect,
                "interaction_effect": a.interaction_effect,
                "rhs_contributions": a.rhs_contributions,
                "obj_contributions": a.obj_contributions,
                "shapley_b": a.shapley_b,
                "shapley_c": a.shapley_c,
                "basis_preserved": a.basis_preserved,
            }
        return d

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)
