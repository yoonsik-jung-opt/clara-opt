"""SolveState — Immutable snapshot of solver state after solving.

This is the central data structure of CLARA. Both InternalSimplex and HiGHS
backends fill the same structure, enabling backend-agnostic explanation and
reoptimization.

Architecture:
    SolveEngine.solve(problem) → SolveState
    Explainer.explain(state) → ExplanationReport
    Differ.diff(state_old, state_new) → DiffReport
    Analyzer.should_reoptimize(state, changes) → ReoptDecision
    Reoptimizer.reoptimize(state, changes) → SolveState (new)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np


# ============================================================
# Enums
# ============================================================

class SolveStatus(Enum):
    """Outcome of the solve attempt."""
    OPTIMAL = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    ITERATION_LIMIT = auto()
    NUMERICAL_ERROR = auto()


class BasisStatus(Enum):
    """Status of a variable or constraint in the basis.

    For decision variables (columns):
        BASIC          — in basis, value determined by constraints
        NONBASIC_LOWER — at lower bound (typically 0)
        NONBASIC_UPPER — at upper bound
        NONBASIC_FIXED — fixed (lower == upper)
        SUPERBASIC     — between bounds but not in basis (rare, MIP/QP)

    For constraints (rows):
        BASIC          — slack/surplus is basic (constraint is non-binding)
        NONBASIC_LOWER — constraint is binding at its lower bound (≥)
        NONBASIC_UPPER — constraint is binding at its upper bound (≤)
    """
    BASIC = auto()
    NONBASIC_LOWER = auto()
    NONBASIC_UPPER = auto()
    NONBASIC_FIXED = auto()
    SUPERBASIC = auto()


class EngineType(Enum):
    """Which solve engine produced this state."""
    INTERNAL_SIMPLEX = auto()
    INTERNAL_BNB = auto()       # Phase 1.5
    HIGHS = auto()


# ============================================================
# Sub-structures
# ============================================================

@dataclass(frozen=True)
class VariableInfo:
    """Per-variable information in the solution.

    Explanation mapping:
        - name, value, basis_status → Understandability
        - reduced_cost → Justifiability
        - obj_coeff_range → Actionability
    """
    name: str
    value: float
    basis_status: BasisStatus
    reduced_cost: float
    obj_coeff_range: tuple[float, float]
    lower_bound: float = 0.0
    upper_bound: float = float('inf')


@dataclass(frozen=True)
class ConstraintInfo:
    """Per-constraint information in the solution.

    Explanation mapping:
        - name, slack, is_binding → Understandability
        - dual_value → Justifiability
        - rhs_range → Actionability
    """
    name: str
    rhs: float
    slack: float
    dual_value: float
    is_binding: bool
    basis_status: BasisStatus
    rhs_range: tuple[float, float]


@dataclass(frozen=True)
class SensitivityRanges:
    """Aggregated sensitivity analysis results.

    For each variable j:
        obj_coeff_ranges[j] = (c_j_lower, c_j_upper)

    For each constraint i:
        rhs_ranges[i] = (b_i_lower, b_i_upper)

    Both backends can compute these:
        InternalSimplex: from B^-1 directly
        HiGHS: via getRanging() API
    """
    obj_coeff_ranges: dict[str, tuple[float, float]]
    rhs_ranges: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class IterationSnapshot:
    """One simplex iteration record (internal engine only).

    Captures a single pivot operation for step-by-step explanation.
    """
    iteration: int
    entering_var: str
    leaving_var: str
    pivot_row: int
    pivot_col: int
    pivot_element: float
    objective_value: float
    basic_variables: tuple[str, ...]
    entering_reduced_cost: float
    leaving_ratio: float
    basic_values: Optional[tuple[float, ...]] = None
    reduced_costs: Optional[tuple[float, ...]] = None


@dataclass(frozen=True)
class BnBNodeSnapshot:
    """One Branch-and-Bound node record (Phase 1.5, internal engine only)."""
    node_id: int
    parent_id: Optional[int]
    depth: int
    branching_var: str
    branching_direction: str
    branching_value: float
    lp_relaxation_value: float
    is_integer_feasible: bool
    is_fathomed: bool
    fathom_reason: Optional[str]
    incumbent_value: Optional[float]


# ============================================================
# Main SolveState
# ============================================================

@dataclass(frozen=True)
class SolveState:
    """Immutable snapshot of solver state after solving.

    This is the single source of truth for:
        1. Explainer (Phase 1)    — reads variables, constraints, iterations
        2. Differ (Phase 1.5)     — compares two SolveStates field by field
        3. Impact Analyzer (Ph 2) — reads sensitivity_ranges + basis_inverse
        4. Reoptimizer (Phase 2)  — uses basis_inverse for warm-start

    Field availability by backend:
        ┌────────────────────────┬──────────────┬──────────────┐
        │ Field                  │ Internal     │ HiGHS        │
        ├────────────────────────┼──────────────┼──────────────┤
        │ status                 │ ✓            │ ✓            │
        │ optimal_value          │ ✓            │ ✓            │
        │ variables              │ ✓            │ ✓            │
        │ constraints            │ ✓            │ ✓            │
        │ sensitivity            │ ✓ (from B⁻¹) │ ✓ (ranging)  │
        │ basis_inverse          │ ✓            │ ✗            │
        │ iteration_history      │ ✓            │ ✗            │
        │ bnb_history            │ ✓ (Ph 1.5)  │ ✗            │
        │ solve_stats            │ ✓            │ ✓            │
        └────────────────────────┴──────────────┴──────────────┘
    """

    # --- Core results (both backends) ---
    status: SolveStatus
    optimal_value: float
    variables: tuple[VariableInfo, ...]
    constraints: tuple[ConstraintInfo, ...]
    sensitivity: SensitivityRanges

    # --- Engine metadata ---
    engine: EngineType
    solve_time_seconds: float
    iteration_count: int

    # --- Internal engine only (None if HiGHS backend) ---
    basis_inverse: Optional[np.ndarray] = None
    basis_indices: Optional[tuple[int, ...]] = None  # column indices of basis in [A|I]
    iteration_history: Optional[tuple[IterationSnapshot, ...]] = None
    bnb_history: Optional[tuple[BnBNodeSnapshot, ...]] = None

    # --- Numerical diagnostics ---
    condition_number: Optional[float] = None   # κ(B), flagged if > 1e8
    degenerate_count: int = 0                  # basic vars with value < 1e-8
    basis_robustness_d0: Optional[float] = None  # min_i x_B[i] / ||B⁻¹_i||₂

    # --- Problem reference (for context in explanations) ---
    problem_name: str = ""
    variable_names: tuple[str, ...] = ()
    constraint_names: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_optimal(self) -> bool:
        return self.status == SolveStatus.OPTIMAL

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    @property
    def binding_constraints(self) -> tuple[ConstraintInfo, ...]:
        """Constraints where slack = 0 — these drive the explanation."""
        return tuple(c for c in self.constraints if c.is_binding)

    @property
    def basic_variables(self) -> tuple[VariableInfo, ...]:
        """Variables currently in the basis."""
        return tuple(v for v in self.variables if v.basis_status == BasisStatus.BASIC)

    @property
    def nonbasic_variables(self) -> tuple[VariableInfo, ...]:
        """Variables not in the basis (at bounds)."""
        return tuple(v for v in self.variables if v.basis_status != BasisStatus.BASIC)

    @property
    def has_internal_trace(self) -> bool:
        """Whether step-by-step iteration data is available."""
        return self.iteration_history is not None

    @property
    def has_basis_inverse(self) -> bool:
        """Whether B⁻¹ is available (needed for Phase 2 warm-start)."""
        return self.basis_inverse is not None

    @property
    def variable_values_dict(self) -> dict[str, float]:
        """Quick lookup: variable name → value."""
        return {v.name: v.value for v in self.variables}

    @property
    def dual_values_dict(self) -> dict[str, float]:
        """Quick lookup: constraint name → shadow price."""
        return {c.name: c.dual_value for c in self.constraints}

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def get_variable(self, name: str) -> VariableInfo:
        """Get variable info by name. Raises KeyError if not found."""
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(f"Variable '{name}' not found")

    def get_constraint(self, name: str) -> ConstraintInfo:
        """Get constraint info by name. Raises KeyError if not found."""
        for c in self.constraints:
            if c.name == name:
                return c
        raise KeyError(f"Constraint '{name}' not found")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (for CLI --format json).

        Note: basis_inverse and iteration_history are excluded by default
        for compact output.
        """
        return {
            "status": self.status.name,
            "optimal_value": self.optimal_value,
            "engine": self.engine.name,
            "solve_time_seconds": self.solve_time_seconds,
            "iteration_count": self.iteration_count,
            "problem_name": self.problem_name,
            "variables": [
                {
                    "name": v.name,
                    "value": v.value,
                    "basis_status": v.basis_status.name,
                    "reduced_cost": v.reduced_cost,
                    "obj_coeff_range": list(v.obj_coeff_range),
                }
                for v in self.variables
            ],
            "constraints": [
                {
                    "name": c.name,
                    "rhs": c.rhs,
                    "slack": c.slack,
                    "dual_value": c.dual_value,
                    "is_binding": c.is_binding,
                    "rhs_range": list(c.rhs_range),
                }
                for c in self.constraints
            ],
            "sensitivity": {
                "obj_coeff_ranges": {
                    k: list(v) for k, v in self.sensitivity.obj_coeff_ranges.items()
                },
                "rhs_ranges": {
                    k: list(v) for k, v in self.sensitivity.rhs_ranges.items()
                },
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent)
