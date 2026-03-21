"""
CLARA SolveState — Detailed Design Document
=============================================

SolveState is the central data structure of CLARA.
It is an immutable snapshot of the solver's complete internal state after solving.

Design Principles:
    1. Immutable — frozen dataclass, never mutated after creation
    2. Backend-agnostic — both InternalSimplex and HiGHS fill the same structure
    3. Self-contained — carries everything needed for explanation and reoptimization
    4. Diffable — two SolveStates can be compared to generate a Diff Report
    5. Serializable — JSON export for CLI --format json

Architecture:
    SolveEngine.solve(problem) → SolveState
    Explainer.explain(state) → ExplanationReport
    Differ.diff(state_old, state_new) → DiffReport
    Analyzer.should_reoptimize(state, changes) → ReoptDecision
    Reoptimizer.reoptimize(state, changes) → SolveState (new)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
        - name, value, basis_status → Understandability ("x1 = 755.56, in basis")
        - reduced_cost → Justifiability ("x3 costs 0.33 more to produce — too expensive")
        - obj_coeff_range → Actionability ("profit of P3 can drop to 4.2 before solution changes")
    """
    name: str
    value: float
    basis_status: BasisStatus
    reduced_cost: float          # 0 if basic; for non-basic: how much obj must improve
    obj_coeff_range: tuple[float, float]  # (lower, upper) for current basis to stay optimal
    lower_bound: float = 0.0
    upper_bound: float = float('inf')


@dataclass(frozen=True)
class ConstraintInfo:
    """Per-constraint information in the solution.

    Explanation mapping:
        - name, slack, is_binding → Understandability ("S1 is binding: department 1 at full capacity")
        - dual_value → Justifiability ("adding 1 unit of S1 capacity → +0.78 profit")
        - rhs_range → Actionability ("S1 capacity can increase to 1350 before basis changes")
    """
    name: str
    rhs: float                   # right-hand side value
    slack: float                 # rhs - (Ax)_i; 0 if binding
    dual_value: float            # shadow price; 0 if non-binding
    is_binding: bool             # slack == 0 (within tolerance)
    basis_status: BasisStatus    # BASIC if non-binding, NONBASIC if binding
    rhs_range: tuple[float, float]  # (lower, upper) for current basis to stay optimal


@dataclass(frozen=True)
class SensitivityRanges:
    """Aggregated sensitivity analysis results.

    For each variable j:
        obj_coeff_ranges[j] = (c_j_lower, c_j_upper)
            Range of objective coefficient c_j for which current basis is optimal.

    For each constraint i:
        rhs_ranges[i] = (b_i_lower, b_i_upper)
            Range of RHS value b_i for which current basis is feasible.

    These ranges are the foundation of the Impact Analyzer (Phase 2):
        - If a parameter change falls within the range → no reoptimization needed
        - If it falls outside → reoptimization required
        - Oguz bound tightens: δ' = (δ - α) / (1 + α) where α = Wendell tolerance

    Both backends can compute these:
        InternalSimplex: from B^-1 directly
        HiGHS: via getRanging() API
    """
    obj_coeff_ranges: dict[str, tuple[float, float]]  # var_name → (lower, upper)
    rhs_ranges: dict[str, tuple[float, float]]         # constraint_name → (lower, upper)


@dataclass(frozen=True)
class IterationSnapshot:
    """One simplex iteration record (internal engine only).

    Captures a single pivot operation for step-by-step explanation:
        "Iteration 3: x2 enters basis (reduced cost -1.33), y3 leaves basis
         (ratio test: min at row 3). Objective: 2400 → 3200."

    The sequence of these snapshots IS the step-by-step explanation.
    """
    iteration: int
    entering_var: str            # variable entering the basis
    leaving_var: str             # variable leaving the basis
    pivot_row: int               # row index of pivot element
    pivot_col: int               # column index of pivot element
    pivot_element: float         # value of pivot element
    objective_value: float       # objective after this pivot
    basic_variables: tuple[str, ...]  # ordered list of basic variables after pivot
    entering_reduced_cost: float      # reduced cost that triggered entry
    leaving_ratio: float              # min ratio that determined exit

    # Optional: full state at this iteration (expensive, enable via flag)
    basic_values: Optional[tuple[float, ...]] = None    # B^-1 * b at this point
    reduced_costs: Optional[tuple[float, ...]] = None   # all reduced costs


@dataclass(frozen=True)
class BnBNodeSnapshot:
    """One Branch-and-Bound node record (Phase 1.5, internal engine only).

    Captures branching decisions for MIP explanation:
        "Node 7: branched on x2 <= 3 (LP relaxation = 4520.5, incumbent = 4300).
         Fathomed by bound at node 12."
    """
    node_id: int
    parent_id: Optional[int]
    depth: int
    branching_var: str
    branching_direction: str     # "<=", ">="
    branching_value: float
    lp_relaxation_value: float
    is_integer_feasible: bool
    is_fathomed: bool
    fathom_reason: Optional[str]  # "bound", "infeasible", "integer_feasible"
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
    basis_inverse: Optional[np.ndarray] = None       # B⁻¹ matrix (m × m)
    iteration_history: Optional[tuple[IterationSnapshot, ...]] = None
    bnb_history: Optional[tuple[BnBNodeSnapshot, ...]] = None  # Phase 1.5

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
        for compact output. Use to_dict_full() for complete serialization.
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


# ============================================================
# Diff structures (Phase 1.5 / Phase 2)
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
    TYPE_R = auto()     # b-vector change
    TYPE_C = auto()     # c-vector change
    TYPE_V = auto()     # column addition
    TYPE_X = auto()     # row addition
    TYPE_M = auto()     # row deletion
    TYPE_RC = auto()    # compound b + c change


@dataclass(frozen=True)
class ParameterChange:
    """Detected change between two problem instances.

    The Change Detector (Phase 2) produces this by comparing
    the old problem and new problem parameter by parameter.
    """
    change_type: ChangeType
    delta_b: Optional[np.ndarray] = None    # RHS change vector (Type R, RC)
    delta_c: Optional[np.ndarray] = None    # Obj coeff change vector (Type C, RC)
    new_columns: Optional[dict] = None      # {name: {"cost": float, "column": list}} (Type V)
    new_rows: Optional[dict] = None         # {name: {"coeffs": list, "rhs": float}} (Type X)
    removed_rows: Optional[list[str]] = None  # constraint names (Type M)

    @property
    def max_rhs_change_ratio(self) -> Optional[float]:
        """Max relative RHS change. Used for sensitivity range comparison."""
        if self.delta_b is None:
            return None
        # Avoid division by zero for zero RHS
        return float(np.max(np.abs(self.delta_b)))

    @property
    def max_obj_change_ratio(self) -> Optional[float]:
        """Max relative obj coeff change (δ in Oguz bound)."""
        if self.delta_c is None:
            return None
        return float(np.max(np.abs(self.delta_c)))


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
    reason: str                          # human-readable decision explanation
    oguz_bound: Optional[float] = None   # 2δ/(1+δ) if applicable
    within_sensitivity: bool = False     # all changes within ranges?
    estimated_pivots: Optional[int] = None  # rough estimate of pivots needed
    recommended_method: Optional[str] = None  # "warm_start", "scratch", "none"


@dataclass(frozen=True)
class SolutionDiff:
    """Comparison of two SolveStates — the "what changed and why" report.

    Maps to XAIOR Actionability: tells the decision-maker what happened
    to their solution and what they should do next.
    """
    state_old: SolveState
    state_new: SolveState
    change: ParameterChange

    # Objective change
    objective_delta: float = 0.0
    objective_delta_pct: float = 0.0

    # Structural changes
    basis_changed: bool = False
    variables_entered: tuple[str, ...] = ()    # newly basic variables
    variables_left: tuple[str, ...] = ()       # newly non-basic variables
    binding_added: tuple[str, ...] = ()        # newly binding constraints
    binding_removed: tuple[str, ...] = ()      # no longer binding constraints

    # Per-variable value changes (only for variables that changed significantly)
    variable_changes: tuple[tuple[str, float, float], ...] = ()  # (name, old, new)


# ============================================================
# Example: constructing a SolveState from Albici base problem
# ============================================================

def _example_albici_base() -> SolveState:
    """Illustrative construction for the Albici base problem.

    This shows how an engine would build a SolveState.
    Not executable without the actual solver — for documentation only.
    """
    return SolveState(
        status=SolveStatus.OPTIMAL,
        optimal_value=33200 / 9,
        engine=EngineType.INTERNAL_SIMPLEX,
        solve_time_seconds=0.001,
        iteration_count=3,
        problem_name="albici_base",
        variable_names=("x1", "x2", "x3"),
        constraint_names=("S1", "S2", "S3", "S4"),
        variables=(
            VariableInfo(
                name="x1", value=6800/9, basis_status=BasisStatus.BASIC,
                reduced_cost=0.0, obj_coeff_range=(1.5, 5.0),
            ),
            VariableInfo(
                name="x2", value=1200/9, basis_status=BasisStatus.BASIC,
                reduced_cost=0.0, obj_coeff_range=(2.4, 10.0),
            ),
            VariableInfo(
                name="x3", value=1600/9, basis_status=BasisStatus.BASIC,
                reduced_cost=0.0, obj_coeff_range=(3.0, 7.5),
            ),
        ),
        constraints=(
            ConstraintInfo(
                name="S1", rhs=1200, slack=0.0, dual_value=7/9,
                is_binding=True, basis_status=BasisStatus.NONBASIC_UPPER,
                rhs_range=(900.0, 1400.0),
            ),
            ConstraintInfo(
                name="S2", rhs=1400, slack=1000/9, dual_value=0.0,
                is_binding=False, basis_status=BasisStatus.BASIC,
                rhs_range=(1288.89, float('inf')),
            ),
            ConstraintInfo(
                name="S3", rhs=2000, slack=0.0, dual_value=10/9,
                is_binding=True, basis_status=BasisStatus.NONBASIC_UPPER,
                rhs_range=(1600.0, 2200.0),
            ),
            ConstraintInfo(
                name="S4", rhs=800, slack=0.0, dual_value=6/9,
                is_binding=True, basis_status=BasisStatus.NONBASIC_UPPER,
                rhs_range=(600.0, 1200.0),
            ),
        ),
        sensitivity=SensitivityRanges(
            obj_coeff_ranges={"x1": (1.5, 5.0), "x2": (2.4, 10.0), "x3": (3.0, 7.5)},
            rhs_ranges={"S1": (900.0, 1400.0), "S2": (1288.89, float('inf')),
                        "S3": (1600.0, 2200.0), "S4": (600.0, 1200.0)},
        ),
        basis_inverse=np.array([
            [11/9, 1, -10/9, -6/9],
            [1/9, 0, 4/9, -3/9],
            [6/9, 0, -3/9, 0],
            [-4/9, 0, 2/9, 3/9],
        ]),
        iteration_history=(
            IterationSnapshot(
                iteration=1, entering_var="x1", leaving_var="y3",
                pivot_row=2, pivot_col=0, pivot_element=2.0,
                objective_value=2000.0,
                basic_variables=("y1", "y2", "x1", "y4"),
                entering_reduced_cost=-3.0, leaving_ratio=1000.0,
            ),
            # ... subsequent iterations would follow
        ),
    )