"""Tests for MIP opportunity cost bound analyzer."""

import numpy as np
import pytest

from clara.model.problem import LPProblem
from clara.model.solve_state import (
    BasisStatus, ConstraintInfo, EngineType, SensitivityRanges,
    SolveState, SolveStatus, VariableInfo,
)
from clara.reopt.detector import ChangeDetector
from clara.reopt.mip_bound import MIPBoundAnalyzer
from clara.reopt.types import ChangeType, ParameterChange

TOL = 0.1


# ============================================================
# Helpers: mock SolveState from known values
# ============================================================

def _make_state(
    z: float, x: list[float], var_names: list[str],
    con_names: list[str], duals: list[float],
    basis_basic: list[bool],
    obj_ranges: dict[str, tuple[float, float]] = None,
    rhs_ranges: dict[str, tuple[float, float]] = None,
    b: list[float] = None,
    engine: EngineType = EngineType.INTERNAL_SIMPLEX,
) -> SolveState:
    """Create a mock SolveState with known values."""
    variables = tuple(
        VariableInfo(
            name=var_names[j], value=x[j],
            basis_status=BasisStatus.BASIC if basis_basic[j] else BasisStatus.NONBASIC_LOWER,
            reduced_cost=0.0, obj_coeff_range=(0.0, 100.0),
        )
        for j in range(len(x))
    )
    constraints = tuple(
        ConstraintInfo(
            name=con_names[i],
            rhs=b[i] if b else 0.0,
            slack=0.0, dual_value=duals[i],
            is_binding=True,
            basis_status=BasisStatus.NONBASIC_UPPER,
            rhs_range=(0.0, 100.0),
        )
        for i in range(len(duals))
    )
    sensitivity = SensitivityRanges(
        obj_coeff_ranges=obj_ranges or {},
        rhs_ranges=rhs_ranges or {},
    )
    return SolveState(
        status=SolveStatus.OPTIMAL,
        optimal_value=z,
        variables=variables,
        constraints=constraints,
        sensitivity=sensitivity,
        engine=engine,
        solve_time_seconds=0.0,
        iteration_count=0,
        problem_name="test",
        variable_names=tuple(var_names),
        constraint_names=tuple(con_names),
    )


# ============================================================
# Knapsack test data (from golden values in spec)
# ============================================================

def knapsack_problem():
    return LPProblem(
        c=[16, 22, 12, 8, 11],
        A=[[5, 7, 4, 3, 4]],
        b=[14],
        var_names=["x1", "x2", "x3", "x4", "x5"],
        constraint_names=["weight"],
        upper_bounds=np.array([1, 1, 1, 1, 1]),
        binary_vars={0, 1, 2, 3, 4},
    )


def knapsack_mip_state():
    """z*_MIP = 42, x* = [0, 1, 1, 1, 0]"""
    return _make_state(
        z=42.0, x=[0, 1, 1, 1, 0],
        var_names=["x1", "x2", "x3", "x4", "x5"],
        con_names=["weight"], duals=[3.0], b=[14],
        basis_basic=[False, True, True, True, False],
        engine=EngineType.INTERNAL_BNB,
    )


def knapsack_lp_state():
    """z*_LP = 44.0, x̄ = [1, 1, 0.5, 0, 0]"""
    return _make_state(
        z=44.0, x=[1, 1, 0.5, 0, 0],
        var_names=["x1", "x2", "x3", "x4", "x5"],
        con_names=["weight"], duals=[3.0], b=[14],
        basis_basic=[True, True, True, False, False],
        obj_ranges={
            "x1": (12.0, 28.0), "x2": (14.0, 44.0), "x3": (9.0, 16.0),
            "x4": (0.0, 12.0), "x5": (0.0, 16.0),
        },
        rhs_ranges={"weight": (7.0, 21.0)},
    )


@pytest.fixture
def analyzer():
    return MIPBoundAnalyzer(epsilon=0.05)


# ============================================================
# 1. Bound validity
# ============================================================

class TestBoundValidity:

    def test_bound_nonnegative(self, analyzer):
        """Bound should always be ≥ 0."""
        old = knapsack_problem()
        new = LPProblem(
            c=[18, 20, 13, 7, 11], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.bound >= -TOL

    def test_bound_small_change(self, analyzer):
        """Small perturbation → small bound."""
        old = knapsack_problem()
        new = LPProblem(
            c=[16.1, 22.1, 12.1, 8.1, 11.1], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.bound < 5.0  # small perturbation → small bound

    def test_bound_large_change(self, analyzer):
        """Large perturbation → larger bound."""
        old = knapsack_problem()
        new = LPProblem(
            c=[24, 14, 18, 4, 16], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.bound > 0


# ============================================================
# 2. Feasibility
# ============================================================

class TestFeasibility:

    def test_feasible_no_rhs_change(self, analyzer):
        """Δc only → always feasible."""
        old = knapsack_problem()
        new = LPProblem(
            c=[20, 20, 15, 10, 15], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.old_solution_feasible
        assert result.decision != "infeasible"

    def test_infeasible_rhs_decrease(self, analyzer):
        """Decrease RHS below Ax* → infeasible."""
        old = knapsack_problem()
        # x* = [0,1,1,1,0], Ax* = 7+4+3 = 14. b=12 < 14 → infeasible
        new = LPProblem(
            c=old.c, A=old.A, b=[12],
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert not result.old_solution_feasible
        assert result.decision == "infeasible"

    def test_feasible_rhs_increase(self, analyzer):
        """Increase RHS → still feasible."""
        old = knapsack_problem()
        new = LPProblem(
            c=old.c, A=old.A, b=[16],
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.old_solution_feasible


# ============================================================
# 3. Gap-corrected
# ============================================================

class TestGapCorrected:

    def test_corrected_le_raw(self, analyzer):
        """Gap-corrected bound ≤ raw bound."""
        old = knapsack_problem()
        new = LPProblem(
            c=[20, 20, 15, 10, 15], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        if result.bound_corrected is not None:
            assert result.bound_corrected <= result.bound + TOL

    def test_corrected_nonneg(self, analyzer):
        """Gap-corrected bound ≥ 0."""
        old = knapsack_problem()
        new = LPProblem(
            c=[16.5, 22.5, 12.5, 8.5, 11.5], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        if result.bound_corrected is not None:
            assert result.bound_corrected >= -TOL


# ============================================================
# 4. Decision
# ============================================================

class TestDecision:

    def test_skip_small_change(self):
        """Very small perturbation → skip."""
        analyzer = MIPBoundAnalyzer(epsilon=0.10)
        old = knapsack_problem()
        new = LPProblem(
            c=[16.01, 22.01, 12.01, 8.01, 11.01], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.decision == "skip"

    def test_reoptimize_large_change(self, analyzer):
        """Large perturbation → reoptimize."""
        old = knapsack_problem()
        new = LPProblem(
            c=[30, 10, 25, 5, 20], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.decision == "reoptimize"


# ============================================================
# 5. Oguz
# ============================================================

class TestOguz:

    def test_oguz_computable(self, analyzer):
        """All c_j > 0 → Oguz bound computable."""
        old = knapsack_problem()
        new = LPProblem(
            c=[20, 20, 15, 10, 15], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.oguz_relative_bound is not None
        assert result.delta is not None

    def test_oguz_none_no_c_change(self, analyzer):
        """No Δc → Oguz is None."""
        old = knapsack_problem()
        new = LPProblem(
            c=old.c, A=old.A, b=[16],
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert result.oguz_relative_bound is None


# ============================================================
# 6. Summary + integrality gap
# ============================================================

class TestSummary:

    def test_summary_contains_bound(self, analyzer):
        old = knapsack_problem()
        new = LPProblem(
            c=[20, 20, 15, 10, 15], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert "Bound" in result.summary or "bound" in result.summary.lower()

    def test_integrality_gap(self, analyzer):
        """Gap = z_LP - z_MIP = 44 - 42 = 2."""
        old = knapsack_problem()
        new = LPProblem(
            c=[20, 20, 15, 10, 15], A=old.A, b=old.b,
            var_names=old.var_names, constraint_names=old.constraint_names,
            upper_bounds=old.upper_bounds, binary_vars=old.binary_vars,
        )
        change = ChangeDetector().detect(old, new)
        result = analyzer.analyze(
            knapsack_mip_state(), knapsack_lp_state(), old, new, change
        )
        assert abs(result.original_integrality_gap - 2.0) < TOL
        assert abs(result.gap_ratio - 2.0 / 44.0) < 0.01
