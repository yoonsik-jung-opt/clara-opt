"""Tests for Parametric LP — Albici compound golden tests."""

import numpy as np
import pytest

from clara.engine.simplex import RevisedSimplex
from clara.model.problem import LPProblem
from clara.reopt.analyzer import ImpactAnalyzer
from clara.reopt.detector import ChangeDetector
from clara.reopt.parametric import ParametricLPSolver
from clara.reopt.reoptimizer import Reoptimizer

TOL = 0.1  # parametric LP path tolerance
TOL_OPT = 1e-4


def albici_base():
    return LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )


def albici_compound():
    """b2 + cost change simultaneously."""
    return LPProblem(
        c=[8, 6, 7], A=albici_base().A, b=[1500, 1300, 2400, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )


DELTA_B = np.array([300, -100, 400, 0], dtype=float)
DELTA_C = np.array([5, 2, 2], dtype=float)


@pytest.fixture
def base_state():
    return RevisedSimplex(albici_base()).solve()


@pytest.fixture
def param_result(base_state):
    return ParametricLPSolver().solve(base_state, albici_base(), DELTA_B, DELTA_C)


# ============================================================
# 1. Breakpoint detection
# ============================================================

class TestBreakpoints:

    def test_num_breakpoints(self, param_result):
        assert param_result.num_breakpoints == 2

    def test_first_breakpoint_primal(self, param_result):
        bp0 = param_result.breakpoints[0]
        assert bp0.breakpoint_type == "primal"
        assert abs(bp0.theta - 5 / 8) < 0.01

    def test_second_breakpoint_dual(self, param_result):
        bp1 = param_result.breakpoints[1]
        assert bp1.breakpoint_type == "dual"
        assert abs(bp1.theta - 2 / 3) < 0.01

    def test_breakpoint_order(self, param_result):
        thetas = [bp.theta for bp in param_result.breakpoints]
        assert thetas == sorted(thetas)
        assert thetas[0] < thetas[1]


# ============================================================
# 2. Full parametric solve
# ============================================================

class TestFullSolve:

    def test_final_optimal(self, param_result):
        assert abs(param_result.new_state.optimal_value - 10000.0) < TOL_OPT

    def test_final_x_values(self, param_result):
        vals = param_result.new_state.variable_values_dict
        assert abs(vals["x1"] - 1100.0) < TOL
        assert abs(vals["x2"] - 200.0) < TOL
        assert abs(vals["x3"] - 0.0) < TOL

    def test_x3_zero_after_breakpoint(self, param_result):
        """x3 should be 0 in final solution (left basis at θ≈0.667)."""
        assert param_result.new_state.variable_values_dict["x3"] < TOL

    def test_matches_scratch(self, param_result):
        scratch = RevisedSimplex(albici_compound()).solve()
        assert abs(param_result.new_state.optimal_value - scratch.optimal_value) < TOL_OPT


# ============================================================
# 3. No breakpoints case
# ============================================================

class TestNoBreakpoints:

    def test_small_change_no_breakpoints(self, base_state):
        """Tiny Δb and Δc → 0 breakpoints."""
        small_db = np.array([1, -1, 1, -1], dtype=float)
        small_dc = np.array([0.001, 0.001, 0.001], dtype=float)
        result = ParametricLPSolver().solve(base_state, albici_base(), small_db, small_dc)
        assert result.num_breakpoints == 0
        assert result.num_pivots == 0

    def test_no_breakpoints_basis_preserved(self, base_state):
        small_db = np.array([1, -1, 1, -1], dtype=float)
        small_dc = np.array([0.001, 0.001, 0.001], dtype=float)
        result = ParametricLPSolver().solve(base_state, albici_base(), small_db, small_dc)
        # No pivots → basis preserved
        assert result.num_pivots == 0


# ============================================================
# 4. Explanations
# ============================================================

class TestExplanations:

    def test_breakpoint_explanation_primal(self, param_result):
        bp0 = param_result.breakpoints[0]
        assert "leaves" in bp0.explanation.lower() or "hits zero" in bp0.explanation.lower()

    def test_breakpoint_explanation_dual(self, param_result):
        bp1 = param_result.breakpoints[1]
        assert "enters" in bp1.explanation.lower() or "reduced cost" in bp1.explanation.lower()

    def test_result_summary(self, param_result):
        assert "2 breakpoint" in param_result.summary


# ============================================================
# 5. Reoptimizer integration
# ============================================================

class TestReoptIntegration:

    def test_type_rc_uses_parametric(self, base_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        decision = ImpactAnalyzer().analyze(base_state, change, albici_base())
        result = Reoptimizer().reoptimize(
            base_state, albici_compound(), change, decision, old_problem=albici_base()
        )
        assert result.method_used == "parametric_lp"
        assert abs(result.new_state.optimal_value - 10000.0) < TOL_OPT

    def test_parametric_matches_scratch(self, base_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        decision = ImpactAnalyzer().analyze(base_state, change, albici_base())
        result = Reoptimizer().reoptimize(
            base_state, albici_compound(), change, decision, old_problem=albici_base()
        )
        scratch = RevisedSimplex(albici_compound()).solve()
        assert abs(result.new_state.optimal_value - scratch.optimal_value) < TOL_OPT


# ============================================================
# 6. Edge cases
# ============================================================

class TestEdgeCases:

    def test_no_binv_fallback(self):
        """No B⁻¹ → scratch fallback."""
        from clara.model.solve_state import SolveState, SolveStatus, EngineType, SensitivityRanges
        from clara.reopt.types import ReoptDecision
        state = SolveState(
            status=SolveStatus.OPTIMAL, optimal_value=100,
            variables=(), constraints=(),
            sensitivity=SensitivityRanges({}, {}),
            engine=EngineType.HIGHS, solve_time_seconds=0, iteration_count=0,
            basis_inverse=None,
            problem_name="test", variable_names=(), constraint_names=(),
        )
        change = ChangeDetector().detect(albici_base(), albici_compound())
        decision = ReoptDecision(
            should_reoptimize=True, reason="test", recommended_method="parametric_lp"
        )
        result = Reoptimizer().reoptimize(state, albici_compound(), change, decision)
        assert result.method_used == "scratch"
