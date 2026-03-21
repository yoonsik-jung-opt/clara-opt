"""Tests for Warm-start Reoptimizer — Albici scenarios, full pipeline."""

import numpy as np
import pytest

from clara.engine.simplex import RevisedSimplex
from clara.model.problem import LPProblem
from clara.reopt.analyzer import ImpactAnalyzer
from clara.reopt.detector import ChangeDetector
from clara.reopt.reoptimizer import Reoptimizer
from clara.reopt.types import ChangeType, ReoptResult

TOL = 1e-4


# ============================================================
# Helpers
# ============================================================

def albici_base() -> LPProblem:
    return LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )

def albici_b1() -> LPProblem:
    p = albici_base()
    return LPProblem(c=p.c, A=p.A, b=[1300, 1200, 1800, 1000],
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))

def albici_b2() -> LPProblem:
    p = albici_base()
    return LPProblem(c=p.c, A=p.A, b=[1500, 1300, 2400, 800],
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))

def albici_cost() -> LPProblem:
    p = albici_base()
    return LPProblem(c=[8, 6, 7], A=p.A, b=p.b,
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))

def albici_columns() -> LPProblem:
    return LPProblem(
        c=[3, 4, 5, 3, 5],
        A=[[1, 2, 1, 2, 3], [1, 0, 3, 0, 1], [2, 1, 2, 1, 0], [0, 2, 3, 0, 1]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3", "x4", "x5"],
        constraint_names=["S1", "S2", "S3", "S4"],
    )


@pytest.fixture
def base_state():
    return RevisedSimplex(albici_base()).solve()


def full_pipeline(old_problem, new_problem, base_state):
    """Run detect → analyze → reoptimize."""
    change = ChangeDetector().detect(old_problem, new_problem)
    decision = ImpactAnalyzer().analyze(base_state, change, old_problem)
    result = Reoptimizer().reoptimize(base_state, new_problem, change, decision)
    return result, decision


# ============================================================
# 1. Method "none"
# ============================================================

class TestMethodNone:

    def test_none_preserves_solution(self, base_state):
        """Small cost change within range → same solution, different obj."""
        new = LPProblem(c=[3.01, 4.01, 5.01], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        result, _ = full_pipeline(albici_base(), new, base_state)
        assert result.method_used == "none"
        assert result.pivots == 0
        assert result.basis_preserved

    def test_none_correct_objective(self, base_state):
        """New objective = c_new^T @ x_old."""
        new = LPProblem(c=[3.01, 4.01, 5.01], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        result, _ = full_pipeline(albici_base(), new, base_state)
        x = np.array([v.value for v in base_state.variables])
        expected_obj = float(new.c @ x)
        assert abs(result.new_state.optimal_value - expected_obj) < TOL


# ============================================================
# 2. Method "recompute" (Type R within range)
# ============================================================

class TestMethodRecompute:

    def test_recompute_single_rhs_change(self, base_state):
        """Single RHS change within range → recompute, 0 pivots."""
        new = LPProblem(c=albici_base().c, A=albici_base().A, b=[1300, 1400, 2000, 800],
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        result, _ = full_pipeline(albici_base(), new, base_state)
        assert result.method_used == "recompute"
        assert result.pivots == 0
        assert result.basis_preserved


# ============================================================
# 3. Method "warm_start"
# ============================================================

class TestMethodWarmStart:

    def test_warmstart_albici_b2(self, base_state):
        """b2: basis broken (primal infeasible) → falls back to scratch → optimal = 4300."""
        result, _ = full_pipeline(albici_base(), albici_b2(), base_state)
        # Falls back to scratch because old basis is primal infeasible with new RHS
        assert result.method_used == "scratch"
        assert abs(result.new_state.optimal_value - 4300.0) < TOL

    def test_warmstart_type_c(self, base_state):
        """Cost change → warm-start → optimal = 24800/3."""
        result, _ = full_pipeline(albici_base(), albici_cost(), base_state)
        assert result.method_used == "warm_start"
        assert abs(result.new_state.optimal_value - 24800 / 3) < TOL

    def test_warmstart_matches_scratch(self, base_state):
        """Warm-start optimal should match scratch solve."""
        for new_problem in [albici_b2(), albici_cost()]:
            result, _ = full_pipeline(albici_base(), new_problem, base_state)
            scratch = RevisedSimplex(new_problem).solve()
            assert abs(result.new_state.optimal_value - scratch.optimal_value) < TOL, (
                f"Warm-start={result.new_state.optimal_value} vs scratch={scratch.optimal_value}"
            )


# ============================================================
# 4. Method "scratch"
# ============================================================

class TestMethodScratch:

    def test_scratch_type_v(self, base_state):
        """Column addition → scratch → optimal = 33700/9."""
        result, _ = full_pipeline(albici_base(), albici_columns(), base_state)
        assert result.method_used == "scratch"
        assert abs(result.new_state.optimal_value - 33700 / 9) < TOL

    def test_scratch_matches_fresh(self, base_state):
        """Scratch result should match independent solve."""
        result, _ = full_pipeline(albici_base(), albici_columns(), base_state)
        fresh = RevisedSimplex(albici_columns()).solve()
        assert abs(result.new_state.optimal_value - fresh.optimal_value) < TOL


# ============================================================
# 5. Full pipeline integration
# ============================================================

class TestFullPipeline:

    def test_pipeline_b2_reopt(self, base_state):
        """b2: primal infeasible warm-start → falls back to scratch."""
        result, decision = full_pipeline(albici_base(), albici_b2(), base_state)
        assert decision.should_reoptimize
        assert abs(result.new_state.optimal_value - 4300.0) < TOL

    def test_pipeline_cost_warmstart(self, base_state):
        result, decision = full_pipeline(albici_base(), albici_cost(), base_state)
        assert decision.should_reoptimize
        assert result.method_used == "warm_start"

    def test_pipeline_columns_scratch(self, base_state):
        result, decision = full_pipeline(albici_base(), albici_columns(), base_state)
        assert decision.should_reoptimize
        assert decision.recommended_method == "scratch"
        assert result.method_used == "scratch"


# ============================================================
# 6. from_warm_start correctness
# ============================================================

class TestFromWarmStart:

    def test_classmethod_works(self):
        p = albici_base()
        cold = RevisedSimplex(p).solve()
        # Warm-start from the optimal basis should converge in 0 iterations
        warm = RevisedSimplex.from_warm_start(
            p, list(range(3, 7)),  # initial slack basis (will be overridden)
            np.eye(4),
        )
        state = warm.solve()
        assert abs(state.optimal_value - cold.optimal_value) < TOL


# ============================================================
# 7. ReoptResult properties
# ============================================================

class TestReoptResultProperties:

    def test_summary_basis_preserved(self, base_state):
        new = LPProblem(c=[3.01, 4.01, 5.01], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        result, _ = full_pipeline(albici_base(), new, base_state)
        assert "Basis preserved" in result.summary

    def test_summary_warm_start(self, base_state):
        """Cost change uses warm-start (primal feasible, dual infeasible)."""
        result, _ = full_pipeline(albici_base(), albici_cost(), base_state)
        assert "warm_start" in result.summary

    def test_speedup_computed(self, base_state):
        result, _ = full_pipeline(albici_base(), albici_b2(), base_state)
        if result.scratch_estimate and result.scratch_estimate > 0:
            assert result.speedup is not None


# ============================================================
# 8. Edge cases
# ============================================================

class TestEdgeCases:

    def test_no_binv_falls_back_to_scratch(self):
        """SolveState without B⁻¹ → scratch for any method."""
        from clara.model.solve_state import SolveState
        from clara.reopt.types import ReoptDecision
        state = RevisedSimplex(albici_base()).solve()
        # Remove B⁻¹
        state_no_binv = SolveState(
            status=state.status, optimal_value=state.optimal_value,
            variables=state.variables, constraints=state.constraints,
            sensitivity=state.sensitivity, engine=state.engine,
            solve_time_seconds=state.solve_time_seconds,
            iteration_count=state.iteration_count,
            basis_inverse=None,
            problem_name=state.problem_name,
            variable_names=state.variable_names,
            constraint_names=state.constraint_names,
        )
        change = ChangeDetector().detect(albici_base(), albici_b2())
        decision = ReoptDecision(
            should_reoptimize=True, reason="test",
            recommended_method="warm_start",
        )
        result = Reoptimizer().reoptimize(state_no_binv, albici_b2(), change, decision)
        assert result.method_used == "scratch"
