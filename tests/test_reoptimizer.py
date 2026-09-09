"""Tests for Warm-start Reoptimizer — Albici scenarios, full pipeline."""

import numpy as np
import pytest

from clara.engine import HiGHSBackend
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
    return HiGHSBackend().solve(albici_base())


def full_pipeline(old_problem, new_problem, base_state):
    """Run detect → analyze → reoptimize."""
    change = ChangeDetector().detect(old_problem, new_problem)
    decision = ImpactAnalyzer().analyze(base_state, change, old_problem)
    result = Reoptimizer().reoptimize(base_state, new_problem, change, decision, old_problem=old_problem)
    return result, decision


# ============================================================
# 1. Type C skip → certified recompute
# ============================================================

class TestTypeCSkip:

    def test_type_c_skip_is_certified_recompute(self, base_state):
        """Small cost change within range → recompute path, same basis, 0 pivots."""
        new = LPProblem(c=[3.01, 4.01, 5.01], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        result, decision = full_pipeline(albici_base(), new, base_state)
        assert not decision.should_reoptimize
        assert decision.recommended_method == "recompute"
        assert result.method_used == "recompute"
        assert result.pivots == 0
        assert result.basis_preserved

    def test_type_c_skip_falls_back_when_joint_change_breaks_basis(self):
        """Every objective change lies inside its one-at-a-time range, so the
        analyzer skips, but the simultaneous change makes the retained basis
        dual infeasible: the certified recompute must fall back to a warm
        start and return the true optimum (stale solution would lose)."""
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "benchmarks" / "scripts"))
        from generate_perturbations import generate_perturbation  # noqa: E402
        from clara.io.lp_parser import read_lp
        old = read_lp(root / "benchmarks" / "instances" / "random" / "rand_n10_m10_d5_s456.lp")
        old_state = HiGHSBackend().solve(old)
        new_c, _, _, _ = generate_perturbation(old, "C_medium", "C", 0, 0.20, 456)
        new = LPProblem(c=new_c, A=old.A, b=old.b, sense=old.sense, var_names=old.var_names,
                        constraint_names=old.constraint_names, upper_bounds=old.upper_bounds)
        change = ChangeDetector().detect(old, new)
        decision = ImpactAnalyzer().analyze(old_state, change, old)
        assert not decision.should_reoptimize and decision.within_sensitivity
        result = Reoptimizer().reoptimize(old_state, new, change, decision, old_problem=old)
        z_true = HiGHSBackend().solve(new).optimal_value
        x_old = np.array([v.value for v in old_state.variables])
        stale_loss = abs(float(new.c @ x_old) - z_true) / abs(z_true)
        assert stale_loss > 1e-3, "instance no longer exercises the fallback"
        assert result.method_used == "warm_start"
        assert not result.basis_preserved
        assert abs(result.new_state.optimal_value - z_true) < 1e-8 * max(1.0, abs(z_true))

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
        """b2: basis broken (primal infeasible, dual feasible) → dual simplex → optimal = 4300."""
        result, _ = full_pipeline(albici_base(), albici_b2(), base_state)
        assert result.method_used == "warm_start_dual"
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
            scratch = HiGHSBackend().solve(new_problem)
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
        fresh = HiGHSBackend().solve(albici_columns())
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
# 6. Advanced-basis warm start correctness
# ============================================================

class TestAdvancedBasisWarmStart:

    def test_warm_start_from_optimal_basis(self):
        p = albici_base()
        cold = HiGHSBackend().solve(p)
        assert cold.basis_indices is not None
        # Warm-start from the optimal basis must reproduce the optimum
        warm = HiGHSBackend().solve(p, initial_basis=cold.basis_indices)
        assert abs(warm.optimal_value - cold.optimal_value) < TOL

    def test_warm_start_from_slack_basis(self):
        p = albici_base()
        cold = HiGHSBackend().solve(p)
        n = p.num_variables
        m = p.num_constraints
        # All-slack basis is valid for b >= 0
        warm = HiGHSBackend().solve(p, initial_basis=list(range(n, n + m)))
        assert abs(warm.optimal_value - cold.optimal_value) < TOL


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
        state = HiGHSBackend().solve(albici_base())
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
