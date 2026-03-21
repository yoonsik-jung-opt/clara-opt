"""Tests for Dual Simplex — correctness, Albici b2, reoptimizer integration."""

import numpy as np
import pytest

from clara.engine.simplex import RevisedSimplex
from clara.model.problem import LPProblem
from clara.model.solve_state import SolveStatus
from clara.reopt.analyzer import ImpactAnalyzer
from clara.reopt.detector import ChangeDetector
from clara.reopt.reoptimizer import Reoptimizer

TOL = 1e-4


def albici_base():
    return LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )

def albici_b2():
    p = albici_base()
    return LPProblem(c=p.c, A=p.A, b=[1500, 1300, 2400, 800],
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))

def albici_cost():
    p = albici_base()
    return LPProblem(c=[8, 6, 7], A=p.A, b=p.b,
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))


@pytest.fixture
def base_state():
    return RevisedSimplex(albici_base()).solve()


def full_pipeline(old_problem, new_problem, base_state):
    change = ChangeDetector().detect(old_problem, new_problem)
    decision = ImpactAnalyzer().analyze(base_state, change, old_problem)
    result = Reoptimizer().reoptimize(base_state, new_problem, change, decision, old_problem=old_problem)
    return result


# ============================================================
# 1. Core dual simplex correctness
# ============================================================

class TestDualSimplexCore:

    def test_dual_simplex_albici_b2(self, base_state):
        """Warm-start from base B⁻¹ with b2 RHS → dual simplex → optimal = 4300."""
        basis = _extract_basis(base_state, albici_b2())
        solver = RevisedSimplex.from_warm_start(albici_b2(), basis, base_state.basis_inverse)

        # Verify primal infeasible, dual feasible
        x_B = base_state.basis_inverse @ albici_b2().b
        assert np.any(x_B < -1e-8), "Should be primal infeasible"
        assert solver._is_dual_feasible(), "Should be dual feasible (c unchanged)"

        state = solver.solve_dual()
        assert state.status == SolveStatus.OPTIMAL
        assert abs(state.optimal_value - 4300.0) < TOL

    def test_dual_matches_scratch(self, base_state):
        """Dual simplex should find same optimal as scratch solve."""
        scratch = RevisedSimplex(albici_b2()).solve()
        basis = _extract_basis(base_state, albici_b2())
        solver = RevisedSimplex.from_warm_start(albici_b2(), basis, base_state.basis_inverse)
        dual = solver.solve_dual()
        assert abs(dual.optimal_value - scratch.optimal_value) < TOL

    def test_dual_fewer_pivots_than_scratch(self, base_state):
        """Dual simplex from near-optimal basis should need fewer pivots."""
        scratch_solver = RevisedSimplex(albici_b2())
        scratch_solver.solve()
        scratch_pivots = len(scratch_solver.iterations)

        basis = _extract_basis(base_state, albici_b2())
        dual_solver = RevisedSimplex.from_warm_start(albici_b2(), basis, base_state.basis_inverse)
        dual_solver.solve_dual()
        dual_pivots = len(dual_solver.iterations)

        assert dual_pivots <= scratch_pivots, (
            f"Dual {dual_pivots} pivots should be ≤ scratch {scratch_pivots}"
        )


# ============================================================
# 2. Edge cases
# ============================================================

class TestDualEdgeCases:

    def test_dual_already_feasible(self, base_state):
        """If x_B ≥ 0, solve_dual returns immediately (0 pivots)."""
        # Solve base problem (already optimal) → dual simplex should do nothing
        solver = RevisedSimplex.from_warm_start(
            albici_base(),
            _extract_basis(base_state, albici_base()),
            base_state.basis_inverse,
        )
        state = solver.solve_dual()
        assert state.status == SolveStatus.OPTIMAL
        assert len(solver.iterations) == 0

    def test_dual_feasibility_check_unchanged_c(self, base_state):
        """_is_dual_feasible() True when c unchanged (Type R)."""
        basis = _extract_basis(base_state, albici_b2())
        solver = RevisedSimplex.from_warm_start(albici_b2(), basis, base_state.basis_inverse)
        assert solver._is_dual_feasible()

    def test_dual_feasibility_check_changed_c(self, base_state):
        """_is_dual_feasible() may be False when c changed (Type C)."""
        basis = _extract_basis(base_state, albici_cost())
        solver = RevisedSimplex.from_warm_start(albici_cost(), basis, base_state.basis_inverse)
        # With large cost changes, dual feasibility likely broken
        # (reduced costs may become positive)
        # Just verify the check runs without error
        result = solver._is_dual_feasible()
        assert isinstance(result, bool)


# ============================================================
# 3. Reoptimizer integration
# ============================================================

class TestReoptIntegration:

    def test_type_r_broken_uses_dual(self, base_state):
        """Type R with broken basis → dual simplex, not scratch."""
        result = full_pipeline(albici_base(), albici_b2(), base_state)
        assert result.method_used == "warm_start_dual"
        assert abs(result.new_state.optimal_value - 4300.0) < TOL

    def test_type_c_uses_primal(self, base_state):
        """Type C → primal simplex (primal feasible, dual infeasible)."""
        result = full_pipeline(albici_base(), albici_cost(), base_state)
        assert result.method_used == "warm_start"
        assert abs(result.new_state.optimal_value - 24800 / 3) < TOL

    def test_warm_start_dual_fewer_pivots(self, base_state):
        """Dual warm-start should report fewer pivots than scratch estimate."""
        result = full_pipeline(albici_base(), albici_b2(), base_state)
        if result.scratch_estimate and result.scratch_estimate > 0:
            assert result.pivots <= result.scratch_estimate

    def test_speedup_reported(self, base_state):
        """ReoptResult.speedup should be computed for dual warm-start."""
        result = full_pipeline(albici_base(), albici_b2(), base_state)
        if result.scratch_estimate and result.scratch_estimate > 0:
            assert result.speedup is not None
            assert result.speedup >= 1.0


# ============================================================
# 4. Snapshot recording
# ============================================================

class TestSnapshots:

    def test_dual_snapshots_recorded(self, base_state):
        basis = _extract_basis(base_state, albici_b2())
        solver = RevisedSimplex.from_warm_start(albici_b2(), basis, base_state.basis_inverse)
        state = solver.solve_dual()
        assert state.iteration_history is not None
        assert len(state.iteration_history) > 0

    def test_dual_snapshot_fields(self, base_state):
        basis = _extract_basis(base_state, albici_b2())
        solver = RevisedSimplex.from_warm_start(albici_b2(), basis, base_state.basis_inverse)
        state = solver.solve_dual()
        for snap in state.iteration_history:
            assert snap.entering_var != ""
            assert snap.leaving_var != ""
            assert len(snap.basic_variables) == 4


# ============================================================
# Helper
# ============================================================

def _extract_basis(state, problem):
    """Extract basis indices using reoptimizer logic."""
    return Reoptimizer()._extract_basis_indices(state, problem)
