"""Tests for Revised Simplex engine against Albici base problem.

Validates:
    - Optimal value = 33200/9 ≈ 3688.8889
    - Basis inverse matches known B⁻¹
    - Iteration history is recorded
    - Sensitivity analysis ranges
    - SolveState structure completeness
"""

import numpy as np
import pytest

from clara.engine.simplex import RevisedSimplex, solve
from clara.model.problem import LPProblem
from clara.model.solve_state import BasisStatus, EngineType, SolveStatus

TOL_OPTIMAL = 1e-6
TOL_VARIABLE = 1e-4


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def albici_problem() -> LPProblem:
    """Albici et al. (2010) base problem."""
    return LPProblem(
        c=[3, 4, 5],
        A=[
            [1, 2, 1],
            [1, 0, 3],
            [2, 1, 2],
            [0, 2, 3],
        ],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"],
        constraint_names=["S1", "S2", "S3", "S4"],
        name="albici_base",
    )


EXPECTED_OPTIMAL = 33200 / 9  # ≈ 3688.8889
EXPECTED_X = np.array([6800 / 9, 1200 / 9, 1600 / 9])


# ============================================================
# Core solve tests
# ============================================================

class TestAlbiciSolve:
    """End-to-end solve of Albici base problem."""

    def test_status_optimal(self, albici_problem):
        state = solve(albici_problem)
        assert state.status == SolveStatus.OPTIMAL

    def test_optimal_value(self, albici_problem):
        state = solve(albici_problem)
        assert abs(state.optimal_value - EXPECTED_OPTIMAL) < TOL_OPTIMAL, (
            f"Expected {EXPECTED_OPTIMAL}, got {state.optimal_value}"
        )

    def test_variable_values(self, albici_problem):
        state = solve(albici_problem)
        vals = state.variable_values_dict
        assert abs(vals["x1"] - EXPECTED_X[0]) < TOL_VARIABLE
        assert abs(vals["x2"] - EXPECTED_X[1]) < TOL_VARIABLE
        assert abs(vals["x3"] - EXPECTED_X[2]) < TOL_VARIABLE

    def test_all_variables_basic(self, albici_problem):
        """All 3 decision variables should be in the basis."""
        state = solve(albici_problem)
        for v in state.variables:
            assert v.basis_status == BasisStatus.BASIC, (
                f"{v.name} should be basic, got {v.basis_status}"
            )

    def test_engine_type(self, albici_problem):
        state = solve(albici_problem)
        assert state.engine == EngineType.INTERNAL_SIMPLEX

    def test_problem_name(self, albici_problem):
        state = solve(albici_problem)
        assert state.problem_name == "albici_base"


# ============================================================
# Basis inverse tests
# ============================================================

class TestBasisInverse:
    """Verify B⁻¹ is correct and consistent."""

    def test_basis_inverse_available(self, albici_problem):
        state = solve(albici_problem)
        assert state.has_basis_inverse
        assert state.basis_inverse is not None

    def test_basis_inverse_shape(self, albici_problem):
        state = solve(albici_problem)
        assert state.basis_inverse.shape == (4, 4)

    def test_basis_inverse_correctness(self, albici_problem):
        """B * B⁻¹ = I for the final basis."""
        state = solve(albici_problem)
        B_inv = state.basis_inverse

        solver = RevisedSimplex(albici_problem)
        solver.solve()
        B = np.column_stack([solver.A_full[:, j] for j in solver.basis])

        assert np.allclose(B @ B_inv, np.eye(4), atol=1e-10), (
            f"B @ B⁻¹ ≠ I:\n{B @ B_inv}"
        )

    def test_x_B_from_basis_inverse(self, albici_problem):
        """B⁻¹ * b gives correct basic variable values."""
        state = solve(albici_problem)
        x_B = state.basis_inverse @ albici_problem.b
        assert np.all(x_B >= -TOL_VARIABLE)

        solver = RevisedSimplex(albici_problem)
        solver.solve()
        c_B = np.array([solver.c_full[j] for j in solver.basis])
        obj = c_B @ x_B
        assert abs(obj - EXPECTED_OPTIMAL) < TOL_OPTIMAL


# ============================================================
# Iteration history tests
# ============================================================

class TestIterationHistory:
    """Verify iteration snapshots are recorded."""

    def test_has_iteration_history(self, albici_problem):
        state = solve(albici_problem)
        assert state.has_internal_trace
        assert state.iteration_history is not None
        assert len(state.iteration_history) > 0

    def test_iteration_count_matches(self, albici_problem):
        state = solve(albici_problem)
        assert state.iteration_count == len(state.iteration_history)

    def test_objective_monotone_increasing(self, albici_problem):
        """Objective should increase at each iteration (maximization)."""
        state = solve(albici_problem)
        prev = 0.0
        for snap in state.iteration_history:
            assert snap.objective_value >= prev - TOL_OPTIMAL, (
                f"Objective decreased: {prev} → {snap.objective_value}"
            )
            prev = snap.objective_value

    def test_final_iteration_objective(self, albici_problem):
        state = solve(albici_problem)
        last = state.iteration_history[-1]
        assert abs(last.objective_value - EXPECTED_OPTIMAL) < TOL_OPTIMAL

    def test_snapshot_fields_populated(self, albici_problem):
        state = solve(albici_problem)
        for snap in state.iteration_history:
            assert snap.entering_var in [f"x{j}" for j in range(1, 4)] + [f"y{i}" for i in range(1, 5)]
            assert snap.leaving_var in [f"x{j}" for j in range(1, 4)] + [f"y{i}" for i in range(1, 5)]
            assert snap.pivot_element != 0.0
            assert len(snap.basic_variables) == 4


# ============================================================
# Constraint / dual value tests
# ============================================================

class TestConstraints:
    """Verify constraint info and dual values."""

    def test_binding_constraints(self, albici_problem):
        """S1, S3, S4 should be binding; S2 should not be."""
        state = solve(albici_problem)
        binding = {c.name for c in state.binding_constraints}
        assert "S1" in binding
        assert "S3" in binding
        assert "S4" in binding
        assert "S2" not in binding

    def test_dual_values_nonneg(self, albici_problem):
        """Shadow prices should be non-negative for <= constraints in max."""
        state = solve(albici_problem)
        for c in state.constraints:
            assert c.dual_value >= -TOL_OPTIMAL, (
                f"Dual of {c.name} is negative: {c.dual_value}"
            )

    def test_s2_dual_zero(self, albici_problem):
        """S2 is non-binding, so its dual should be 0."""
        state = solve(albici_problem)
        s2 = state.get_constraint("S2")
        assert abs(s2.dual_value) < TOL_OPTIMAL

    def test_slack_values(self, albici_problem):
        """Slacks should be non-negative."""
        state = solve(albici_problem)
        for c in state.constraints:
            assert c.slack >= -TOL_VARIABLE, (
                f"Negative slack for {c.name}: {c.slack}"
            )


# ============================================================
# Sensitivity analysis tests
# ============================================================

class TestSensitivity:
    """Verify sensitivity ranges contain current values."""

    def test_obj_ranges_contain_current(self, albici_problem):
        """Current obj coefficients should be inside their ranges."""
        state = solve(albici_problem)
        for j, v in enumerate(state.variables):
            lo, hi = v.obj_coeff_range
            c_j = albici_problem.c[j]
            assert lo <= c_j + TOL_OPTIMAL, (
                f"{v.name}: c={c_j} below range lower {lo}"
            )
            assert c_j <= hi + TOL_OPTIMAL, (
                f"{v.name}: c={c_j} above range upper {hi}"
            )

    def test_rhs_ranges_contain_current(self, albici_problem):
        """Current RHS values should be inside their ranges."""
        state = solve(albici_problem)
        for i, c in enumerate(state.constraints):
            lo, hi = c.rhs_range
            b_i = albici_problem.b[i]
            assert lo <= b_i + TOL_OPTIMAL, (
                f"{c.name}: b={b_i} below range lower {lo}"
            )
            assert b_i <= hi + TOL_OPTIMAL, (
                f"{c.name}: b={b_i} above range upper {hi}"
            )


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    """Simple edge-case LPs."""

    def test_single_variable(self):
        """max 5x s.t. x <= 10."""
        p = LPProblem(c=[5], A=[[1]], b=[10])
        state = solve(p)
        assert state.status == SolveStatus.OPTIMAL
        assert abs(state.optimal_value - 50.0) < TOL_OPTIMAL

    def test_two_variables(self):
        """max 3x + 2y s.t. x + y <= 4, x + 3y <= 6."""
        p = LPProblem(c=[3, 2], A=[[1, 1], [1, 3]], b=[4, 6])
        state = solve(p)
        assert state.status == SolveStatus.OPTIMAL
        assert abs(state.optimal_value - 12.0) < TOL_OPTIMAL

    def test_already_optimal(self):
        """max -x s.t. x <= 10 → optimal at x=0, value=0."""
        p = LPProblem(c=[-1], A=[[1]], b=[10])
        state = solve(p)
        assert state.status == SolveStatus.OPTIMAL
        assert abs(state.optimal_value - 0.0) < TOL_OPTIMAL
