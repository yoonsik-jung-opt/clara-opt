"""B^-1 reconstruction validation for the HiGHS backend.

Replaces the old internal-vs-HiGHS cross-validation (the internal
engine was removed in v0.3.0). Validates that the basis inverse
reconstructed from the HiGHS optimal basis is numerically consistent
with the reported solution:

    - B_inv @ B = I on the augmented system [A_aug | I]
    - x_B = B_inv @ b_aug reproduces the primal solution
    - duals from c_B^T B_inv match HiGHS row duals (native sense)
    - diagnostics (condition number, d0, degenerate count) are populated
"""

import numpy as np
import pytest
from pathlib import Path

from clara.io.lp_parser import read_lp
from clara.engine import HiGHSBackend
import clara.engine.standard_form as standard_form
from clara.model.solve_state import BasisStatus, EngineType, SolveStatus

FIXTURES = Path(__file__).parent / "fixtures"

TOL = 1e-8
TOL_SOL = 1e-6

LP_FILES = [
    str(FIXTURES / "albici_base.lp"),
    str(FIXTURES / "albici_reopt_b1.lp"),
    str(FIXTURES / "albici_reopt_b2.lp"),
    str(FIXTURES / "albici_reopt_cost.lp"),
    str(FIXTURES / "minimize_small.lp"),
]


@pytest.fixture(params=LP_FILES)
def solved(request):
    problem = read_lp(request.param)
    state = HiGHSBackend().solve(problem)
    return state, problem


class TestReconstruction:

    def test_optimal(self, solved):
        state, _ = solved
        assert state.status == SolveStatus.OPTIMAL
        assert state.engine == EngineType.HIGHS

    def test_basis_inverse_present(self, solved):
        state, _ = solved
        assert state.basis_inverse is not None
        assert state.basis_indices is not None

    def test_binv_times_b_is_identity(self, solved):
        state, problem = solved
        A_aug, _, _ = standard_form.add_upper_bound_rows(problem)
        m_aug = A_aug.shape[0]
        A_full = np.hstack([A_aug, np.eye(m_aug)])
        B = A_full[:, list(state.basis_indices)]
        err = np.max(np.abs(state.basis_inverse @ B - np.eye(m_aug)))
        assert err < 1e-8

    def test_xb_reproduces_solution(self, solved):
        state, problem = solved
        n = problem.num_variables
        _, b_aug, _ = standard_form.add_upper_bound_rows(problem)
        x_B = state.basis_inverse @ b_aug
        values = {v.name: v.value for v in state.variables}
        for pos, j in enumerate(state.basis_indices):
            if j < n:
                assert abs(x_B[pos] - values[problem.var_names[j]]) < TOL_SOL
        # Nonbasic structural variables not at an upper bound sit at zero
        basis_set = set(state.basis_indices)
        for j in range(n):
            if j not in basis_set:
                assert abs(values[problem.var_names[j]]) < TOL_SOL

    def test_duals_match(self, solved):
        state, problem = solved
        n = problem.num_variables
        m = problem.num_constraints
        m_aug = state.basis_inverse.shape[0]
        c_full = np.zeros(n + m_aug)
        c_full[:n] = problem.c
        c_B = np.array([c_full[j] for j in state.basis_indices])
        y = c_B @ state.basis_inverse
        duals = np.array([c.dual_value for c in state.constraints])
        assert np.max(np.abs(y[:m] - duals)) < 1e-6

    def test_diagnostics_populated(self, solved):
        state, _ = solved
        assert state.condition_number is not None and state.condition_number >= 1.0
        assert state.basis_robustness_d0 is not None
        assert state.basis_robustness_d0 >= -TOL_SOL
        assert state.degenerate_count >= 0

    def test_d0_definition(self, solved):
        state, problem = solved
        _, b_aug, _ = standard_form.add_upper_bound_rows(problem)
        B_inv = state.basis_inverse
        x_B = B_inv @ b_aug
        norms = np.linalg.norm(B_inv, axis=1)
        safe = np.where(norms > 1e-12, norms, np.inf)
        d0 = float(np.min(x_B / safe))
        assert abs(d0 - state.basis_robustness_d0) < 1e-9


class TestWarmStartRoundTrip:

    def test_basis_round_trip(self, solved):
        """Feeding the reconstructed basis back reproduces the optimum."""
        state, problem = solved
        warm = HiGHSBackend().solve(problem, initial_basis=state.basis_indices)
        assert warm.status == SolveStatus.OPTIMAL
        assert abs(warm.optimal_value - state.optimal_value) < TOL_SOL
