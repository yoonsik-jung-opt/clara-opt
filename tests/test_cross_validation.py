"""Cross-validation: Internal Simplex vs HiGHS on the same problems."""

import numpy as np
import pytest
from pathlib import Path

from clara.io.lp_parser import read_lp
from clara.engine.simplex import RevisedSimplex
from clara.engine.highs_backend import HiGHSBackend
from clara.model.solve_state import EngineType, SolveStatus

FIXTURES = Path(__file__).parent / "fixtures"

TOL_OPTIMAL = 1e-6
TOL_DUAL = 1e-4
TOL_SENSITIVITY = 1e-2


# Only use fixtures with pure <= constraints (internal simplex has no Phase I)
LP_FILES = [str(FIXTURES / "albici_base.lp")]


@pytest.fixture(params=LP_FILES)
def dual_solve(request):
    """Solve the same problem with both engines."""
    problem = read_lp(request.param)
    internal_state = RevisedSimplex(problem).solve()
    highs_state = HiGHSBackend().solve(problem)
    return internal_state, highs_state, problem


class TestCrossValidation:
    """Compare InternalSimplex vs HiGHS."""

    def test_both_optimal(self, dual_solve):
        internal, highs, _ = dual_solve
        assert internal.status == SolveStatus.OPTIMAL
        assert highs.status == SolveStatus.OPTIMAL

    def test_optimal_value_match(self, dual_solve):
        internal, highs, _ = dual_solve
        assert abs(internal.optimal_value - highs.optimal_value) < TOL_OPTIMAL, (
            f"Internal={internal.optimal_value}, HiGHS={highs.optimal_value}"
        )

    def test_variable_values_match(self, dual_solve):
        internal, highs, _ = dual_solve
        for vi, vh in zip(internal.variables, highs.variables):
            assert abs(vi.value - vh.value) < TOL_DUAL, (
                f"{vi.name}: internal={vi.value}, highs={vh.value}"
            )

    def test_dual_values_match(self, dual_solve):
        internal, highs, _ = dual_solve
        for ci, ch in zip(internal.constraints, highs.constraints):
            assert abs(ci.dual_value - ch.dual_value) < TOL_DUAL, (
                f"{ci.name}: internal={ci.dual_value}, highs={ch.dual_value}"
            )

    def test_binding_constraints_match(self, dual_solve):
        internal, highs, _ = dual_solve
        i_binding = {c.name for c in internal.binding_constraints}
        h_binding = {c.name for c in highs.binding_constraints}
        assert i_binding == h_binding

    def test_sensitivity_ranges_close(self, dual_solve):
        internal, highs, _ = dual_solve
        for var_name in internal.sensitivity.obj_coeff_ranges:
            i_lo, i_hi = internal.sensitivity.obj_coeff_ranges[var_name]
            h_lo, h_hi = highs.sensitivity.obj_coeff_ranges[var_name]
            if i_lo != float("-inf") and h_lo != float("-inf"):
                assert abs(i_lo - h_lo) < TOL_SENSITIVITY, (
                    f"{var_name} obj lo: internal={i_lo}, highs={h_lo}"
                )
            if i_hi != float("inf") and h_hi != float("inf"):
                assert abs(i_hi - h_hi) < TOL_SENSITIVITY, (
                    f"{var_name} obj hi: internal={i_hi}, highs={h_hi}"
                )

    def test_highs_no_basis_inverse(self, dual_solve):
        _, highs, _ = dual_solve
        assert highs.basis_inverse is None

    def test_highs_no_iteration_history(self, dual_solve):
        _, highs, _ = dual_solve
        assert highs.iteration_history is None

    def test_engine_types(self, dual_solve):
        internal, highs, _ = dual_solve
        assert internal.engine == EngineType.INTERNAL_SIMPLEX
        assert highs.engine == EngineType.HIGHS


class TestHiGHSStandalone:
    """HiGHS-specific tests."""

    def test_highs_albici_optimal_value(self):
        problem = read_lp(FIXTURES / "albici_base.lp")
        state = HiGHSBackend().solve(problem)
        assert abs(state.optimal_value - 33200 / 9) < TOL_OPTIMAL

    def test_highs_explain_works(self):
        """Explainer produces valid report from HiGHS SolveState."""
        from clara.explain.explainer import Explainer
        problem = read_lp(FIXTURES / "albici_base.lp")
        state = HiGHSBackend().solve(problem)
        report = Explainer().explain(state, problem=problem)
        assert "3688.8889" in report.header

    def test_highs_json_output(self):
        """JSON output includes engine: HIGHS."""
        import json
        problem = read_lp(FIXTURES / "albici_base.lp")
        state = HiGHSBackend().solve(problem)
        d = state.to_dict()
        assert d["engine"] == "HIGHS"
