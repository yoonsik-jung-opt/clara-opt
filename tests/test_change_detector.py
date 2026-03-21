"""Tests for Change Detector using Albici scenarios and edge cases."""

import numpy as np
import pytest

from clara.model.problem import LPProblem
from clara.reopt.detector import ChangeDetector
from clara.reopt.types import ChangeType, IncompatibleProblemsError

TOL = 1e-10


# ============================================================
# Helpers
# ============================================================

def albici_base() -> LPProblem:
    return LPProblem(
        c=[3, 4, 5],
        A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"],
        constraint_names=["S1", "S2", "S3", "S4"],
        name="albici_base",
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
    """Albici with 2 new variables x4, x5."""
    return LPProblem(
        c=[3, 4, 5, 3, 5],
        A=[[1, 2, 1, 2, 3], [1, 0, 3, 0, 1], [2, 1, 2, 1, 0], [0, 2, 3, 0, 1]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3", "x4", "x5"],
        constraint_names=["S1", "S2", "S3", "S4"],
    )


# ============================================================
# 1. Albici scenarios
# ============================================================

class TestAlbiciScenarios:
    """Golden tests from Albici et al. (2010)."""

    def test_base_to_b1(self):
        change = ChangeDetector().detect(albici_base(), albici_b1())
        assert change.change_type == ChangeType.TYPE_R
        np.testing.assert_array_almost_equal(change.delta_b, [100, -200, -200, 200])

    def test_base_to_b2(self):
        change = ChangeDetector().detect(albici_base(), albici_b2())
        assert change.change_type == ChangeType.TYPE_R
        np.testing.assert_array_almost_equal(change.delta_b, [300, -100, 400, 0])

    def test_base_to_cost(self):
        change = ChangeDetector().detect(albici_base(), albici_cost())
        assert change.change_type == ChangeType.TYPE_C
        np.testing.assert_array_almost_equal(change.delta_c, [5, 2, 2])

    def test_base_to_columns(self):
        change = ChangeDetector().detect(albici_base(), albici_columns())
        assert change.change_type == ChangeType.TYPE_V
        assert "x4" in change.new_columns
        assert "x5" in change.new_columns

    def test_compound_b2_cost(self):
        """Change both RHS and obj coefficients → TYPE_RC."""
        new = LPProblem(
            c=[8, 6, 7],
            A=albici_base().A,
            b=[1500, 1300, 2400, 800],
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3", "S4"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change.change_type == ChangeType.TYPE_RC
        assert change.delta_b is not None
        assert change.delta_c is not None


# ============================================================
# 2. No change
# ============================================================

class TestNoChange:

    def test_identical_problems(self):
        change = ChangeDetector().detect(albici_base(), albici_base())
        assert change is None

    def test_negligible_change(self):
        """Change smaller than tolerance → None."""
        new = LPProblem(
            c=[3 + 1e-15, 4, 5],
            A=albici_base().A,
            b=albici_base().b,
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3", "S4"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change is None


# ============================================================
# 3. Structural changes
# ============================================================

class TestStructuralChanges:

    def test_add_constraint(self):
        new = LPProblem(
            c=[3, 4, 5],
            A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3], [1, 1, 1]],
            b=[1200, 1400, 2000, 800, 500],
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3", "S4", "S5"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change.change_type == ChangeType.TYPE_X
        assert "S5" in change.new_rows

    def test_remove_constraint(self):
        """Remove S4."""
        new = LPProblem(
            c=[3, 4, 5],
            A=[[1, 2, 1], [1, 0, 3], [2, 1, 2]],
            b=[1200, 1400, 2000],
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change.change_type == ChangeType.TYPE_M
        assert "S4" in change.removed_rows

    def test_mixed_structural(self):
        """Add variable + change b → TYPE_MULTI."""
        new = LPProblem(
            c=[3, 4, 5, 2],
            A=[[1, 2, 1, 1], [1, 0, 3, 0], [2, 1, 2, 1], [0, 2, 3, 0]],
            b=[1300, 1400, 2000, 800],
            var_names=["x1", "x2", "x3", "x4"],
            constraint_names=["S1", "S2", "S3", "S4"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change.change_type == ChangeType.TYPE_MULTI


# ============================================================
# 4. Edge cases
# ============================================================

class TestEdgeCases:

    def test_incompatible_problems(self):
        old = LPProblem(c=[1], A=[[1]], b=[10], var_names=["x"])
        new = LPProblem(c=[1], A=[[1]], b=[10], var_names=["y"])
        with pytest.raises(IncompatibleProblemsError):
            ChangeDetector().detect(old, new)

    def test_single_variable_change(self):
        new = LPProblem(
            c=[3, 4, 10],  # only x3 changed
            A=albici_base().A,
            b=albici_base().b,
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3", "S4"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change.change_type == ChangeType.TYPE_C
        assert abs(change.delta_c[2] - 5.0) < TOL  # x3: 10-5=5

    def test_single_constraint_change(self):
        new = LPProblem(
            c=albici_base().c,
            A=albici_base().A,
            b=[1200, 1500, 2000, 800],  # only S2 changed
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3", "S4"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change.change_type == ChangeType.TYPE_R
        assert abs(change.delta_b[1] - 100.0) < TOL

    def test_a_matrix_change(self):
        A_new = np.array(albici_base().A, dtype=float)
        A_new[0, 0] = 999.0
        new = LPProblem(
            c=albici_base().c, A=A_new, b=albici_base().b,
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3", "S4"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert change.change_type == ChangeType.TYPE_A


# ============================================================
# 5. Summary property
# ============================================================

class TestSummary:

    def test_summary_type_r(self):
        change = ChangeDetector().detect(albici_base(), albici_b1())
        assert "RHS changed" in change.summary
        assert "4 constraint(s)" in change.summary

    def test_summary_type_c(self):
        change = ChangeDetector().detect(albici_base(), albici_cost())
        assert "Objective coefficients changed" in change.summary
        assert "3 variable(s)" in change.summary

    def test_summary_type_rc(self):
        new = LPProblem(
            c=[8, 6, 7], A=albici_base().A, b=[1500, 1300, 2400, 800],
            var_names=["x1", "x2", "x3"],
            constraint_names=["S1", "S2", "S3", "S4"],
        )
        change = ChangeDetector().detect(albici_base(), new)
        assert "compound" in change.summary.lower()

    def test_summary_type_v(self):
        change = ChangeDetector().detect(albici_base(), albici_columns())
        assert "2 new variable(s)" in change.summary


# ============================================================
# 6. Delta vector correctness
# ============================================================

class TestDeltaVectors:

    def test_delta_b_matches_manual(self):
        """Compare against hand-computed Albici deltas from albici_expected.py."""
        change = ChangeDetector().detect(albici_base(), albici_b1())
        np.testing.assert_array_almost_equal(change.delta_b, [100, -200, -200, 200])

    def test_delta_c_preserves_order(self):
        """delta_c indices match old.var_names ordering."""
        change = ChangeDetector().detect(albici_base(), albici_cost())
        # old order: x1(3), x2(4), x3(5) → new: x1(8), x2(6), x3(7)
        # delta: [5, 2, 2]
        np.testing.assert_array_almost_equal(change.delta_c, [5, 2, 2])

    def test_new_column_values(self):
        """Column coefficients match new problem's A matrix."""
        change = ChangeDetector().detect(albici_base(), albici_columns())
        x4_col = change.new_columns["x4"]["column"]
        assert x4_col == [2.0, 0.0, 1.0, 0.0]
        x5_col = change.new_columns["x5"]["column"]
        assert x5_col == [3.0, 1.0, 0.0, 1.0]

    def test_new_column_costs(self):
        change = ChangeDetector().detect(albici_base(), albici_columns())
        assert change.new_columns["x4"]["cost"] == 3.0
        assert change.new_columns["x5"]["cost"] == 5.0
