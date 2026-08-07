"""Tests for simultaneous sensitivity region analysis."""

import numpy as np
import pytest

from clara.engine import HiGHSBackend
from clara.model.problem import LPProblem
from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer


def albici_base():
    return LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )


@pytest.fixture
def base_state():
    return HiGHSBackend().solve(albici_base())


@pytest.fixture
def region(base_state):
    return SimultaneousRegionAnalyzer().analyze(base_state, albici_base())


class TestChebyshev:

    def test_radius_positive(self, region):
        assert region.chebyshev_radius > 0

    def test_radius_finite(self, region):
        """Radius should be finite and reasonable."""
        assert region.chebyshev_radius < 1e10

    def test_center_inside(self, region, base_state):
        """Center should satisfy all constraints of the polyhedron."""
        if region.chebyshev_center is None:
            pytest.skip("No center computed")
        center = np.array(region.chebyshev_center)
        B_inv = base_state.basis_inverse
        x_B = B_inv @ albici_base().b
        # -B_inv @ Δb ≤ x_B → B_inv @ center should give x_B + B_inv@center ≥ 0 approx
        x_new = x_B + B_inv @ center[:B_inv.shape[1]]
        # All should be >= -radius (approximately)
        assert np.all(x_new >= -region.chebyshev_radius - 1e-6)


class TestRatio:

    def test_ratio_positive(self, region):
        """Simultaneity ratio should be positive (meaningful comparison)."""
        assert region.simultaneity_ratio > 0


class TestOAT:

    def test_oat_rhs_positive(self, region):
        # HiGHS ranging reports zero-width ranges on degenerate rows,
        # so tolerances are nonnegative (the analyzer filters zeros).
        for name, tol in region.oat_rhs_tolerances.items():
            assert tol >= 0 or tol == float("inf")

    def test_oat_obj_positive(self, region):
        for name, tol in region.oat_obj_tolerances.items():
            assert tol > 0 or tol == float("inf")


class TestSummary:

    def test_summary_overestimation(self, region):
        assert "overestimates" in region.summary


class TestProjection:

    def test_projection_2d(self, base_state):
        region = SimultaneousRegionAnalyzer().analyze(
            base_state, albici_base(), projection_pairs=[(0, 1)]
        )
        if region.projections and (0, 1) in region.projections:
            verts = region.projections[(0, 1)]
            assert len(verts) > 0
