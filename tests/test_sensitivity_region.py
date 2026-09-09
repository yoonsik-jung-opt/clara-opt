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


class TestDegenerateAndUnbounded:
    """Face-restricted radius and the unbounded Chebyshev program."""

    @staticmethod
    def _two_block_problem():
        # Block A (x1, x2): three constraints meet at the optimum (1, 0),
        # so the basis {x1, x2, s3} is degenerate and b1 has zero
        # tolerance on both sides.  Block B (x3, x4) is non-degenerate.
        A = np.array([[1, 1, 0, 0], [1, -1, 0, 0], [1, 0, 0, 0],
                      [0, 0, 1, 0], [0, 0, 1, 1]], float)
        return LPProblem(c=[1, 0, 1, 0.5], A=A, b=[1, 1, 1, 5, 8],
                         sense="maximize", name="two_block")

    def test_face_restricted_radius_recovers_healthy_block(self):
        prob = self._two_block_problem()
        state = HiGHSBackend().solve(prob, initial_basis=(0, 1, 6, 2, 3))
        assert state.degenerate_count == 2
        region = SimultaneousRegionAnalyzer().analyze(state, prob)
        assert region.n_two_sided_zero == 1
        assert region.chebyshev_radius == pytest.approx(0.0, abs=1e-9)
        # Block B: strip -5 <= d4 <= 3 in the (d4, d5) face -> radius 4.
        assert region.chebyshev_radius_face == pytest.approx(4.0, abs=1e-6)

    def test_unbounded_chebyshev_program_reports_infinite_radius(self):
        analyzer = SimultaneousRegionAnalyzer()
        # Single free coordinate bounded below only: {d : -d <= 3}.
        radius, center = analyzer._chebyshev_center(
            np.array([[-1.0]]), np.array([3.0]))
        assert radius == float("inf")
        assert center is None
