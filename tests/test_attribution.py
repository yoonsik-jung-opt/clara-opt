"""Tests for objective change attribution."""

import numpy as np
import pytest

from clara.engine import HiGHSBackend
from clara.model.problem import LPProblem
from clara.reopt.attribution import ChangeAttributor
from clara.reopt.detector import ChangeDetector

TOL = 1.0  # loose tolerance for first-order approximation


def albici_base():
    return LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )


def albici_compound():
    return LPProblem(
        c=[8, 6, 7], A=albici_base().A, b=[1500, 1300, 2400, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )


def albici_b1():
    p = albici_base()
    return LPProblem(c=p.c, A=p.A, b=[1300, 1200, 1800, 1000],
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))


@pytest.fixture
def base_state():
    return HiGHSBackend().solve(albici_base())


@pytest.fixture
def compound_state():
    return HiGHSBackend().solve(albici_compound())


class TestFirstOrder:

    def test_rhs_effect(self, base_state, compound_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        attr = ChangeAttributor().attribute(
            base_state, compound_state, albici_base(), albici_compound(), change
        )
        # yᵀΔb ≈ 0.7778*300 + 0*(-100) + 1.1111*400 + 0.6667*0 ≈ 677.78
        assert abs(attr.rhs_effect - 677.78) < TOL

    def test_obj_effect(self, base_state, compound_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        attr = ChangeAttributor().attribute(
            base_state, compound_state, albici_base(), albici_compound(), change
        )
        # Δcᵀx = 5*755.56 + 2*133.33 + 2*177.78 ≈ 4400
        assert abs(attr.obj_effect - 4400.0) < 10

    def test_per_constraint(self, base_state, compound_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        attr = ChangeAttributor().attribute(
            base_state, compound_state, albici_base(), albici_compound(), change
        )
        assert abs(attr.rhs_contributions["S1"] - 233.33) < TOL
        assert abs(attr.rhs_contributions["S3"] - 444.44) < TOL
        assert abs(attr.rhs_contributions["S2"]) < TOL  # S2 dual ≈ 0

    def test_per_variable(self, base_state, compound_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        attr = ChangeAttributor().attribute(
            base_state, compound_state, albici_base(), albici_compound(), change
        )
        assert abs(attr.obj_contributions["x1"] - 3777.78) < 10
        assert abs(attr.obj_contributions["x2"] - 266.67) < 5
        assert abs(attr.obj_contributions["x3"] - 355.56) < 5


class TestShapley:

    def test_shapley_sum(self, base_state, compound_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        attr = ChangeAttributor().attribute(
            base_state, compound_state, albici_base(), albici_compound(), change
        )
        assert attr.shapley_b is not None
        assert attr.shapley_c is not None
        assert abs(attr.shapley_b + attr.shapley_c - attr.delta_z) < 0.1

    def test_shapley_z_values(self, base_state, compound_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        attr = ChangeAttributor().attribute(
            base_state, compound_state, albici_base(), albici_compound(), change
        )
        # z_b_only: base with b2 RHS → 4300
        assert abs(attr.z_b_only - 4300.0) < 1.0
        # z_c_only: base with cost [8,6,7] → 24800/3 ≈ 8266.67
        assert abs(attr.z_c_only - 24800 / 3) < 1.0


class TestBasisPreserved:

    def test_within_range(self, base_state):
        """Small single-RHS change → exact first-order."""
        new = LPProblem(c=albici_base().c, A=albici_base().A, b=[1300, 1400, 2000, 800],
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        new_state = HiGHSBackend().solve(new)
        change = ChangeDetector().detect(albici_base(), new)
        attr = ChangeAttributor().attribute(
            base_state, new_state, albici_base(), new, change,
            compute_shapley=False,
        )
        assert attr.basis_preserved
        assert abs(attr.first_order_residual) < 0.01


class TestSummary:

    def test_summary_has_percentages(self, base_state, compound_state):
        change = ChangeDetector().detect(albici_base(), albici_compound())
        attr = ChangeAttributor().attribute(
            base_state, compound_state, albici_base(), albici_compound(), change
        )
        assert "%" in attr.summary
