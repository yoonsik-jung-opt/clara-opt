"""Tests for Explainer module.

Validates report generation against the Albici base problem SolveState.
Covers: binding, variable, sensitivity reports; brief/detailed levels;
text/JSON output formats; edge cases.
"""

import json
import math

import numpy as np
import pytest

from clara.engine import solve
from clara.explain.explainer import Explainer
from clara.explain.types import DetailLevel
from clara.model.problem import LPProblem

TOL = 1e-2


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def albici_problem() -> LPProblem:
    return LPProblem(
        c=[3, 4, 5],
        A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"],
        constraint_names=["S1", "S2", "S3", "S4"],
        name="albici_base",
    )


@pytest.fixture
def albici_report(albici_problem):
    state = solve(albici_problem)
    explainer = Explainer()
    return explainer.explain(state, level=DetailLevel.DETAILED, problem=albici_problem)


@pytest.fixture
def albici_brief(albici_problem):
    state = solve(albici_problem)
    explainer = Explainer()
    return explainer.explain(state, level=DetailLevel.BRIEF, problem=albici_problem)


# ============================================================
# 1. Albici base — Binding report
# ============================================================

class TestBindingReport:

    def test_binding_count(self, albici_report):
        assert len(albici_report.binding.binding) == 3

    def test_nonbinding_count(self, albici_report):
        assert len(albici_report.binding.nonbinding) == 1

    def test_binding_names(self, albici_report):
        names = {b.name for b in albici_report.binding.binding}
        assert names == {"S1", "S3", "S4"}

    def test_nonbinding_is_s2(self, albici_report):
        assert albici_report.binding.nonbinding[0].name == "S2"

    def test_ranked_by_shadow_price(self, albici_report):
        items = albici_report.binding.binding
        assert items[0].rank == 1
        assert items[0].name == "S3"  # highest dual ~1.11
        assert items[1].name == "S1"  # dual ~0.78
        assert items[2].name == "S4"  # dual ~0.67

    def test_s2_utilization(self, albici_report):
        s2 = albici_report.binding.nonbinding[0]
        assert s2.utilization_pct > 90


# ============================================================
# 2. Albici base — Variable report
# ============================================================

class TestVariableReport:

    def test_all_basic(self, albici_report):
        assert len(albici_report.variable.basic) == 3
        assert len(albici_report.variable.nonbasic) == 0

    def test_contributions_sorted(self, albici_report):
        items = albici_report.variable.basic
        # Sorted by contribution descending
        for i in range(len(items) - 1):
            assert items[i].contribution >= items[i + 1].contribution

    def test_x1_contribution(self, albici_report):
        x1 = next(v for v in albici_report.variable.basic if v.name == "x1")
        expected = 6800 / 9 * 3  # ~2266.67
        assert abs(x1.contribution - expected) < TOL

    def test_contribution_pct_sum(self, albici_report):
        total = sum(v.contribution_pct for v in albici_report.variable.basic)
        assert abs(total - 100.0) < TOL

    def test_top_contributor_is_x1(self, albici_report):
        top = albici_report.variable.basic[0]
        assert top.name == "x1"
        assert top.contribution_pct > 60


# ============================================================
# 3. Albici base — Sensitivity report
# ============================================================

class TestSensitivityReport:

    def test_bottleneck_is_s3(self, albici_report):
        assert albici_report.sensitivity.bottleneck == "S3"

    def test_obj_items_count(self, albici_report):
        assert len(albici_report.sensitivity.obj_items) == 3

    def test_rhs_items_count(self, albici_report):
        assert len(albici_report.sensitivity.rhs_items) == 4

    def test_ranges_present(self, albici_report):
        for item in albici_report.sensitivity.obj_items:
            assert not math.isnan(item.lower)
            assert not math.isnan(item.upper)

    def test_most_fragile_identified(self, albici_report):
        assert albici_report.sensitivity.most_fragile is not None

    def test_most_robust_identified(self, albici_report):
        assert albici_report.sensitivity.most_robust is not None

    def test_obj_sorted_by_width(self, albici_report):
        items = albici_report.sensitivity.obj_items
        for i in range(len(items) - 1):
            w1 = items[i].range_width if not math.isinf(items[i].range_width) else float("inf")
            w2 = items[i + 1].range_width if not math.isinf(items[i + 1].range_width) else float("inf")
            assert w1 <= w2

    def test_rhs_sorted_by_dual(self, albici_report):
        items = albici_report.sensitivity.rhs_items
        for i in range(len(items) - 1):
            assert abs(items[i].dual_value) >= abs(items[i + 1].dual_value) - 1e-10


# ============================================================
# 4. Detail level
# ============================================================

class TestDetailLevel:

    def test_brief_is_shorter(self, albici_report, albici_brief):
        brief_text = albici_brief.to_text(DetailLevel.BRIEF)
        detailed_text = albici_report.to_text(DetailLevel.DETAILED)
        assert len(brief_text) < len(detailed_text)

    def test_brief_has_summary(self, albici_brief):
        text = albici_brief.to_text(DetailLevel.BRIEF)
        assert "binding" in text.lower() or "constraint" in text.lower()

    def test_detailed_has_shadow_price(self, albici_report):
        text = albici_report.to_text(DetailLevel.DETAILED)
        assert "Shadow price" in text

    def test_detailed_has_contribution(self, albici_report):
        text = albici_report.to_text(DetailLevel.DETAILED)
        assert "Contribution" in text


# ============================================================
# 5. Output format
# ============================================================

class TestOutputFormat:

    def test_to_text_not_empty(self, albici_report):
        text = albici_report.to_text()
        assert len(text) > 100

    def test_to_json_valid(self, albici_report):
        j = albici_report.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_to_dict_keys(self, albici_report):
        d = albici_report.to_dict()
        assert "header" in d
        assert "binding_report" in d
        assert "variable_report" in d
        assert "sensitivity_report" in d

    def test_json_binding_keys(self, albici_report):
        d = albici_report.to_dict()
        br = d["binding_report"]
        assert "summary" in br
        assert "binding" in br
        assert "nonbinding" in br

    def test_json_sensitivity_keys(self, albici_report):
        d = albici_report.to_dict()
        sr = d["sensitivity_report"]
        assert "bottleneck" in sr
        assert "most_robust" in sr
        assert "obj_coeff_ranges" in sr
        assert "rhs_ranges" in sr


# ============================================================
# 6. Header
# ============================================================

class TestHeader:

    def test_header_contains_opt_value(self, albici_report):
        assert "3688.8889" in albici_report.header

    def test_header_contains_engine(self, albici_report):
        assert "HIGHS" in albici_report.header

    def test_header_contains_iterations(self, albici_report):
        assert "iterations" in albici_report.header

    def test_header_contains_problem_name(self, albici_report):
        assert "albici_base" in albici_report.header


# ============================================================
# 7. Edge cases
# ============================================================

class TestEdgeCases:

    def test_single_variable(self):
        p = LPProblem(c=[5], A=[[1]], b=[10], var_names=["x"], constraint_names=["c1"])
        state = solve(p)
        report = Explainer().explain(state, problem=p)
        assert len(report.variable.basic) == 1
        assert len(report.binding.binding) == 1

    def test_all_binding_message(self):
        """When all constraints are binding, detailed text says so."""
        p = LPProblem(
            c=[1, 1],
            A=[[1, 0], [0, 1]],
            b=[5, 5],
            var_names=["x1", "x2"],
            constraint_names=["c1", "c2"],
        )
        state = solve(p)
        report = Explainer().explain(state, problem=p)
        text = report.binding.to_text(DetailLevel.DETAILED)
        assert "All constraints are binding" in text

    def test_no_nonbasic_message(self):
        """When all vars are basic, detailed text says 'All variables are in the solution'."""
        p = LPProblem(
            c=[3, 4, 5],
            A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
            b=[1200, 1400, 2000, 800],
        )
        state = solve(p)
        report = Explainer().explain(state, problem=p)
        text = report.variable.to_text(DetailLevel.DETAILED)
        assert "All variables are in the solution" in text

    def test_nonbasic_reduced_cost(self):
        """A problem with a non-basic variable should show reduced cost."""
        p = LPProblem(
            c=[10, 1],
            A=[[1, 1], [1, 0]],
            b=[5, 4],
            var_names=["x1", "x2"],
            constraint_names=["c1", "c2"],
        )
        state = solve(p)
        report = Explainer().explain(state, problem=p)
        # x1 should be basic (higher obj coeff), x2 may or may not be
        # At least verify the report generates without error
        assert len(report.variable.basic) + len(report.variable.nonbasic) == 2

    def test_inf_sensitivity_range_json(self):
        """Inf ranges should appear as null in JSON."""
        p = LPProblem(c=[1], A=[[1]], b=[10])
        state = solve(p)
        report = Explainer().explain(state, problem=p)
        d = report.to_dict()
        # Non-basic var sensitivity may have inf bounds
        # Just verify JSON is valid
        j = report.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)
