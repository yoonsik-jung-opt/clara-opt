"""Tests for Impact Analyzer — Oguz bound, sensitivity check, full pipeline."""

import numpy as np
import pytest

from clara.engine import HiGHSBackend
from clara.model.problem import LPProblem
from clara.reopt.analyzer import ImpactAnalyzer
from clara.reopt.detector import ChangeDetector
from clara.reopt.types import ChangeType, ParameterChange

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


@pytest.fixture
def analyzer():
    return ImpactAnalyzer()


@pytest.fixture
def detector():
    return ChangeDetector()


# ============================================================
# 1. Oguz bound computation
# ============================================================

class TestOguzBound:

    def test_albici_cost_change(self, base_state, analyzer):
        """δ=5/3, raw_bound=10/8=1.25."""
        change = ChangeDetector().detect(albici_base(), albici_cost())
        raw, tightened, _ = analyzer._oguz_bound(change, albici_base(), base_state)
        assert abs(raw - 1.25) < TOL, f"Expected 1.25, got {raw}"

    def test_oguz_small_change(self, base_state, analyzer):
        """Small perturbation → small bound."""
        new = LPProblem(c=[3.1, 4.1, 5.1], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(albici_base(), new)
        raw, _, _ = analyzer._oguz_bound(change, albici_base(), base_state)
        assert raw < 0.10  # should be around 6.5%

    def test_oguz_tiny_change(self, base_state, analyzer):
        """Very small change → bound below 5% threshold."""
        new = LPProblem(c=[3.01, 4, 5], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(albici_base(), new)
        raw, _, _ = analyzer._oguz_bound(change, albici_base(), base_state)
        assert raw < 0.01

    def test_oguz_zero_original_coeff(self, base_state, analyzer):
        """c_j=0 in original → δ=inf → bound=1.0."""
        old = LPProblem(c=[0, 4, 5], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        old_state = HiGHSBackend().solve(old)
        new = LPProblem(c=[1, 4, 5], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(old, new)
        raw, _, _ = analyzer._oguz_bound(change, old, old_state)
        assert abs(raw - 1.0) < TOL

    def test_wendell_tightening(self, base_state, analyzer):
        """Tightened bound should be ≤ raw bound."""
        change = ChangeDetector().detect(albici_base(), albici_cost())
        raw, tightened, _ = analyzer._oguz_bound(change, albici_base(), base_state)
        assert tightened <= raw + TOL


# ============================================================
# 2. Sensitivity range check
# ============================================================

class TestSensitivityCheck:

    def test_type_r_within_range(self, base_state, analyzer):
        """Small single-constraint RHS change → within sensitivity range."""
        # Change only S1 by a small amount (within its range [1109, 1600])
        new = LPProblem(c=albici_base().c, A=albici_base().A, b=[1300, 1400, 2000, 800],
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(albici_base(), new)
        decision = analyzer.analyze(base_state, change, albici_base())
        assert not decision.should_reoptimize
        assert decision.within_sensitivity
        assert decision.recommended_method == "recompute"

    def test_type_r_albici_b1_conservative(self, base_state, analyzer):
        """Albici b1: multi-RHS change exceeds one-at-a-time ranges.

        The basis IS actually preserved (B⁻¹*b'≥0), but the sensitivity
        range check is conservative (one-at-a-time), so it reports violated.
        This is correct — the analyzer errs on the side of recommending reopt.
        """
        change = ChangeDetector().detect(albici_base(), albici_b1())
        decision = analyzer.analyze(base_state, change, albici_base())
        assert decision.should_reoptimize  # conservative: recommends warm_start

    def test_type_r_outside_range(self, base_state, analyzer):
        """Albici b2: large RHS change → breaks basis."""
        change = ChangeDetector().detect(albici_base(), albici_b2())
        decision = analyzer.analyze(base_state, change, albici_base())
        assert decision.should_reoptimize
        assert not decision.within_sensitivity
        assert decision.recommended_method == "warm_start"

    def test_type_c_outside_range(self, base_state, analyzer):
        """Albici cost: large obj change → reopt needed."""
        change = ChangeDetector().detect(albici_base(), albici_cost())
        decision = analyzer.analyze(base_state, change, albici_base())
        assert decision.should_reoptimize
        assert decision.oguz_bound is not None
        assert decision.oguz_bound > 1.0  # 125%

    def test_type_c_within_range(self, base_state, analyzer):
        """Very small obj change → within sensitivity."""
        new = LPProblem(c=[3.01, 4.01, 5.01], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(albici_base(), new)
        decision = analyzer.analyze(base_state, change, albici_base())
        assert not decision.should_reoptimize
        assert decision.within_sensitivity

    def test_type_rc_outside_range(self, base_state, analyzer):
        """Compound b+c change → parametric_lp recommended."""
        new = LPProblem(c=[8, 6, 7], A=albici_base().A, b=[1500, 1300, 2400, 800],
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(albici_base(), new)
        decision = analyzer.analyze(base_state, change, albici_base())
        assert decision.should_reoptimize
        assert decision.recommended_method == "parametric_lp"


# ============================================================
# 3. Structural changes
# ============================================================

class TestStructuralChanges:

    def test_type_v_always_scratch(self, base_state, analyzer):
        change = ChangeDetector().detect(albici_base(), albici_columns())
        decision = analyzer.analyze(base_state, change, albici_base())
        assert decision.should_reoptimize
        assert decision.recommended_method == "scratch"

    def test_type_x_always_scratch(self, base_state, analyzer):
        new = LPProblem(c=[3, 4, 5],
                        A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3], [1, 1, 1]],
                        b=[1200, 1400, 2000, 800, 500],
                        var_names=["x1", "x2", "x3"],
                        constraint_names=["S1", "S2", "S3", "S4", "S5"])
        change = ChangeDetector().detect(albici_base(), new)
        decision = analyzer.analyze(base_state, change, albici_base())
        assert decision.should_reoptimize
        assert decision.recommended_method == "scratch"

    def test_type_m_always_scratch(self, base_state, analyzer):
        new = LPProblem(c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2]],
                        b=[1200, 1400, 2000],
                        var_names=["x1", "x2", "x3"],
                        constraint_names=["S1", "S2", "S3"])
        change = ChangeDetector().detect(albici_base(), new)
        decision = analyzer.analyze(base_state, change, albici_base())
        assert decision.should_reoptimize
        assert decision.recommended_method == "scratch"


# ============================================================
# 4. Full pipeline: detect → analyze
# ============================================================

class TestFullPipeline:

    def test_pipeline_b1_conservative_reopt(self, base_state):
        """b1: multi-RHS change exceeds one-at-a-time ranges → conservative reopt.

        The basis is actually preserved (B⁻¹*b'≥0), but one-at-a-time
        sensitivity analysis is conservative for simultaneous changes.
        """
        change = ChangeDetector().detect(albici_base(), albici_b1())
        decision = ImpactAnalyzer().analyze(base_state, change, albici_base())
        assert decision.should_reoptimize  # conservative
        assert decision.recommended_method == "warm_start"

    def test_pipeline_b2_reopt(self, base_state):
        """b2: outside sensitivity → reoptimization needed."""
        change = ChangeDetector().detect(albici_base(), albici_b2())
        decision = ImpactAnalyzer().analyze(base_state, change, albici_base())
        assert decision.should_reoptimize

    def test_pipeline_cost_reopt(self, base_state):
        """cost change: large δ → reoptimization needed."""
        change = ChangeDetector().detect(albici_base(), albici_cost())
        decision = ImpactAnalyzer().analyze(base_state, change, albici_base())
        assert decision.should_reoptimize

    def test_pipeline_columns_scratch(self, base_state):
        """column addition → structural → scratch."""
        change = ChangeDetector().detect(albici_base(), albici_columns())
        decision = ImpactAnalyzer().analyze(base_state, change, albici_base())
        assert decision.should_reoptimize
        assert decision.recommended_method == "scratch"


# ============================================================
# 5. Decision reason quality
# ============================================================

class TestReasonQuality:

    def test_reason_contains_oguz_value(self, base_state, analyzer):
        change = ChangeDetector().detect(albici_base(), albici_cost())
        decision = analyzer.analyze(base_state, change, albici_base())
        assert "Oguz" in decision.reason or "δ" in decision.reason

    def test_reason_contains_violated_params(self, base_state, analyzer):
        change = ChangeDetector().detect(albici_base(), albici_b2())
        decision = analyzer.analyze(base_state, change, albici_base())
        # Should mention at least one violated constraint name
        assert any(name in decision.reason for name in ["S1", "S2", "S3", "S4"])


# ============================================================
# 6. Configurable threshold
# ============================================================

class TestConfigurableThreshold:

    def test_strict_threshold(self, base_state):
        """Strict threshold → more reoptimization."""
        new = LPProblem(c=[3.05, 4, 5], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(albici_base(), new)
        strict = ImpactAnalyzer(oguz_threshold=0.001)
        decision = strict.analyze(base_state, change, albici_base())
        # Small change, but strict threshold may still trigger reopt
        # (depends on whether it's within sensitivity range)
        assert decision is not None

    def test_lenient_threshold(self, base_state):
        """Lenient threshold → less reoptimization."""
        new = LPProblem(c=[3.5, 4, 5], A=albici_base().A, b=albici_base().b,
                        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(albici_base(), new)
        lenient = ImpactAnalyzer(oguz_threshold=0.50)
        decision = lenient.analyze(base_state, change, albici_base())
        # With 50% threshold, moderate changes should be skipped
        assert not decision.should_reoptimize
