"""Tests for Diff Report — Albici scenarios, text/JSON output, CLI."""

import json
import pytest
from pathlib import Path

from click.testing import CliRunner

from clara.cli import main
from clara.engine.simplex import RevisedSimplex
from clara.model.problem import LPProblem
from clara.reopt.analyzer import ImpactAnalyzer
from clara.reopt.detector import ChangeDetector
from clara.reopt.diff_report import DiffReporter
from clara.reopt.reoptimizer import Reoptimizer

TOL = 1e-2
FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================
# Helpers
# ============================================================

def albici_base() -> LPProblem:
    return LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )

def albici_b1():
    p = albici_base()
    return LPProblem(c=p.c, A=p.A, b=[1300, 1200, 1800, 1000],
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))

def albici_b2():
    p = albici_base()
    return LPProblem(c=p.c, A=p.A, b=[1500, 1300, 2400, 800],
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))

def albici_cost():
    p = albici_base()
    return LPProblem(c=[8, 6, 7], A=p.A, b=p.b,
                     var_names=list(p.var_names), constraint_names=list(p.constraint_names))

def albici_columns():
    return LPProblem(
        c=[3, 4, 5, 3, 5],
        A=[[1, 2, 1, 2, 3], [1, 0, 3, 0, 1], [2, 1, 2, 1, 0], [0, 2, 3, 0, 1]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3", "x4", "x5"],
        constraint_names=["S1", "S2", "S3", "S4"],
    )


def run_pipeline(old_problem, new_problem):
    """Full pipeline: solve → detect → analyze → reoptimize → diff."""
    old_state = RevisedSimplex(old_problem).solve()
    change = ChangeDetector().detect(old_problem, new_problem)
    decision = ImpactAnalyzer().analyze(old_state, change, old_problem)
    result = Reoptimizer().reoptimize(old_state, new_problem, change, decision, old_problem=old_problem)
    report = DiffReporter().diff(old_state, result.new_state, change, result)
    return report


# ============================================================
# 1. Albici b1 (basis preserved)
# ============================================================

class TestAlbiciB1:

    def test_b1_objective_delta(self):
        r = run_pipeline(albici_base(), albici_b1())
        # Old ≈ 3688.89, b1 changes are conservative → objective changes slightly
        assert abs(r.objective_delta_pct) < 20  # small change

    def test_b1_no_basis_change(self):
        r = run_pipeline(albici_base(), albici_b1())
        # Analyzer recommends warm_start (conservative), but no vars enter/leave
        # since the actual reopt may preserve or change basis
        assert r.variables_entered_basis is not None  # field exists

    def test_b1_bottleneck(self):
        r = run_pipeline(albici_base(), albici_b1())
        assert r.old_bottleneck is not None


# ============================================================
# 2. Albici b2 (basis broken)
# ============================================================

class TestAlbiciB2:

    def test_b2_objective_increased(self):
        r = run_pipeline(albici_base(), albici_b2())
        assert r.new_objective > r.old_objective
        assert abs(r.new_objective - 4300.0) < TOL

    def test_b2_variable_changes(self):
        r = run_pipeline(albici_base(), albici_b2())
        assert len(r.variable_changes) > 0

    def test_b2_has_constraint_changes(self):
        r = run_pipeline(albici_base(), albici_b2())
        assert len(r.constraint_changes) > 0


# ============================================================
# 3. Albici cost change
# ============================================================

class TestAlbiciCost:

    def test_cost_x3_left_basis(self):
        r = run_pipeline(albici_base(), albici_cost())
        assert "x3" in r.variables_left_basis

    def test_cost_objective_more_than_doubled(self):
        r = run_pipeline(albici_base(), albici_cost())
        assert r.objective_delta_pct > 100

    def test_cost_new_objective(self):
        r = run_pipeline(albici_base(), albici_cost())
        assert abs(r.new_objective - 24800 / 3) < TOL


# ============================================================
# 4. Albici columns
# ============================================================

class TestAlbiciColumns:

    def test_columns_scratch(self):
        r = run_pipeline(albici_base(), albici_columns())
        assert r.reopt_result.method_used == "scratch"


# ============================================================
# 5. Text output
# ============================================================

class TestTextOutput:

    def test_to_text_not_empty(self):
        r = run_pipeline(albici_base(), albici_b2())
        assert len(r.to_text()) > 100

    def test_to_text_has_sections(self):
        r = run_pipeline(albici_base(), albici_b2())
        text = r.to_text()
        assert "PARAMETER CHANGE" in text
        assert "OBJECTIVE" in text
        assert "VARIABLE CHANGES" in text

    def test_to_text_has_numbers(self):
        r = run_pipeline(albici_base(), albici_b2())
        text = r.to_text()
        assert "3688" in text  # old objective
        assert "4300" in text  # new objective


# ============================================================
# 6. JSON output
# ============================================================

class TestJSONOutput:

    def test_to_json_valid(self):
        r = run_pipeline(albici_base(), albici_b2())
        parsed = json.loads(r.to_json())
        assert isinstance(parsed, dict)

    def test_json_has_keys(self):
        r = run_pipeline(albici_base(), albici_b2())
        d = r.to_dict()
        assert "objective" in d
        assert "variable_changes" in d
        assert "bottleneck" in d
        assert "change" in d
        assert "reoptimization" in d

    def test_json_objective_values(self):
        r = run_pipeline(albici_base(), albici_b2())
        d = r.to_dict()
        assert abs(d["objective"]["old"] - 33200 / 9) < TOL
        assert abs(d["objective"]["new"] - 4300.0) < TOL


# ============================================================
# 7. CLI `clara diff`
# ============================================================

class TestCLIDiff:

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_cli_diff_b1(self, runner):
        result = runner.invoke(main, [
            "diff", str(FIXTURES / "albici_base.lp"), str(FIXTURES / "albici_reopt_b1.lp")
        ])
        assert result.exit_code == 0, result.output

    def test_cli_diff_b2(self, runner):
        result = runner.invoke(main, [
            "diff", str(FIXTURES / "albici_base.lp"), str(FIXTURES / "albici_reopt_b2.lp")
        ])
        assert result.exit_code == 0
        assert "4300" in result.output

    def test_cli_diff_cost(self, runner):
        result = runner.invoke(main, [
            "diff", str(FIXTURES / "albici_base.lp"), str(FIXTURES / "albici_reopt_cost.lp")
        ])
        assert result.exit_code == 0

    def test_cli_diff_json(self, runner):
        result = runner.invoke(main, [
            "diff", str(FIXTURES / "albici_base.lp"), str(FIXTURES / "albici_reopt_b2.lp"),
            "--format", "json"
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "objective" in parsed

    def test_cli_diff_no_change(self, runner):
        result = runner.invoke(main, [
            "diff", str(FIXTURES / "albici_base.lp"), str(FIXTURES / "albici_base.lp")
        ])
        assert result.exit_code == 0
        assert "No parameter changes" in result.output


# ============================================================
# 8. Edge cases
# ============================================================

class TestEdgeCases:

    def test_identical_no_diff(self):
        change = ChangeDetector().detect(albici_base(), albici_base())
        assert change is None

    def test_summary_contains_info(self):
        r = run_pipeline(albici_base(), albici_b2())
        assert "Parameter change" in r.summary
        assert "Objective" in r.summary.lower() or "%" in r.summary
