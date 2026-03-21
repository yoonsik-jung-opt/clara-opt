"""Tests for CLARA CLI using click.testing.CliRunner.

Covers: end-to-end explain/solve/info, options, error handling, formats.
"""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner
from pathlib import Path

from clara.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
ALBICI = str(FIXTURES / "albici_base.lp")


@pytest.fixture
def runner():
    return CliRunner()


# ============================================================
# 1. End-to-end: explain
# ============================================================

class TestExplainCommand:

    def test_explain_albici_base(self, runner):
        result = runner.invoke(main, ["explain", ALBICI])
        assert result.exit_code == 0, result.output
        assert "3688.8889" in result.output

    def test_explain_brief(self, runner):
        result = runner.invoke(main, ["explain", ALBICI, "--level", "brief"])
        assert result.exit_code == 0

    def test_explain_detailed(self, runner):
        result = runner.invoke(main, ["explain", ALBICI, "--level", "detailed"])
        assert result.exit_code == 0
        assert "Shadow price" in result.output

    def test_explain_brief_shorter_than_detailed(self, runner):
        brief = runner.invoke(main, ["explain", ALBICI, "--level", "brief"])
        detailed = runner.invoke(main, ["explain", ALBICI, "--level", "detailed"])
        assert len(brief.output) < len(detailed.output)

    def test_explain_json(self, runner):
        result = runner.invoke(main, ["explain", ALBICI, "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "binding_report" in parsed
        assert "variable_report" in parsed
        assert "sensitivity_report" in parsed

    def test_explain_quiet(self, runner):
        normal = runner.invoke(main, ["explain", ALBICI])
        quiet = runner.invoke(main, ["explain", ALBICI, "--quiet"])
        assert quiet.exit_code == 0
        assert len(quiet.output) < len(normal.output)
        # Header contains CLARA — should be stripped
        assert "CLARA" not in quiet.output

    def test_explain_output_file(self, runner):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            result = runner.invoke(main, ["explain", ALBICI, "-o", path])
            assert result.exit_code == 0
            content = Path(path).read_text()
            assert "3688.8889" in content
        finally:
            os.unlink(path)

    def test_explain_all_fixtures(self, runner):
        # Only test fixtures with pure <= constraints (simplex lacks Phase I
        # for negative RHS from >= negation)
        skip = {"parser_minimal.lp", "parser_comprehensive.lp"}
        for lp_file in FIXTURES.glob("*.lp"):
            if lp_file.name in skip:
                continue
            result = runner.invoke(main, ["explain", str(lp_file)])
            assert result.exit_code == 0, f"{lp_file.name} failed: {result.output}"

    def test_explain_contains_binding(self, runner):
        result = runner.invoke(main, ["explain", ALBICI])
        assert "BINDING CONSTRAINTS" in result.output

    def test_explain_contains_variable(self, runner):
        result = runner.invoke(main, ["explain", ALBICI])
        assert "VARIABLE STATUS" in result.output

    def test_explain_contains_sensitivity(self, runner):
        result = runner.invoke(main, ["explain", ALBICI])
        assert "SENSITIVITY ANALYSIS" in result.output


# ============================================================
# 2. Solve command
# ============================================================

class TestSolveCommand:

    def test_solve_albici_base(self, runner):
        result = runner.invoke(main, ["solve", ALBICI])
        assert result.exit_code == 0
        assert "3688.8889" in result.output
        assert "x1" in result.output

    def test_solve_json(self, runner):
        result = runner.invoke(main, ["solve", ALBICI, "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "optimal_value" in parsed
        assert "variables" in parsed


# ============================================================
# 3. Info command
# ============================================================

class TestInfoCommand:

    def test_info_albici_base(self, runner):
        result = runner.invoke(main, ["info", ALBICI])
        assert result.exit_code == 0
        assert "3" in result.output  # 3 variables
        assert "4" in result.output  # 4 constraints

    def test_info_shows_problem_name(self, runner):
        result = runner.invoke(main, ["info", ALBICI])
        assert "albici_base" in result.output

    def test_info_shows_nonzeros(self, runner):
        result = runner.invoke(main, ["info", ALBICI])
        assert "Nonzeros" in result.output


# ============================================================
# 4. Engine option
# ============================================================

class TestEngineOption:

    def test_engine_internal(self, runner):
        result = runner.invoke(main, ["explain", ALBICI, "--engine", "internal"])
        assert result.exit_code == 0

    def test_engine_highs_fallback(self, runner):
        result = runner.invoke(main, ["explain", ALBICI, "--engine", "highs"])
        # Should fallback to internal with warning
        assert result.exit_code == 0


# ============================================================
# 5. Error handling
# ============================================================

class TestErrorHandling:

    def test_file_not_found(self, runner):
        result = runner.invoke(main, ["explain", "nonexistent.lp"])
        assert result.exit_code != 0

    def test_parse_error(self, runner):
        with tempfile.NamedTemporaryFile(suffix=".lp", mode="w", delete=False) as f:
            f.write("this is not valid LP\n")
            path = f.name
        try:
            result = runner.invoke(main, ["explain", path])
            assert result.exit_code != 0
        finally:
            os.unlink(path)


# ============================================================
# 6. Version and help
# ============================================================

class TestVersionHelp:

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "CLARA" in result.output

    def test_explain_help(self, runner):
        result = runner.invoke(main, ["explain", "--help"])
        assert result.exit_code == 0
        assert "--engine" in result.output
        assert "--level" in result.output
        assert "--format" in result.output


# ============================================================
# 7. Placeholder commands
# ============================================================

class TestPlaceholders:

    def test_what_if(self, runner):
        result = runner.invoke(main, ["what-if", ALBICI])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output

    def test_watch(self, runner):
        result = runner.invoke(main, ["watch", ALBICI, "--params", ALBICI])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output
