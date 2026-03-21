"""Tests for LP file parser.

Covers: basic parsing, coefficients, constraints, bounds, MIP features,
error cases, round-trip (parse → solve), and edge cases.
"""

import numpy as np
import pytest
from pathlib import Path

from clara.io.lp_parser import LPParseError, parse_lp, read_lp
from clara.engine.simplex import solve

FIXTURES = Path(__file__).parent / "fixtures"
TOL = 1e-6


# ============================================================
# 1. Basic parsing
# ============================================================

class TestBasicParsing:
    """Core parsing of LP files."""

    def test_parse_albici_base(self):
        p = read_lp(FIXTURES / "albici_base.lp")
        assert p.num_variables == 3
        assert p.num_constraints == 4
        assert p.var_names == ["x1", "x2", "x3"]
        np.testing.assert_array_almost_equal(p.c, [3, 4, 5])

    def test_parse_albici_rhs(self):
        p = read_lp(FIXTURES / "albici_base.lp")
        np.testing.assert_array_almost_equal(p.b, [1200, 1400, 2000, 800])

    def test_parse_constraint_names(self):
        p = read_lp(FIXTURES / "albici_base.lp")
        assert "S1" in p.constraint_names
        assert "S2" in p.constraint_names

    def test_parse_maximize_sense(self):
        p = parse_lp("Maximize\n obj: 3 x1\nSubject To\n c1: x1 <= 10\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [3])

    def test_parse_minimize_sense(self):
        """Minimize should negate the objective (LPProblem is max-form)."""
        p = parse_lp("Minimize\n obj: 3 x1\nSubject To\n c1: x1 <= 10\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [-3])

    def test_parse_minimal(self):
        p = read_lp(FIXTURES / "parser_minimal.lp")
        assert p.num_variables == 2

    def test_parse_no_bounds_section(self):
        """No bounds section → defaults to 0 <= x <= +inf."""
        p = parse_lp("Max\n x1\nST\n x1 <= 5\nEnd\n")
        assert p.num_variables == 1

    def test_parse_problem_name(self):
        p = read_lp(FIXTURES / "albici_base.lp")
        assert p.name == "albici_base"


# ============================================================
# 2. Objective parsing
# ============================================================

class TestObjective:
    """Objective function parsing."""

    def test_named_objective(self):
        p = parse_lp("Max\n myobj: 2 x + 3 y\nST\n x <= 5\nEnd\n")
        assert p.num_variables == 2

    def test_unnamed_objective(self):
        p = parse_lp("Max\n 2 x + 3 y\nST\n x <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [2, 3])

    def test_single_var_obj(self):
        p = parse_lp("Max\n x1\nST\n x1 <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [1])

    def test_negative_coeff_obj(self):
        p = parse_lp("Max\n - x1 + 2 x2\nST\n x1 <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [-1, 2])


# ============================================================
# 3. Coefficients & expressions
# ============================================================

class TestCoefficients:
    """Coefficient parsing edge cases."""

    def test_implicit_one(self):
        p = parse_lp("Max\n x1 + x2\nST\n x1 <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [1, 1])

    def test_implicit_negative_one(self):
        p = parse_lp("Max\n - x1\nST\n x1 <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [-1])

    def test_adjacent_coeff(self):
        """'3x1' without space."""
        p = parse_lp("Max\n 3x1 + 4x2\nST\n x1 <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [3, 4])

    def test_float_coeff(self):
        p = parse_lp("Max\n 2.5 x1\nST\n x1 <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [2.5])

    def test_negative_float_coeff(self):
        p = parse_lp("Max\n - 3.5 x1\nST\n x1 <= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.c, [-3.5])


# ============================================================
# 4. Constraints
# ============================================================

class TestConstraints:
    """Constraint parsing."""

    def test_leq_constraint(self):
        p = parse_lp("Max\n x1\nST\n c1: x1 <= 10\nEnd\n")
        np.testing.assert_array_almost_equal(p.A, [[1]])
        np.testing.assert_array_almost_equal(p.b, [10])

    def test_geq_constraint(self):
        """'>=' is negated to '<='."""
        p = parse_lp("Max\n x1\nST\n c1: x1 >= 5\nEnd\n")
        np.testing.assert_array_almost_equal(p.A, [[-1]])
        np.testing.assert_array_almost_equal(p.b, [-5])

    def test_eq_constraint(self):
        """'=' is split into <= and >= (two rows)."""
        p = parse_lp("Max\n x1\nST\n c1: x1 = 5\nEnd\n")
        assert p.num_constraints == 2
        # x1 <= 5 and -x1 <= -5
        np.testing.assert_array_almost_equal(p.A[0], [1])
        np.testing.assert_array_almost_equal(p.b[0], 5)
        np.testing.assert_array_almost_equal(p.A[1], [-1])
        np.testing.assert_array_almost_equal(p.b[1], -5)

    def test_unnamed_constraint(self):
        p = parse_lp("Max\n x1\nST\n x1 <= 10\nEnd\n")
        assert "c1" in p.constraint_names

    def test_multiple_constraints(self):
        lp = """\
Max
 x1 + x2
ST
 c1: x1 + x2 <= 10
 c2: x1 - x2 <= 5
End
"""
        p = parse_lp(lp)
        assert p.num_constraints == 2


# ============================================================
# 5. Bounds
# ============================================================

class TestBounds:
    """Bounds section parsing."""

    def test_double_bound(self):
        lp = "Max\n x1\nST\n x1 <= 100\nBounds\n 0 <= x1 <= 50\nEnd\n"
        p = parse_lp(lp)
        # Parser recognizes bounds but LPProblem doesn't store them yet
        # Just verify it parses without error
        assert p.num_variables == 1

    def test_free_variable(self):
        lp = "Max\n x1\nST\n x1 <= 100\nBounds\n x1 free\nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 1

    def test_fixed_variable(self):
        lp = "Max\n x1\nST\n x1 <= 100\nBounds\n x1 = 25\nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 1

    def test_upper_only(self):
        lp = "Max\n x1\nST\n x1 <= 100\nBounds\n x1 <= 50\nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 1

    def test_lower_only(self):
        lp = "Max\n x1\nST\n x1 <= 100\nBounds\n x1 >= 10\nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 1


# ============================================================
# 6. MIP features
# ============================================================

class TestMIPFeatures:
    """Integer and binary variable declarations."""

    def test_comprehensive_fixture(self):
        """Parse the comprehensive fixture with General and Binary sections."""
        p = read_lp(FIXTURES / "parser_comprehensive.lp")
        assert p.num_variables == 6


# ============================================================
# 7. Error cases
# ============================================================

class TestErrors:
    """Parser error handling."""

    def test_error_empty_file(self):
        with pytest.raises(LPParseError, match="Empty"):
            parse_lp("")

    def test_error_missing_objective(self):
        with pytest.raises(LPParseError, match="MINIMIZE or MAXIMIZE"):
            parse_lp("Subject To\n x1 <= 10\nEnd\n")

    def test_error_invalid_var_name(self):
        """Variable name starting with period should be rejected by _register_var."""
        # Directly test the parser's variable registration
        from clara.io.lp_parser import _LPParser
        parser = _LPParser("dummy", "test")
        with pytest.raises(LPParseError, match="cannot start with"):
            parser._register_var(".abc")

    def test_error_duplicate_constraint_name(self):
        with pytest.raises(LPParseError, match="Duplicate constraint"):
            parse_lp("Max\n x1\nST\n c1: x1 <= 5\n c1: x1 <= 10\nEnd\n")


# ============================================================
# 8. Round-trip: parse → solve
# ============================================================

class TestRoundTrip:
    """Parse LP file → solve with Revised Simplex → verify optimal value."""

    def test_albici_base_roundtrip(self):
        """Parse albici_base.lp → solve → optimal = 33200/9."""
        p = read_lp(FIXTURES / "albici_base.lp")
        state = solve(p)
        expected = 33200 / 9
        assert abs(state.optimal_value - expected) < TOL, (
            f"Expected {expected}, got {state.optimal_value}"
        )

    def test_minimal_roundtrip(self):
        """Parse a minimize problem → solve.

        Note: current simplex doesn't support Phase I (negative RHS),
        so we use a problem with non-negative RHS after conversion.
        """
        # Minimize 2x1 + 3x2 s.t. x1 + x2 <= 10, x1 <= 6
        # → max -2x1 - 3x2, optimal at x1=0, x2=0, value=0
        p = parse_lp(
            "Minimize\n 2 x1 + 3 x2\nST\n x1 + x2 <= 10\n x1 <= 6\nEnd\n"
        )
        state = solve(p)
        assert abs(state.optimal_value - 0.0) < TOL

    def test_all_fixtures_parse(self):
        """All .lp files in fixtures/ should parse without error."""
        lp_files = list(FIXTURES.glob("*.lp"))
        assert len(lp_files) >= 2, "Expected at least 2 .lp fixtures"
        for lp_file in lp_files:
            p = read_lp(lp_file)
            assert p.num_variables > 0, f"{lp_file.name} parsed with 0 variables"

    def test_parse_then_solve_simple(self):
        """Inline LP → parse → solve."""
        lp = """\
Maximize
 obj: 5 x1 + 4 x2
Subject To
 c1: x1 + x2 <= 5
 c2: 10 x1 + 6 x2 <= 45
End
"""
        p = parse_lp(lp, name="simple_test")
        state = solve(p)
        assert abs(state.optimal_value - 23.75) < TOL


# ============================================================
# 9. Edge cases
# ============================================================

class TestEdgeCases:
    """Parser edge cases."""

    def test_comments_ignored(self):
        lp = "\\ comment line\nMax\n x1\nST\n x1 <= 5\nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 1

    def test_inline_comments(self):
        lp = "Max\n x1 \\ this is x1\nST\n x1 <= 5 \\ upper bound\nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 1

    def test_extra_whitespace(self):
        lp = "  Maximize  \n   x1  +   x2  \nSubject To\n  x1  <=  5  \nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 2

    def test_case_insensitive_keywords(self):
        for kw in ["maximize", "MAXIMIZE", "Maximize", "MAX", "max"]:
            p = parse_lp(f"{kw}\n x1\nST\n x1 <= 5\nEnd\n")
            assert p.num_variables == 1

    def test_empty_lines_between(self):
        lp = "Max\n\n x1\n\nST\n\n x1 <= 5\n\nEnd\n"
        p = parse_lp(lp)
        assert p.num_variables == 1

    def test_no_end_keyword(self):
        """Should parse even without END."""
        p = parse_lp("Max\n x1\nST\n x1 <= 5\n")
        assert p.num_variables == 1

    def test_st_variants(self):
        """All constraint section keywords should work."""
        for kw in ["Subject To", "Such That", "ST", "S.T.", "ST."]:
            p = parse_lp(f"Max\n x1\n{kw}\n x1 <= 5\nEnd\n")
            assert p.num_variables == 1
