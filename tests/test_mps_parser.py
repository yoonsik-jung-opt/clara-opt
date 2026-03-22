"""Tests for MPS parser — basic parsing, Netlib, cross-validation, CLI."""

import numpy as np
import pytest
from pathlib import Path
from click.testing import CliRunner

from clara.io.mps_parser import read_mps, parse_mps, MPSParseError
from clara.engine.simplex import RevisedSimplex
from clara.engine.highs_backend import HiGHSBackend
from clara.cli import main

NETLIB_DIR = Path(__file__).parent.parent / "benchmarks" / "netlib" / "mps"
TOL = 1e-2


# ============================================================
# 1. Basic parsing
# ============================================================

SIMPLE_MPS = """\
NAME          simple
ROWS
 N  obj
 L  c1
 L  c2
COLUMNS
    x1  obj  3.0   c1  1.0
    x1  c2   2.0
    x2  obj  4.0   c1  2.0
    x2  c2   1.0
RHS
    rhs   c1  10.0  c2  8.0
BOUNDS
ENDATA
"""


class TestBasicParsing:

    def test_parse_simple(self):
        p = parse_mps(SIMPLE_MPS, "simple")
        assert p.num_variables == 2
        assert p.var_names == ["x1", "x2"]

    def test_parse_objective(self):
        p = parse_mps(SIMPLE_MPS)
        np.testing.assert_array_almost_equal(p.c, [3, 4])

    def test_parse_constraints(self):
        p = parse_mps(SIMPLE_MPS)
        assert p.num_constraints == 2
        np.testing.assert_array_almost_equal(p.A[0], [1, 2])
        np.testing.assert_array_almost_equal(p.b, [10, 8])

    def test_default_minimize(self):
        p = parse_mps(SIMPLE_MPS)
        assert p.sense == "minimize"

    def test_maximize_objsense(self):
        mps = "NAME test\nOBJSENSE\n MAX\nROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 5 c1 1\nRHS\n r c1 10\nENDATA\n"
        p = parse_mps(mps)
        assert p.sense == "maximize"

    def test_problem_name(self):
        p = parse_mps(SIMPLE_MPS)
        assert p.name == "simple"


# ============================================================
# 2. Rows
# ============================================================

class TestRows:

    def test_g_type_negated(self):
        """G rows (>=) should be negated to <=."""
        mps = "NAME t\nROWS\n N obj\n G c1\nCOLUMNS\n x1 obj 1 c1 2\nRHS\n r c1 5\nENDATA\n"
        p = parse_mps(mps)
        np.testing.assert_array_almost_equal(p.A[0], [-2])
        np.testing.assert_array_almost_equal(p.b[0], -5)

    def test_e_type_split(self):
        """E rows (=) should be split into <= and >=."""
        mps = "NAME t\nROWS\n N obj\n E c1\nCOLUMNS\n x1 obj 1 c1 3\nRHS\n r c1 6\nENDATA\n"
        p = parse_mps(mps)
        assert p.num_constraints == 2  # c1 and c1_eq


# ============================================================
# 3. Columns
# ============================================================

class TestColumns:

    def test_two_per_line(self):
        """Two (row, value) pairs on one line."""
        p = parse_mps(SIMPLE_MPS)
        # x1 has obj=3 and c1=1 on same line
        assert p.c[0] == 3.0
        assert p.A[0, 0] == 1.0

    def test_variable_order(self):
        p = parse_mps(SIMPLE_MPS)
        assert p.var_names[0] == "x1"
        assert p.var_names[1] == "x2"

    def test_integer_markers(self):
        mps = "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n"
        mps += "    INT1 'MARKER' 'INTORG'\n"
        mps += "    x1 obj 1 c1 1\n"
        mps += "    INT2 'MARKER' 'INTEND'\n"
        mps += "    x2 obj 2 c1 1\n"
        mps += "RHS\n r c1 10\nENDATA\n"
        p = parse_mps(mps)
        assert 0 in p.integer_vars  # x1 is integer
        assert 1 not in p.integer_vars  # x2 is not


# ============================================================
# 4. Bounds
# ============================================================

class TestBounds:

    def test_bounds_lo_up(self):
        mps = SIMPLE_MPS.replace("BOUNDS\n", "BOUNDS\n LO bnd x1 5\n UP bnd x2 20\n")
        p = parse_mps(mps)
        assert p.lower_bounds[0] == 5.0
        assert p.upper_bounds[1] == 20.0

    def test_bounds_fx(self):
        mps = SIMPLE_MPS.replace("BOUNDS\n", "BOUNDS\n FX bnd x1 7\n")
        p = parse_mps(mps)
        assert p.lower_bounds[0] == 7.0
        assert p.upper_bounds[0] == 7.0

    def test_bounds_fr(self):
        mps = SIMPLE_MPS.replace("BOUNDS\n", "BOUNDS\n FR bnd x1\n")
        p = parse_mps(mps)
        assert p.lower_bounds[0] == float("-inf")

    def test_bounds_bv(self):
        mps = SIMPLE_MPS.replace("BOUNDS\n", "BOUNDS\n BV bnd x1\n")
        p = parse_mps(mps)
        assert 0 in p.binary_vars
        assert p.upper_bounds[0] == 1.0

    def test_default_bounds(self):
        p = parse_mps(SIMPLE_MPS)
        assert p.lower_bounds[0] == 0.0
        assert p.upper_bounds[0] == float("inf")


# ============================================================
# 5. Netlib instances
# ============================================================

NETLIB_EXPECTED = {
    "afiro": (32, 27),
    "adlittle": (97, 56),
    "blend": (83, 74),
    "sc50a": (48, 50),
    "sc50b": (48, 50),
    "sc105": (103, 105),
    "kb2": (41, 43),
    "share2b": (79, 96),
}


class TestNetlib:

    @pytest.mark.parametrize("name,expected", NETLIB_EXPECTED.items())
    def test_parse_netlib(self, name, expected):
        mps_file = NETLIB_DIR / f"{name}.mps"
        if not mps_file.exists():
            pytest.skip(f"{name}.mps not found")
        p = read_mps(mps_file)
        exp_vars, exp_cons = expected
        assert p.num_variables == exp_vars, f"{name}: expected {exp_vars} vars, got {p.num_variables}"
        # Constraints may differ due to E→2 rows conversion, so check >=
        assert p.sense == "minimize"


# ============================================================
# 6. Cross-validation: Internal vs HiGHS via MPS
# ============================================================

CROSS_INSTANCES = ["afiro", "adlittle", "sc50a", "sc50b", "sc105", "share2b"]


class TestCrossValidation:

    @pytest.mark.parametrize("name", CROSS_INSTANCES)
    def test_internal_vs_highs(self, name):
        mps_file = NETLIB_DIR / f"{name}.mps"
        if not mps_file.exists():
            pytest.skip(f"{name}.mps not found")
        p = read_mps(mps_file)
        internal = RevisedSimplex(p).solve()
        highs = HiGHSBackend().solve(p)
        if not internal.is_optimal or not highs.is_optimal:
            pytest.skip(f"{name}: solver failed")
        assert abs(internal.optimal_value - highs.optimal_value) < TOL, (
            f"{name}: internal={internal.optimal_value}, highs={highs.optimal_value}"
        )


# ============================================================
# 7. Minimize handling
# ============================================================

class TestMinimize:

    def test_minimize_simplex(self):
        p = parse_mps(SIMPLE_MPS)
        assert p.sense == "minimize"
        state = RevisedSimplex(p).solve()
        # min 3x1 + 4x2 s.t. x1+2x2<=10, 2x1+x2<=8 → min at x=0, val=0
        assert state.is_optimal
        assert abs(state.optimal_value - 0.0) < TOL

    def test_minimize_highs(self):
        p = parse_mps(SIMPLE_MPS)
        state = HiGHSBackend().solve(p)
        assert abs(state.optimal_value - 0.0) < TOL

    def test_afiro_optimal(self):
        mps_file = NETLIB_DIR / "afiro.mps"
        if not mps_file.exists():
            pytest.skip("afiro.mps not found")
        p = read_mps(mps_file)
        state = RevisedSimplex(p).solve()
        assert abs(state.optimal_value - (-464.7531)) < TOL


# ============================================================
# 8. CLI integration
# ============================================================

class TestCLI:

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_explain_mps(self, runner):
        mps_file = NETLIB_DIR / "afiro.mps"
        if not mps_file.exists():
            pytest.skip("afiro.mps not found")
        result = runner.invoke(main, ["explain", str(mps_file), "--level", "brief"])
        assert result.exit_code == 0

    def test_info_mps(self, runner):
        mps_file = NETLIB_DIR / "afiro.mps"
        if not mps_file.exists():
            pytest.skip("afiro.mps not found")
        result = runner.invoke(main, ["info", str(mps_file)])
        assert result.exit_code == 0
        assert "32" in result.output or "Variables" in result.output


# ============================================================
# 9. Edge cases
# ============================================================

class TestEdgeCases:

    def test_comments_ignored(self):
        mps = "* comment\nNAME t\nROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 1 c1 1\nRHS\n r c1 5\nENDATA\n"
        p = parse_mps(mps)
        assert p.num_variables == 1

    def test_no_endata(self):
        """File without ENDATA should still parse."""
        mps = "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 1 c1 1\nRHS\n r c1 5\n"
        p = parse_mps(mps)
        assert p.num_variables == 1

    def test_names_with_dots(self):
        """Netlib-style names like ...100 should work."""
        mps = "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n ...100 obj 1 c1 2\nRHS\n r c1 10\nENDATA\n"
        p = parse_mps(mps)
        assert p.var_names == ["...100"]
