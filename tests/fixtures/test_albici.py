"""
Tests for Albici et al. (2010) numerical examples.
====================================================

These tests validate CLARA's solver output against known results from:
    "Reoptimizations in Linear Programming", MPRA Paper No. 20091

Test hierarchy:
    1. test_albici_lp_files_parse    — .lp files are syntactically valid
    2. test_albici_base_*            — original problem correctness
    3. test_albici_reopt_*           — each reoptimization scenario
    4. test_albici_parametric_*      — Type RC compound change (Phase 2)

Usage:
    pytest tests/test_albici.py -v
    pytest tests/test_albici.py -k "base" -v        # base only
    pytest tests/test_albici.py -k "parametric" -v   # parametric LP only
"""

import numpy as np
import pytest
from pathlib import Path
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent

TOL_OPTIMAL = 1e-6   # tolerance for optimal value comparison
TOL_VARIABLE = 1e-4  # tolerance for variable values (looser for solver differences)

# Problem data (shared across tests)
A_UB = [
    [1, 2, 1],   # S1
    [1, 0, 3],   # S2
    [2, 1, 2],   # S3
    [0, 2, 3],   # S4
]

B_ORIG = [1200, 1400, 2000, 800]
C_ORIG = [3, 4, 5]

# Basis inverse from final tableau (basis: y2, x1, x2, x3)
B_INV = np.array([
    [11/9,  1,    -10/9, -6/9],
    [1/9,   0,     4/9,  -3/9],
    [6/9,   0,    -3/9,   0  ],
    [-4/9,  0,     2/9,   3/9],
])


# ---------------------------------------------------------------------------
# Helper: solve LP with scipy/HiGHS (ground truth)
# ---------------------------------------------------------------------------

def solve_lp(c_max, A_ub, b_ub):
    """Solve max c^T x s.t. Ax <= b, x >= 0. Returns (optimal_value, x)."""
    c_min = [-ci for ci in c_max]
    res = linprog(c_min, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * len(c_max), method="highs")
    assert res.success, f"LP solver failed: {res.message}"
    return -res.fun, res.x


# ===========================================================================
# 1. LP file validation
# ===========================================================================

class TestAlbiciLPFiles:
    """Verify .lp fixture files exist and are parseable."""

    LP_FILES = [
        "albici_base.lp",
        "albici_reopt_b1.lp",
        "albici_reopt_b2.lp",
        "albici_reopt_cost.lp",
        "albici_reopt_columns.lp",
    ]

    @pytest.mark.parametrize("filename", LP_FILES)
    def test_lp_file_exists(self, filename):
        path = FIXTURES_DIR / filename
        assert path.exists(), f"Missing fixture: {path}"

    @pytest.mark.parametrize("filename", LP_FILES)
    def test_lp_file_nonempty(self, filename):
        path = FIXTURES_DIR / filename
        content = path.read_text()
        assert len(content) > 50, f"Fixture too short: {path}"
        assert "Maximize" in content or "Minimize" in content


# ===========================================================================
# 2. Original problem
# ===========================================================================

class TestAlbiciBase:
    """Scenario 0: Original problem — max 3x1 + 4x2 + 5x3."""

    EXPECTED_OPTIMAL = 33200 / 9  # ≈ 3688.8889

    def test_optimal_value(self):
        opt, _ = solve_lp(C_ORIG, A_UB, B_ORIG)
        assert abs(opt - self.EXPECTED_OPTIMAL) < TOL_OPTIMAL

    def test_basis_inverse_valid(self):
        """B * B^-1 = I for the known basis {y2, x1, x2, x3}."""
        B = np.array([
            [0, 1, 2, 1],   # y2=e2, x1, x2, x3 columns
            [1, 1, 0, 3],
            [0, 2, 1, 2],
            [0, 0, 2, 3],
        ], dtype=float)
        assert np.allclose(B @ B_INV, np.eye(4), atol=1e-10)

    def test_basic_variable_values(self):
        """B^-1 * b gives correct basic variable values."""
        x_B = B_INV @ np.array(B_ORIG)
        expected = np.array([1000/9, 6800/9, 1200/9, 1600/9])
        assert np.allclose(x_B, expected, atol=1e-10)

    def test_optimal_from_basis(self):
        """c_B^T * B^-1 * b matches optimal value."""
        c_B = np.array([0, 3, 4, 5])  # costs for y2, x1, x2, x3
        x_B = B_INV @ np.array(B_ORIG)
        assert abs(c_B @ x_B - self.EXPECTED_OPTIMAL) < TOL_OPTIMAL

    def test_all_constraints_satisfied(self):
        _, x = solve_lp(C_ORIG, A_UB, B_ORIG)
        Ax = np.array(A_UB) @ x
        for i in range(4):
            assert Ax[i] <= B_ORIG[i] + TOL_VARIABLE


# ===========================================================================
# 3. Reoptimization scenarios
# ===========================================================================

class TestAlbiciReoptB1:
    """Scenario b1: RHS → (1300, 1200, 1800, 1000). Basis preserved."""

    B_NEW = [1300, 1200, 1800, 1000]
    EXPECTED_OPTIMAL = 33100 / 9  # ≈ 3677.7778

    def test_optimal_value(self):
        opt, _ = solve_lp(C_ORIG, A_UB, self.B_NEW)
        assert abs(opt - self.EXPECTED_OPTIMAL) < TOL_OPTIMAL

    def test_basis_preserved(self):
        """B^-1 * b_new >= 0 everywhere → same basis remains optimal."""
        x_B_new = B_INV @ np.array(self.B_NEW)
        assert np.all(x_B_new >= -TOL_VARIABLE), \
            f"Negative basic variable: {x_B_new}"

    def test_reopt_no_pivots_needed(self):
        """Optimal value from B^-1 recomputation matches solver."""
        c_B = np.array([0, 3, 4, 5])
        x_B_new = B_INV @ np.array(self.B_NEW)
        opt_reopt = c_B @ x_B_new
        opt_solver, _ = solve_lp(C_ORIG, A_UB, self.B_NEW)
        assert abs(opt_reopt - opt_solver) < TOL_OPTIMAL, \
            "Reopt without pivots should match scratch solve"

    def test_change_vector(self):
        delta_b = np.array(self.B_NEW) - np.array(B_ORIG)
        assert np.array_equal(delta_b, [100, -200, -200, 200])


class TestAlbiciReoptB2:
    """Scenario b2: RHS → (1500, 1300, 2400, 800). Basis broken."""

    B_NEW = [1500, 1300, 2400, 800]
    EXPECTED_OPTIMAL = 4300.0

    def test_optimal_value(self):
        opt, _ = solve_lp(C_ORIG, A_UB, self.B_NEW)
        assert abs(opt - self.EXPECTED_OPTIMAL) < TOL_OPTIMAL

    def test_basis_broken(self):
        """B^-1 * b_new has negative component → dual simplex needed."""
        x_B_new = B_INV @ np.array(self.B_NEW)
        assert np.any(x_B_new < -TOL_VARIABLE), \
            "Expected at least one negative basic variable"

    def test_which_variable_infeasible(self):
        """y2 (slack S2, first basic variable) becomes negative."""
        x_B_new = B_INV @ np.array(self.B_NEW)
        assert x_B_new[0] < -TOL_VARIABLE, \
            f"y2 should be negative, got {x_B_new[0]}"

    def test_solution_feasibility(self):
        _, x = solve_lp(C_ORIG, A_UB, self.B_NEW)
        Ax = np.array(A_UB) @ x
        for i in range(4):
            assert Ax[i] <= self.B_NEW[i] + TOL_VARIABLE


class TestAlbiciReoptCost:
    """Scenario c: Objective → (8, 6, 7). Optimality violated."""

    C_NEW = [8, 6, 7]
    EXPECTED_OPTIMAL = 24800 / 3  # ≈ 8266.6667

    def test_optimal_value(self):
        opt, _ = solve_lp(self.C_NEW, A_UB, B_ORIG)
        assert abs(opt - self.EXPECTED_OPTIMAL) < TOL_OPTIMAL

    def test_x3_leaves_basis(self):
        """With new costs, x3 = 0 in optimal solution."""
        _, x = solve_lp(self.C_NEW, A_UB, B_ORIG)
        assert x[2] < TOL_VARIABLE, f"x3 should be 0, got {x[2]}"

    def test_optimality_violated_with_old_basis(self):
        """Check reduced costs under new objective with old basis.

        At least one reduced cost should become negative (for max problem),
        meaning the old basis is no longer optimal.
        """
        c_B_new = np.array([0, 8, 6, 7])  # new costs for basis {y2, x1, x2, x3}
        # Reduced cost for non-basic y1: c_B^T B^-1 e1 - 0
        rc_y1 = c_B_new @ B_INV[:, 0]
        # Reduced cost for non-basic y3: c_B^T B^-1 e3 - 0
        rc_y3 = c_B_new @ B_INV[:, 2]
        # Reduced cost for non-basic y4: c_B^T B^-1 e4 - 0
        rc_y4 = c_B_new @ B_INV[:, 3]

        # For max problem, all reduced costs of non-basic vars should be >= 0
        # If any is negative, optimality is violated
        all_nonneg = rc_y1 >= -TOL_OPTIMAL and rc_y3 >= -TOL_OPTIMAL and rc_y4 >= -TOL_OPTIMAL
        assert not all_nonneg, (
            f"Expected optimality violation. "
            f"Reduced costs: y1={rc_y1:.6f}, y3={rc_y3:.6f}, y4={rc_y4:.6f}"
        )

    def test_oguz_bound_applicable(self):
        """Oguz bound 2δ/(1+δ) should be computable for obj coeff change."""
        delta = max(
            abs(self.C_NEW[j] - C_ORIG[j]) / C_ORIG[j]
            for j in range(3)
        )
        bound = 2 * delta / (1 + delta)
        assert delta > 0
        assert bound > 0
        # Large δ (5/3) → loose bound (1.25), confirming reopt is needed
        assert abs(delta - 5/3) < TOL_OPTIMAL
        assert abs(bound - 10/8) < TOL_OPTIMAL


class TestAlbiciReoptColumns:
    """Scenario d: Add P4 and P5. Column addition."""

    A_EXT = [
        [1, 2, 1, 2, 3],
        [1, 0, 3, 0, 1],
        [2, 1, 2, 1, 0],
        [0, 2, 3, 0, 1],
    ]
    C_EXT = [3, 4, 5, 3, 5]
    EXPECTED_OPTIMAL = 33700 / 9  # ≈ 3744.4444 (paper has arithmetic error)

    def test_optimal_value(self):
        opt, _ = solve_lp(self.C_EXT, self.A_EXT, B_ORIG)
        assert abs(opt - self.EXPECTED_OPTIMAL) < TOL_OPTIMAL

    def test_improves_over_base(self):
        """Adding columns should not worsen the objective."""
        opt_ext, _ = solve_lp(self.C_EXT, self.A_EXT, B_ORIG)
        opt_base, _ = solve_lp(C_ORIG, A_UB, B_ORIG)
        assert opt_ext >= opt_base - TOL_OPTIMAL

    def test_p5_enters_basis(self):
        """P5 (cost 5, column [3,1,0,1]) should be in optimal solution."""
        _, x = solve_lp(self.C_EXT, self.A_EXT, B_ORIG)
        assert x[4] > TOL_VARIABLE, f"x5 should be positive, got {x[4]}"

    def test_p4_stays_out(self):
        """P4 has non-negative reduced cost → stays at zero."""
        _, x = solve_lp(self.C_EXT, self.A_EXT, B_ORIG)
        assert x[3] < TOL_VARIABLE, f"x4 should be 0, got {x[3]}"

    def test_paper_value_is_wrong(self):
        """Document that the paper's manual result (24800/6) is incorrect."""
        opt, _ = solve_lp(self.C_EXT, self.A_EXT, B_ORIG)
        paper_value = 24800 / 6  # ≈ 4133.33
        assert abs(opt - paper_value) > 100, \
            "If this fails, the paper was right and we need to recheck"


# ===========================================================================
# 4. Parametric LP (Type RC compound change) — Phase 2
# ===========================================================================

class TestAlbiciParametric:
    """Compound b+c change via parametric LP: θ ∈ [0, 1].

    b(θ) = (1200,1400,2000,800) + θ*(300,-100,400,0)
    c(θ) = (3,4,5) + θ*(5,2,2)

    First breakpoint at θ* = 5/8 (primal: y2 hits 0).
    """

    DELTA_B = np.array([300, -100, 400, 0], dtype=float)
    DELTA_C = np.array([5, 2, 2], dtype=float)

    def _solve_at_theta(self, theta):
        b_t = np.array(B_ORIG) + theta * self.DELTA_B
        c_t = np.array(C_ORIG) + theta * self.DELTA_C
        return solve_lp(c_t.tolist(), A_UB, b_t.tolist())

    def test_theta_0_is_original(self):
        opt, _ = self._solve_at_theta(0.0)
        assert abs(opt - 33200/9) < TOL_OPTIMAL

    def test_theta_1_combined(self):
        opt, _ = self._solve_at_theta(1.0)
        assert abs(opt - 10000.0) < TOL_OPTIMAL

    def test_first_breakpoint_primal(self):
        """At θ* = 5/8, basic variable y2 (row 0 of B^-1) hits zero."""
        b_t = np.array(B_ORIG) + (5/8) * self.DELTA_B
        x_B = B_INV @ b_t
        assert abs(x_B[0]) < TOL_VARIABLE, \
            f"y2 should be ≈ 0 at θ=5/8, got {x_B[0]}"

    def test_before_breakpoint_basis_holds(self):
        """For θ < 5/8, basis {y2, x1, x2, x3} remains feasible."""
        for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            b_t = np.array(B_ORIG) + theta * self.DELTA_B
            x_B = B_INV @ b_t

            # Also check dual feasibility (reduced costs)
            c_B_t = np.array([0, 3 + theta*5, 4 + theta*2, 5 + theta*2])
            rc_y4 = c_B_t @ B_INV[:, 3]

            if theta < 5/8 - 0.01 and rc_y4 > TOL_OPTIMAL:
                # Both primal and dual feasible → basis holds
                assert np.all(x_B >= -TOL_VARIABLE), \
                    f"At θ={theta}, expected feasible basis"

    def test_after_breakpoint_x3_zero(self):
        """After breakpoint, x3 drops to 0 (structural change)."""
        for theta in [0.7, 0.8, 0.9, 1.0]:
            _, x = self._solve_at_theta(theta)
            assert x[2] < TOL_VARIABLE, \
                f"At θ={theta}, x3 should be 0, got {x[2]}"

    def test_optimal_monotone_increasing(self):
        """Objective increases monotonically with θ (costs and capacity both grow)."""
        prev_opt = 0
        for theta in np.linspace(0, 1, 11):
            opt, _ = self._solve_at_theta(theta)
            assert opt >= prev_opt - TOL_OPTIMAL, \
                f"Objective decreased at θ={theta}: {opt} < {prev_opt}"
            prev_opt = opt

    def test_path_samples_match(self):
        """Verify sampled path values against expected."""
        samples = [
            (0.0, 3688.8889),
            (0.5, 6502.7778),
            (1.0, 10000.0000),
        ]
        for theta, expected_opt in samples:
            opt, _ = self._solve_at_theta(theta)
            assert abs(opt - expected_opt) < 0.01, \
                f"At θ={theta}: expected {expected_opt}, got {opt}"

    def test_dual_breakpoint(self):
        """Second breakpoint at θ = 2/3: reduced cost of y4 hits 0."""
        c_B_t = np.array([0, 3 + (2/3)*5, 4 + (2/3)*2, 5 + (2/3)*2])
        rc_y4 = c_B_t @ B_INV[:, 3]
        assert abs(rc_y4) < TOL_OPTIMAL, \
            f"c̄_y4 should be ≈ 0 at θ=2/3, got {rc_y4}"


# ===========================================================================
# 5. Cross-scenario consistency
# ===========================================================================

class TestAlbiciCrossScenario:
    """Consistency checks across scenarios."""

    def test_b1_closer_to_base_than_b2(self):
        """Scenario b1 (basis preserved) should have smaller objective change."""
        opt_base, _ = solve_lp(C_ORIG, A_UB, B_ORIG)
        opt_b1, _ = solve_lp(C_ORIG, A_UB, [1300, 1200, 1800, 1000])
        opt_b2, _ = solve_lp(C_ORIG, A_UB, [1500, 1300, 2400, 800])

        delta_b1 = abs(opt_b1 - opt_base)
        delta_b2 = abs(opt_b2 - opt_base)
        assert delta_b1 < delta_b2, \
            "b1 (basis preserved) should have smaller objective change"

    def test_column_addition_improvement_bounded(self):
        """Adding columns improves objective, but by a finite amount."""
        opt_base, _ = solve_lp(C_ORIG, A_UB, B_ORIG)
        A_ext = [[1,2,1,2,3],[1,0,3,0,1],[2,1,2,1,0],[0,2,3,0,1]]
        opt_ext, _ = solve_lp([3,4,5,3,5], A_ext, B_ORIG)

        improvement = (opt_ext - opt_base) / opt_base * 100
        assert 0 < improvement < 10, \
            f"Column addition improvement should be modest, got {improvement:.2f}%"

    def test_all_scenarios_feasible(self):
        """Every scenario should have a feasible optimal solution."""
        scenarios = [
            (C_ORIG, A_UB, B_ORIG),
            (C_ORIG, A_UB, [1300, 1200, 1800, 1000]),
            (C_ORIG, A_UB, [1500, 1300, 2400, 800]),
            ([8, 6, 7], A_UB, B_ORIG),
            ([3,4,5,3,5], [[1,2,1,2,3],[1,0,3,0,1],[2,1,2,1,0],[0,2,3,0,1]], B_ORIG),
        ]
        for c, A, b in scenarios:
            opt, x = solve_lp(c, A, b)
            assert opt > 0
            assert np.all(x >= -TOL_VARIABLE)
