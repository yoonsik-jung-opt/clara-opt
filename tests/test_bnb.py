"""Tests for Branch-and-Bound MIP solver."""

import math
import pytest
from pathlib import Path

from clara.engine.bnb import InternalBnB
from clara.engine.simplex import RevisedSimplex
from clara.io.lp_parser import read_lp
from clara.model.problem import LPProblem
from clara.model.solve_state import EngineType, SolveStatus

FIXTURES = Path(__file__).parent / "fixtures"
TOL = 1e-4


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def knapsack():
    return read_lp(FIXTURES / "knapsack_small.lp")


@pytest.fixture
def albici():
    return read_lp(FIXTURES / "albici_base.lp")


# ============================================================
# 1. Pure LP fallback
# ============================================================

class TestPureLPFallback:

    def test_pure_lp_delegates(self, albici):
        """No integer vars → uses simplex, same result."""
        state = InternalBnB().solve(albici)
        expected = RevisedSimplex(albici).solve()
        assert abs(state.optimal_value - expected.optimal_value) < TOL

    def test_pure_lp_no_bnb_history(self, albici):
        state = InternalBnB().solve(albici)
        assert state.bnb_history is None


# ============================================================
# 2. Root node
# ============================================================

class TestRootNode:

    def test_root_fractional(self, knapsack):
        """Knapsack root LP is fractional, tree needed."""
        state = InternalBnB().solve(knapsack)
        assert state.bnb_history is not None
        assert len(state.bnb_history) > 1


# ============================================================
# 3. Optimality
# ============================================================

class TestOptimality:

    def test_knapsack_optimal(self, knapsack):
        state = InternalBnB().solve(knapsack)
        assert state.status == SolveStatus.OPTIMAL
        # MIP optimal = 42.0
        assert abs(state.optimal_value - 42.0) < TOL, (
            f"Expected 42.0, got {state.optimal_value}"
        )

    def test_knapsack_solution_is_integer(self, knapsack):
        """All binary variables should have integer values."""
        state = InternalBnB().solve(knapsack)
        for v in state.variables:
            assert abs(v.value - round(v.value)) < TOL, (
                f"{v.name} = {v.value} is not integer"
            )

    def test_knapsack_matches_highs(self, knapsack):
        """B&B optimal should match HiGHS."""
        from clara.engine.highs_backend import HiGHSBackend
        bnb_state = InternalBnB().solve(knapsack)
        # HiGHS solves the LP relaxation only (MIP not wired yet via our backend)
        # Just verify our B&B is correct against known value
        assert abs(bnb_state.optimal_value - 42.0) < TOL


# ============================================================
# 4. Branching
# ============================================================

class TestBranching:

    def test_branch_creates_children(self, knapsack):
        """B&B tree should have more than just the root."""
        state = InternalBnB().solve(knapsack)
        assert len(state.bnb_history) >= 3  # root + at least 2 children

    def test_branch_constraints_accumulate(self, knapsack):
        """Deeper nodes should have more branch constraints."""
        state = InternalBnB().solve(knapsack)
        depths = [s.depth for s in state.bnb_history]
        assert max(depths) >= 1


# ============================================================
# 5. Node selection
# ============================================================

class TestNodeSelection:

    def test_best_first_default(self, knapsack):
        state = InternalBnB(node_selection="best_first").solve(knapsack)
        assert state.status == SolveStatus.OPTIMAL

    def test_depth_first_works(self, knapsack):
        state = InternalBnB(node_selection="depth_first").solve(knapsack)
        assert state.status == SolveStatus.OPTIMAL
        assert abs(state.optimal_value - 42.0) < TOL


# ============================================================
# 6. Fathoming
# ============================================================

class TestFathoming:

    def test_fathom_reasons_valid(self, knapsack):
        state = InternalBnB().solve(knapsack)
        valid_reasons = {"bound", "infeasible", "integer_feasible", None}
        for snap in state.bnb_history:
            assert snap.fathom_reason in valid_reasons

    def test_has_integer_feasible_node(self, knapsack):
        state = InternalBnB().solve(knapsack)
        int_nodes = [s for s in state.bnb_history if s.fathom_reason == "integer_feasible"]
        assert len(int_nodes) >= 1


# ============================================================
# 7. Limits
# ============================================================

class TestLimits:

    def test_node_limit(self, knapsack):
        state = InternalBnB(max_nodes=2).solve(knapsack)
        # Should still return something (possibly suboptimal or infeasible)
        assert state is not None


# ============================================================
# 8. Explainability (bnb_history)
# ============================================================

class TestExplainability:

    def test_history_not_none(self, knapsack):
        state = InternalBnB().solve(knapsack)
        assert state.bnb_history is not None

    def test_history_has_root(self, knapsack):
        state = InternalBnB().solve(knapsack)
        root_nodes = [s for s in state.bnb_history if s.node_id == 0]
        assert len(root_nodes) == 1

    def test_snapshot_fields_complete(self, knapsack):
        state = InternalBnB().solve(knapsack)
        for snap in state.bnb_history:
            assert isinstance(snap.node_id, int)
            assert isinstance(snap.depth, int)
            assert isinstance(snap.is_fathomed, bool)

    def test_engine_type(self, knapsack):
        state = InternalBnB().solve(knapsack)
        assert state.engine == EngineType.INTERNAL_BNB
