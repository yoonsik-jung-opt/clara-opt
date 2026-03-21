"""Educational Branch-and-Bound MIP solver with full explainability.

Uses InternalSimplex as LP relaxation solver at each node.
Every branching decision is recorded as a BnBNodeSnapshot.

For performance on real MIP problems, use HiGHS backend.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from clara.engine.simplex import RevisedSimplex
from clara.model.problem import LPProblem
from clara.model.solve_state import (
    BnBNodeSnapshot,
    EngineType,
    SensitivityRanges,
    SolveState,
    SolveStatus,
)


@dataclass
class BranchDecision:
    """Result of branching variable selection."""
    variable: str
    variable_index: int
    current_value: float
    branch_down: float
    branch_up: float
    fractionality: float


@dataclass
class _BnBNode:
    """Internal B&B tree node."""
    node_id: int
    parent_id: Optional[int]
    depth: int
    lp_bound: float
    branch_constraints: list[tuple[int, str, float]] = field(default_factory=list)
    # (var_index, "<="|">=", value)

    def __lt__(self, other: _BnBNode) -> bool:
        # For max: higher bound = better → negate for min-heap
        return self.lp_bound > other.lp_bound


class _BestFirstQueue:
    """Priority queue ordered by LP bound (best first for maximization)."""

    def __init__(self) -> None:
        self._heap: list[_BnBNode] = []

    def push(self, node: _BnBNode) -> None:
        heapq.heappush(self._heap, node)

    def pop(self) -> _BnBNode:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def prune(self, cutoff: float) -> int:
        """Remove nodes with bound <= cutoff (for max). Returns count removed."""
        before = len(self._heap)
        self._heap = [n for n in self._heap if n.lp_bound > cutoff]
        heapq.heapify(self._heap)
        return before - len(self._heap)


class _DepthFirstQueue:
    """Stack (LIFO) for depth-first search."""

    def __init__(self) -> None:
        self._stack: list[_BnBNode] = []

    def push(self, node: _BnBNode) -> None:
        self._stack.append(node)

    def pop(self) -> _BnBNode:
        return self._stack.pop()

    def __len__(self) -> int:
        return len(self._stack)

    def prune(self, cutoff: float) -> int:
        before = len(self._stack)
        self._stack = [n for n in self._stack if n.lp_bound > cutoff]
        return before - len(self._stack)


class InternalBnB:
    """Educational Branch-and-Bound MIP solver.

    Args:
        node_selection: "best_first" or "depth_first".
        max_nodes: Maximum number of nodes to explore.
        max_time: Time limit in seconds.
        gap_tolerance: Relative MIP gap tolerance.
        int_tolerance: Integrality tolerance.
    """

    def __init__(
        self,
        node_selection: str = "best_first",
        max_nodes: int = 10_000,
        max_time: float = 60.0,
        gap_tolerance: float = 1e-6,
        int_tolerance: float = 1e-6,
    ) -> None:
        self.node_selection = node_selection
        self.max_nodes = max_nodes
        self.max_time = max_time
        self.gap_tolerance = gap_tolerance
        self.int_tolerance = int_tolerance

    def solve(self, problem: LPProblem) -> SolveState:
        """Solve a MIP and return SolveState with full B&B trace."""
        start = time.perf_counter()

        # Pure LP: delegate to simplex
        if not problem.has_integers:
            return RevisedSimplex(problem).solve()

        # Add upper bound constraints for bounded variables
        # (InternalSimplex only handles Ax <= b, x >= 0 — no native upper bounds)
        problem = self._add_upper_bound_constraints(problem)

        # All integer/binary var indices
        int_vars = problem.integer_vars | problem.binary_vars

        # Root LP relaxation
        root_state = RevisedSimplex(problem).solve()
        if not root_state.is_optimal:
            return root_state

        # Check if root is already integer feasible
        if self._is_integer_feasible(root_state, int_vars):
            return self._finalize(root_state, [], problem, time.perf_counter() - start)

        # Initialize tree
        incumbent: Optional[SolveState] = None
        incumbent_value = float("-inf")
        queue = self._make_queue()
        snapshots: list[BnBNodeSnapshot] = []
        next_id = 1
        nodes_explored = 0

        # Process root node inline (already solved above)
        decision = self._select_branching_variable(root_state, problem, int_vars)
        snapshots.append(BnBNodeSnapshot(
            node_id=0, parent_id=None, depth=0,
            branching_var=decision.variable if decision else "",
            branching_direction="",
            branching_value=decision.current_value if decision else 0.0,
            lp_relaxation_value=root_state.optimal_value,
            is_integer_feasible=False,
            is_fathomed=False,
            fathom_reason=None,
            incumbent_value=None,
        ))

        if decision:
            down = _BnBNode(
                node_id=next_id, parent_id=0, depth=1,
                lp_bound=root_state.optimal_value,
                branch_constraints=[(decision.variable_index, "<=", decision.branch_down)],
            )
            up = _BnBNode(
                node_id=next_id + 1, parent_id=0, depth=1,
                lp_bound=root_state.optimal_value,
                branch_constraints=[(decision.variable_index, ">=", decision.branch_up)],
            )
            next_id += 2
            queue.push(down)
            queue.push(up)

        while len(queue) > 0:
            elapsed = time.perf_counter() - start
            if nodes_explored >= self.max_nodes or elapsed > self.max_time:
                break

            node = queue.pop()

            # Bound pruning
            if incumbent and node.lp_bound <= incumbent_value + self.gap_tolerance:
                snapshots.append(BnBNodeSnapshot(
                    node_id=node.node_id, parent_id=node.parent_id, depth=node.depth,
                    branching_var="", branching_direction="",
                    branching_value=0.0,
                    lp_relaxation_value=node.lp_bound,
                    is_integer_feasible=False, is_fathomed=True,
                    fathom_reason="bound",
                    incumbent_value=incumbent_value,
                ))
                continue

            # Solve LP at this node
            node_problem = self._add_bound_constraints(problem, node.branch_constraints)
            node_state = RevisedSimplex(node_problem).solve()
            nodes_explored += 1

            # Infeasible
            if not node_state.is_optimal:
                snapshots.append(BnBNodeSnapshot(
                    node_id=node.node_id, parent_id=node.parent_id, depth=node.depth,
                    branching_var="", branching_direction="",
                    branching_value=0.0,
                    lp_relaxation_value=float("nan"),
                    is_integer_feasible=False, is_fathomed=True,
                    fathom_reason="infeasible",
                    incumbent_value=incumbent_value if incumbent else None,
                ))
                continue

            # Bound pruning after solve
            if incumbent and node_state.optimal_value <= incumbent_value + self.gap_tolerance:
                snapshots.append(BnBNodeSnapshot(
                    node_id=node.node_id, parent_id=node.parent_id, depth=node.depth,
                    branching_var="", branching_direction="",
                    branching_value=0.0,
                    lp_relaxation_value=node_state.optimal_value,
                    is_integer_feasible=False, is_fathomed=True,
                    fathom_reason="bound",
                    incumbent_value=incumbent_value,
                ))
                continue

            # Integer feasible?
            if self._is_integer_feasible(node_state, int_vars):
                if node_state.optimal_value > incumbent_value:
                    incumbent = node_state
                    incumbent_value = node_state.optimal_value
                    queue.prune(incumbent_value)
                snapshots.append(BnBNodeSnapshot(
                    node_id=node.node_id, parent_id=node.parent_id, depth=node.depth,
                    branching_var="", branching_direction="",
                    branching_value=0.0,
                    lp_relaxation_value=node_state.optimal_value,
                    is_integer_feasible=True, is_fathomed=True,
                    fathom_reason="integer_feasible",
                    incumbent_value=incumbent_value,
                ))
                continue

            # Branch
            decision = self._select_branching_variable(node_state, problem, int_vars)
            if not decision:
                continue

            snapshots.append(BnBNodeSnapshot(
                node_id=node.node_id, parent_id=node.parent_id, depth=node.depth,
                branching_var=decision.variable,
                branching_direction="",
                branching_value=decision.current_value,
                lp_relaxation_value=node_state.optimal_value,
                is_integer_feasible=False, is_fathomed=False,
                fathom_reason=None,
                incumbent_value=incumbent_value if incumbent else None,
            ))

            down = _BnBNode(
                node_id=next_id, parent_id=node.node_id, depth=node.depth + 1,
                lp_bound=node_state.optimal_value,
                branch_constraints=node.branch_constraints + [
                    (decision.variable_index, "<=", decision.branch_down)
                ],
            )
            up = _BnBNode(
                node_id=next_id + 1, parent_id=node.node_id, depth=node.depth + 1,
                lp_bound=node_state.optimal_value,
                branch_constraints=node.branch_constraints + [
                    (decision.variable_index, ">=", decision.branch_up)
                ],
            )
            next_id += 2
            queue.push(down)
            queue.push(up)

        elapsed = time.perf_counter() - start

        if incumbent is None:
            # No integer feasible solution found
            status = SolveStatus.INFEASIBLE
            if nodes_explored >= self.max_nodes:
                status = SolveStatus.ITERATION_LIMIT
            return SolveState(
                status=status,
                optimal_value=float("nan"),
                variables=(),
                constraints=(),
                sensitivity=SensitivityRanges({}, {}),
                engine=EngineType.INTERNAL_BNB,
                solve_time_seconds=elapsed,
                iteration_count=nodes_explored,
                bnb_history=tuple(snapshots),
                problem_name=problem.name,
                variable_names=tuple(problem.var_names),
                constraint_names=tuple(problem.constraint_names),
            )

        return self._finalize(incumbent, snapshots, problem, elapsed)

    def _add_upper_bound_constraints(self, problem: LPProblem) -> LPProblem:
        """Add x_j <= ub as explicit constraints for finite upper bounds.

        InternalSimplex only handles Ax <= b, x >= 0 with no native upper bounds.
        """
        extra_rows = []
        extra_rhs = []
        extra_names = []
        for j in range(problem.num_variables):
            ub = problem.upper_bounds[j]
            if not np.isinf(ub):
                row = np.zeros(problem.num_variables)
                row[j] = 1.0
                extra_rows.append(row)
                extra_rhs.append(ub)
                extra_names.append(f"ub_{problem.var_names[j]}")

        if not extra_rows:
            return problem

        A_new = np.vstack([problem.A] + extra_rows)
        b_new = np.concatenate([problem.b, extra_rhs])
        names_new = list(problem.constraint_names) + extra_names

        return LPProblem(
            c=problem.c.copy(),
            A=A_new,
            b=b_new,
            var_names=list(problem.var_names),
            constraint_names=names_new,
            name=problem.name,
            lower_bounds=problem.lower_bounds.copy() if problem.lower_bounds is not None else None,
            upper_bounds=problem.upper_bounds.copy() if problem.upper_bounds is not None else None,
            integer_vars=set(problem.integer_vars),
            binary_vars=set(problem.binary_vars),
        )

    def _make_queue(self) -> _BestFirstQueue | _DepthFirstQueue:
        if self.node_selection == "depth_first":
            return _DepthFirstQueue()
        return _BestFirstQueue()

    def _is_integer_feasible(self, state: SolveState, int_vars: set[int]) -> bool:
        """Check if all integer variables have integer values."""
        for j in int_vars:
            if j < len(state.variables):
                val = state.variables[j].value
                if abs(val - round(val)) > self.int_tolerance:
                    return False
        return True

    def _select_branching_variable(
        self, state: SolveState, problem: LPProblem, int_vars: set[int]
    ) -> Optional[BranchDecision]:
        """Most fractional branching: pick the integer variable closest to 0.5."""
        best: Optional[BranchDecision] = None
        best_score = -1.0

        for j in int_vars:
            if j >= len(state.variables):
                continue
            val = state.variables[j].value
            frac = val - math.floor(val)
            if frac < self.int_tolerance or frac > 1 - self.int_tolerance:
                continue  # already integer
            score = 0.5 - abs(frac - 0.5)
            if score > best_score or (score == best_score and (best is None or j < best.variable_index)):
                best_score = score
                best = BranchDecision(
                    variable=problem.var_names[j],
                    variable_index=j,
                    current_value=val,
                    branch_down=math.floor(val),
                    branch_up=math.ceil(val),
                    fractionality=frac,
                )

        return best

    def _add_bound_constraints(
        self, problem: LPProblem, constraints: list[tuple[int, str, float]]
    ) -> LPProblem:
        """Create a copy of the problem with additional bound constraints as rows."""
        if not constraints:
            return problem

        extra_rows = []
        extra_rhs = []
        for var_idx, sense, value in constraints:
            row = np.zeros(problem.num_variables)
            if sense == "<=":
                row[var_idx] = 1.0
                extra_rows.append(row)
                extra_rhs.append(value)
            elif sense == ">=":
                row[var_idx] = -1.0
                extra_rows.append(row)
                extra_rhs.append(-value)

        A_new = np.vstack([problem.A] + extra_rows)
        b_new = np.concatenate([problem.b, extra_rhs])

        # Generate constraint names for new rows
        new_names = list(problem.constraint_names) + [
            f"bnb_{i}" for i in range(len(extra_rows))
        ]

        return LPProblem(
            c=problem.c.copy(),
            A=A_new,
            b=b_new,
            var_names=list(problem.var_names),
            constraint_names=new_names,
            name=problem.name,
            lower_bounds=problem.lower_bounds.copy() if problem.lower_bounds is not None else None,
            upper_bounds=problem.upper_bounds.copy() if problem.upper_bounds is not None else None,
            integer_vars=set(problem.integer_vars),
            binary_vars=set(problem.binary_vars),
        )

    def _finalize(
        self,
        incumbent: SolveState,
        snapshots: list[BnBNodeSnapshot],
        problem: LPProblem,
        elapsed: float,
    ) -> SolveState:
        """Build final SolveState from incumbent with B&B metadata."""
        # Only keep variables/constraints from the original problem dimensions
        orig_n = problem.num_variables
        orig_m = problem.num_constraints
        variables = incumbent.variables[:orig_n]
        constraints = incumbent.constraints[:orig_m]

        return SolveState(
            status=SolveStatus.OPTIMAL,
            optimal_value=incumbent.optimal_value,
            variables=variables,
            constraints=constraints,
            sensitivity=incumbent.sensitivity,
            engine=EngineType.INTERNAL_BNB,
            solve_time_seconds=elapsed,
            iteration_count=len(snapshots),
            basis_inverse=incumbent.basis_inverse,
            iteration_history=incumbent.iteration_history,
            bnb_history=tuple(snapshots) if snapshots else None,
            problem_name=problem.name,
            variable_names=tuple(problem.var_names),
            constraint_names=tuple(problem.constraint_names),
        )
