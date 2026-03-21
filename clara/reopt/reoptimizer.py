"""Warm-start Reoptimizer — reoptimize using the old solution's B⁻¹.

Implements the reoptimization procedures from Albici et al. (2010)
with B⁻¹-based warm-start for the Internal Simplex engine.

Pipeline position:
    Change Detector → Impact Analyzer → **Reoptimizer** → Diff Report
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from clara.engine.simplex import RevisedSimplex
from clara.model.problem import LPProblem
from clara.model.solve_state import (
    BasisStatus,
    ConstraintInfo,
    EngineType,
    SensitivityRanges,
    SolveState,
    SolveStatus,
    VariableInfo,
)
from clara.reopt.types import ParameterChange, ReoptDecision, ReoptResult


class Reoptimizer:
    """Reoptimize using the old solution's B⁻¹ for warm-start."""

    def reoptimize(
        self,
        old_state: SolveState,
        new_problem: LPProblem,
        change: ParameterChange,
        decision: ReoptDecision,
    ) -> ReoptResult:
        """Reoptimize the new problem using information from the old solution.

        Args:
            old_state: Previous SolveState (must have B⁻¹ for warm-start).
            new_problem: The modified LP problem.
            change: Detected parameter change.
            decision: Impact Analyzer's recommendation.

        Returns:
            ReoptResult with the new SolveState and metadata.
        """
        method = decision.recommended_method or "scratch"

        if method == "none":
            return self._no_action(old_state, new_problem, change)
        elif method == "recompute":
            return self._recompute(old_state, new_problem, change)
        elif method == "warm_start":
            return self._warm_start(old_state, new_problem, change)
        elif method == "parametric_lp":
            return self._parametric_lp(old_state, new_problem, change)
        else:
            return self._scratch(new_problem)

    def _no_action(
        self,
        old_state: SolveState,
        new_problem: LPProblem,
        change: ParameterChange,
    ) -> ReoptResult:
        """No action needed — same solution, just recompute objective."""
        start = time.perf_counter()

        # Recompute objective with new coefficients
        x_vals = np.array([v.value for v in old_state.variables])
        new_obj = float(new_problem.c @ x_vals)

        # Build new state with updated objective
        new_state = SolveState(
            status=old_state.status,
            optimal_value=new_obj,
            variables=old_state.variables,
            constraints=old_state.constraints,
            sensitivity=old_state.sensitivity,
            engine=old_state.engine,
            solve_time_seconds=time.perf_counter() - start,
            iteration_count=0,
            basis_inverse=old_state.basis_inverse,
            iteration_history=(),
            problem_name=new_problem.name or old_state.problem_name,
            variable_names=old_state.variable_names,
            constraint_names=old_state.constraint_names,
        )

        return ReoptResult(
            new_state=new_state,
            method_used="none",
            pivots=0,
            scratch_estimate=old_state.iteration_count,
            basis_preserved=True,
            reopt_time_seconds=new_state.solve_time_seconds,
        )

    def _recompute(
        self,
        old_state: SolveState,
        new_problem: LPProblem,
        change: ParameterChange,
    ) -> ReoptResult:
        """Basis preserved — recompute x_B = B⁻¹ * b_new (zero pivots)."""
        start = time.perf_counter()

        B_inv = old_state.basis_inverse
        if B_inv is None:
            return self._scratch(new_problem)

        # Extract basis indices from old state
        basis = self._extract_basis_indices(old_state, new_problem)
        n = new_problem.num_variables
        m = new_problem.num_constraints

        # Recompute basic variable values
        x_B_new = B_inv @ new_problem.b

        # Verify feasibility
        if np.any(x_B_new < -1e-8):
            return self._warm_start(old_state, new_problem, change)

        # Build full solution
        N = n + m  # decision + slack variables
        x_full = np.zeros(N)
        for i, j in enumerate(basis):
            x_full[j] = x_B_new[i]

        new_obj = float(new_problem.c @ x_full[:n])

        # Dual values (same basis → same duals for same c)
        c_full = np.zeros(N)
        c_full[:n] = new_problem.c
        c_B = np.array([c_full[j] for j in basis])
        y = c_B @ B_inv

        # Build VariableInfo
        basis_set = set(basis)
        variables = []
        for j in range(n):
            variables.append(VariableInfo(
                name=new_problem.var_names[j],
                value=float(x_full[j]),
                basis_status=BasisStatus.BASIC if j in basis_set else BasisStatus.NONBASIC_LOWER,
                reduced_cost=0.0 if j in basis_set else float(c_full[j] - y @ new_problem.A[:, j]),
                obj_coeff_range=old_state.sensitivity.obj_coeff_ranges.get(
                    new_problem.var_names[j], (float("-inf"), float("inf"))
                ),
            ))

        # Build ConstraintInfo
        constraints = []
        for i in range(m):
            slack = float(x_full[n + i])
            constraints.append(ConstraintInfo(
                name=new_problem.constraint_names[i],
                rhs=float(new_problem.b[i]),
                slack=slack,
                dual_value=float(y[i]),
                is_binding=abs(slack) < 1e-8,
                basis_status=BasisStatus.BASIC if (n + i) in basis_set else BasisStatus.NONBASIC_UPPER,
                rhs_range=old_state.sensitivity.rhs_ranges.get(
                    new_problem.constraint_names[i], (float("-inf"), float("inf"))
                ),
            ))

        elapsed = time.perf_counter() - start
        new_state = SolveState(
            status=SolveStatus.OPTIMAL,
            optimal_value=new_obj,
            variables=tuple(variables),
            constraints=tuple(constraints),
            sensitivity=old_state.sensitivity,
            engine=EngineType.INTERNAL_SIMPLEX,
            solve_time_seconds=elapsed,
            iteration_count=0,
            basis_inverse=B_inv,
            iteration_history=(),
            problem_name=new_problem.name or old_state.problem_name,
            variable_names=tuple(new_problem.var_names),
            constraint_names=tuple(new_problem.constraint_names),
        )

        return ReoptResult(
            new_state=new_state,
            method_used="recompute",
            pivots=0,
            scratch_estimate=old_state.iteration_count,
            basis_preserved=True,
            reopt_time_seconds=elapsed,
        )

    def _warm_start(
        self,
        old_state: SolveState,
        new_problem: LPProblem,
        change: ParameterChange,
    ) -> ReoptResult:
        """Warm-start simplex from the old basis."""
        B_inv = old_state.basis_inverse
        if B_inv is None:
            return self._scratch(new_problem)

        basis = self._extract_basis_indices(old_state, new_problem)
        solver = RevisedSimplex.from_warm_start(new_problem, basis, B_inv)
        new_state = solver.solve()

        return ReoptResult(
            new_state=new_state,
            method_used="warm_start",
            pivots=len(solver.iterations),
            scratch_estimate=old_state.iteration_count,
            basis_preserved=False,
            reopt_time_seconds=new_state.solve_time_seconds,
        )

    def _scratch(self, new_problem: LPProblem) -> ReoptResult:
        """Full re-solve from scratch."""
        solver = RevisedSimplex(new_problem)
        new_state = solver.solve()

        return ReoptResult(
            new_state=new_state,
            method_used="scratch",
            pivots=len(solver.iterations),
            scratch_estimate=None,
            basis_preserved=False,
            reopt_time_seconds=new_state.solve_time_seconds,
        )

    def _parametric_lp(
        self,
        old_state: SolveState,
        new_problem: LPProblem,
        change: ParameterChange,
    ) -> ReoptResult:
        """Parametric LP — placeholder, falls back to warm-start."""
        return self._warm_start(old_state, new_problem, change)

    def _extract_basis_indices(
        self,
        state: SolveState,
        problem: LPProblem,
    ) -> list[int]:
        """Extract basis column indices from SolveState.

        In the augmented system [A|I]:
            Columns 0..n-1 are decision variables
            Columns n..n+m-1 are slack variables
        """
        n = problem.num_variables
        m = problem.num_constraints
        basis_set = set()

        basic_decision = []
        for j, v in enumerate(state.variables):
            if j < n and v.basis_status == BasisStatus.BASIC:
                basic_decision.append(j)

        basic_slack = []
        for i, c in enumerate(state.constraints):
            if i < m and c.basis_status == BasisStatus.BASIC:
                basic_slack.append(i)

        # Build basis: slack in its natural row, decision vars fill remaining
        basis = [0] * m
        used_rows: set[int] = set()

        for i in basic_slack:
            basis[i] = n + i
            used_rows.add(i)

        remaining = [i for i in range(m) if i not in used_rows]
        for j, row in zip(basic_decision, remaining):
            basis[row] = j

        return basis
