"""HiGHS solve engine — for large problems and cross-validation.

Uses highspy to solve LPs via the HiGHS solver.
Provides result-level explanation only (no iteration trace, no B^-1).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

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


class HiGHSBackend:
    """Solve engine using HiGHS via highspy."""

    def solve(self, problem: LPProblem) -> SolveState:
        """Solve using HiGHS and return a SolveState.

        Args:
            problem: The LP problem to solve.

        Returns:
            SolveState with sensitivity but no B^-1 or iteration history.
        """
        import highspy

        start = time.perf_counter()
        m, n = problem.A.shape

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)

        # Add variables with default bounds [0, +inf]
        for j in range(n):
            h.addVar(0.0, highspy.kHighsInf)

        # Set objective coefficients and sense from problem
        for j in range(n):
            h.changeColCost(j, float(problem.c[j]))
        if problem.sense == "minimize":
            h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        else:
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        # Add constraints: Ax <= b
        for i in range(m):
            row = problem.A[i]
            nonzero_idx = [int(j) for j in range(n) if row[j] != 0]
            nonzero_val = [float(row[j]) for j in nonzero_idx]
            h.addRow(-highspy.kHighsInf, float(problem.b[i]),
                     len(nonzero_idx), nonzero_idx, nonzero_val)

        # Solve
        h.run()
        elapsed = time.perf_counter() - start

        # Map status
        model_status = h.getInfoValue("primal_solution_status")[1]
        highs_status = h.getModelStatus()
        status = self._map_status(highs_status)

        if status != SolveStatus.OPTIMAL:
            return SolveState(
                status=status,
                optimal_value=float("nan"),
                variables=(),
                constraints=(),
                sensitivity=SensitivityRanges({}, {}),
                engine=EngineType.HIGHS,
                solve_time_seconds=elapsed,
                iteration_count=0,
                problem_name=problem.name,
                variable_names=tuple(problem.var_names),
                constraint_names=tuple(problem.constraint_names),
            )

        # Extract solution
        sol = h.getSolution()
        info = h.getInfoValue("objective_function_value")[1]
        iteration_count = int(h.getInfoValue("simplex_iteration_count")[1])
        basis = h.getBasis()

        # Sensitivity analysis via ranging
        _, ranging_data = h.getRanging()

        # Build variable info
        variables = []
        obj_ranges: dict[str, tuple[float, float]] = {}
        for j in range(n):
            val = sol.col_value[j]
            rc = sol.col_dual[j]
            bs = self._map_col_basis(basis.col_status[j])

            # Obj coeff ranges from ranging
            lo = ranging_data.col_cost_dn.value_[j]
            hi = ranging_data.col_cost_up.value_[j]
            obj_range = (lo, hi)
            var_name = problem.var_names[j]
            obj_ranges[var_name] = obj_range

            variables.append(VariableInfo(
                name=var_name,
                value=val,
                basis_status=bs,
                reduced_cost=rc,
                obj_coeff_range=obj_range,
            ))

        # Build constraint info
        constraints = []
        rhs_ranges: dict[str, tuple[float, float]] = {}
        for i in range(m):
            row_val = sol.row_value[i]
            dual = sol.row_dual[i]
            slack = float(problem.b[i]) - row_val
            binding = abs(slack) < 1e-8
            bs = self._map_row_basis(basis.row_status[i])
            con_name = problem.constraint_names[i]

            # RHS ranges from ranging
            rhs_lo = ranging_data.row_bound_dn.value_[i]
            rhs_hi = ranging_data.row_bound_up.value_[i]
            rhs_ranges[con_name] = (rhs_lo, rhs_hi)

            constraints.append(ConstraintInfo(
                name=con_name,
                rhs=float(problem.b[i]),
                slack=slack,
                dual_value=dual,
                is_binding=binding,
                basis_status=bs,
                rhs_range=(rhs_lo, rhs_hi),
            ))

        sensitivity = SensitivityRanges(
            obj_coeff_ranges=obj_ranges,
            rhs_ranges=rhs_ranges,
        )

        return SolveState(
            status=SolveStatus.OPTIMAL,
            optimal_value=info,
            variables=tuple(variables),
            constraints=tuple(constraints),
            sensitivity=sensitivity,
            engine=EngineType.HIGHS,
            solve_time_seconds=elapsed,
            iteration_count=iteration_count,
            basis_inverse=None,
            iteration_history=None,
            problem_name=problem.name,
            variable_names=tuple(problem.var_names),
            constraint_names=tuple(problem.constraint_names),
        )

    def _map_status(self, highs_status) -> SolveStatus:
        """Map HiGHS model status to SolveStatus."""
        import highspy
        mapping = {
            highspy.HighsModelStatus.kOptimal: SolveStatus.OPTIMAL,
            highspy.HighsModelStatus.kInfeasible: SolveStatus.INFEASIBLE,
            highspy.HighsModelStatus.kUnbounded: SolveStatus.UNBOUNDED,
        }
        return mapping.get(highs_status, SolveStatus.NUMERICAL_ERROR)

    def _map_col_basis(self, status) -> BasisStatus:
        """Map HiGHS column basis status."""
        import highspy
        mapping = {
            highspy.HighsBasisStatus.kBasic: BasisStatus.BASIC,
            highspy.HighsBasisStatus.kLower: BasisStatus.NONBASIC_LOWER,
            highspy.HighsBasisStatus.kUpper: BasisStatus.NONBASIC_UPPER,
            highspy.HighsBasisStatus.kZero: BasisStatus.NONBASIC_LOWER,
            highspy.HighsBasisStatus.kNonbasic: BasisStatus.NONBASIC_LOWER,
        }
        return mapping.get(status, BasisStatus.NONBASIC_LOWER)

    def _map_row_basis(self, status) -> BasisStatus:
        """Map HiGHS row basis status."""
        import highspy
        mapping = {
            highspy.HighsBasisStatus.kBasic: BasisStatus.BASIC,
            highspy.HighsBasisStatus.kLower: BasisStatus.NONBASIC_LOWER,
            highspy.HighsBasisStatus.kUpper: BasisStatus.NONBASIC_UPPER,
            highspy.HighsBasisStatus.kZero: BasisStatus.NONBASIC_UPPER,
            highspy.HighsBasisStatus.kNonbasic: BasisStatus.NONBASIC_UPPER,
        }
        return mapping.get(status, BasisStatus.NONBASIC_UPPER)
