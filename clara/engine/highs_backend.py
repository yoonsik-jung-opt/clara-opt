"""HiGHS solve engine — CLARA's single solver backend.

Solves LPs via highspy and reconstructs the simplex basis inverse
B^-1 of the augmented system [A_aug | I] from the optimal basis that
HiGHS reports. All downstream CLARA modules (region analyzer,
reoptimizer, parametric tracer, attribution) consume the reconstructed
SolveState and are therefore independent of solver internals.

Warm-starting: ``solve(problem, initial_basis=...)`` passes an
augmented-basis index list (from a previous SolveState) back to HiGHS
as an advanced starting basis, optionally forcing the primal or dual
simplex strategy.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

import clara.engine.standard_form as standard_form
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

# HiGHS option values for "simplex_strategy"
SIMPLEX_STRATEGY_CHOOSE = 0
SIMPLEX_STRATEGY_DUAL = 1
SIMPLEX_STRATEGY_PRIMAL = 4

_FEAS_TOL = 1e-6
_DEGEN_TOL = 1e-8


class HiGHSBackend:
    """Solve engine using HiGHS via highspy, with B^-1 reconstruction."""

    def solve(
        self,
        problem: LPProblem,
        initial_basis: Optional[Sequence[int]] = None,
        simplex_strategy: Optional[int] = None,
    ) -> SolveState:
        """Solve an LP with HiGHS and return a fully populated SolveState.

        Args:
            problem: The LP problem to solve.
            initial_basis: Optional augmented-basis column indices (as
                stored in ``SolveState.basis_indices`` of a previous
                solve with compatible dimensions). Passed to HiGHS as an
                advanced starting basis.
            simplex_strategy: Optional HiGHS ``simplex_strategy`` value
                (``SIMPLEX_STRATEGY_DUAL`` or ``SIMPLEX_STRATEGY_PRIMAL``)
                to force the warm-start method; default lets HiGHS choose.

        Returns:
            SolveState with sensitivity ranges (HiGHS ranging), the
            reconstructed ``basis_inverse``/``basis_indices`` of the
            augmented system, and numerical diagnostics
            (condition number, degenerate count, basis robustness radius).
        """
        import highspy

        start = time.perf_counter()
        m, n = problem.A.shape

        h = self._build_model(problem)

        if initial_basis is not None:
            basis_obj = self._augmented_to_highs_basis(problem, initial_basis)
            if basis_obj is not None:
                h.setBasis(basis_obj)
        if simplex_strategy is not None:
            h.setOptionValue("simplex_strategy", int(simplex_strategy))

        h.run()
        elapsed = time.perf_counter() - start

        status = self._map_status(h.getModelStatus())
        iteration_count = int(h.getInfoValue("simplex_iteration_count")[1])

        if status != SolveStatus.OPTIMAL:
            return SolveState(
                status=status,
                optimal_value=float("nan"),
                variables=(),
                constraints=(),
                sensitivity=SensitivityRanges({}, {}),
                engine=EngineType.HIGHS,
                solve_time_seconds=elapsed,
                iteration_count=iteration_count,
                problem_name=problem.name,
                variable_names=tuple(problem.var_names),
                constraint_names=tuple(problem.constraint_names),
            )

        sol = h.getSolution()
        obj_value = float(h.getInfoValue("objective_function_value")[1])
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
                lower_bound=float(problem.lower_bounds[j]) if problem.lower_bounds is not None else 0.0,
                upper_bound=float(problem.upper_bounds[j]) if problem.upper_bounds is not None else float("inf"),
            ))

        # Reconstruct B^-1 of the augmented system from the HiGHS basis
        recon = self._reconstruct_basis_inverse(problem, sol, basis)
        if recon is not None:
            B_inv, basis_indices, cond_num, degen_count, d0 = recon
        else:
            B_inv, basis_indices, cond_num, degen_count, d0 = None, None, None, 0, None

        # Exact basis-preserving RHS ranges from B^-1. HiGHS row-bound
        # ranging is only a basis-preserving range for binding (nonbasic)
        # rows; for basic (non-binding) rows it reports the range of the
        # row activity, which does not even contain the current rhs. The
        # exact range for row k follows from x_B + B^-1[:, k] * delta >= 0.
        exact_rhs = None
        if B_inv is not None:
            exact_rhs = self._exact_rhs_ranges(problem, B_inv)

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

            if exact_rhs is not None:
                rhs_lo, rhs_hi = exact_rhs[i]
            else:
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
            optimal_value=obj_value,
            variables=tuple(variables),
            constraints=tuple(constraints),
            sensitivity=sensitivity,
            engine=EngineType.HIGHS,
            solve_time_seconds=elapsed,
            iteration_count=iteration_count,
            basis_inverse=B_inv,
            basis_indices=tuple(basis_indices) if basis_indices is not None else None,
            iteration_history=None,
            condition_number=cond_num,
            degenerate_count=degen_count,
            basis_robustness_d0=d0,
            problem_name=problem.name,
            variable_names=tuple(problem.var_names),
            constraint_names=tuple(problem.constraint_names),
        )


    # ------------------------------------------------------------------
    # Exact RHS ranging from the reconstructed basis inverse
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_rhs_ranges(problem: LPProblem, B_inv) -> list[tuple[float, float]]:
        """Basis-preserving range of each original rhs b_k, from B^-1.

        With x_B(delta) = x_B + B^-1[:, k] * delta, the basis stays
        primal feasible (hence optimal) iff every component stays
        nonnegative, giving the allowable decrease and increase by two
        ratio tests. Rows of the augmented system beyond the original m
        (explicit upper-bound rows) are not ranged here.
        """
        import clara.engine.standard_form as standard_form

        _, b_aug, _ = standard_form.add_upper_bound_rows(problem)
        x_B = B_inv @ b_aug
        m = problem.A.shape[0]
        out = []
        for k in range(m):
            col = B_inv[:, k]
            dec = float("inf")
            inc = float("inf")
            for i in range(len(col)):
                if col[i] > 1e-12:
                    dec = min(dec, max(x_B[i], 0.0) / col[i])
                elif col[i] < -1e-12:
                    inc = min(inc, max(x_B[i], 0.0) / (-col[i]))
            b_k = float(problem.b[k])
            lo = b_k - dec if dec != float("inf") else float("-inf")
            hi = b_k + inc if inc != float("inf") else float("inf")
            out.append((lo, hi))
        return out

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, problem: LPProblem):
        """Build the HiGHS model: bounds native, rows as Ax <= b."""
        import highspy

        m, n = problem.A.shape
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)

        for j in range(n):
            lb = float(problem.lower_bounds[j]) if problem.lower_bounds is not None else 0.0
            ub = float(problem.upper_bounds[j]) if problem.upper_bounds is not None else highspy.kHighsInf
            if not np.isfinite(ub):
                ub = highspy.kHighsInf
            h.addVar(lb, ub)

        for j in range(n):
            h.changeColCost(j, float(problem.c[j]))
        if problem.sense == "minimize":
            h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        else:
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        for i in range(m):
            row = problem.A[i]
            nz = np.nonzero(row)[0]
            h.addRow(-highspy.kHighsInf, float(problem.b[i]),
                     len(nz), [int(j) for j in nz], [float(row[j]) for j in nz])
        return h

    # ------------------------------------------------------------------
    # Basis reconstruction (HiGHS basis -> augmented B^-1)
    # ------------------------------------------------------------------

    def _reconstruct_basis_inverse(self, problem: LPProblem, sol, basis):
        """Reconstruct B^-1 of the augmented system [A_aug | I].

        Augmented convention: one explicit row per finite upper bound.
        Basic columns are:
            - structural j with HiGHS status kBasic,
            - structural j at its upper bound (kUpper) — basic through
              its UB row, with that row's slack nonbasic at zero,
            - slack n+i for every original row i with status kBasic,
            - the UB-row slack for every finite-ub variable NOT at its
              upper bound.

        Returns (B_inv, basis_indices, cond, degen_count, d0), or None
        when reconstruction is not applicable (nonzero lower bounds) or
        fails numerically.
        """
        import highspy

        m, n = problem.A.shape

        # Convention requires x >= 0
        if problem.lower_bounds is not None and np.any(np.abs(problem.lower_bounds) > 1e-12):
            return None

        A_aug, b_aug, _ = standard_form.add_upper_bound_rows(problem)
        m_aug = A_aug.shape[0]
        ub_vars = standard_form.finite_ub_indices(problem)

        kBasic = highspy.HighsBasisStatus.kBasic
        kUpper = highspy.HighsBasisStatus.kUpper

        basis_indices: list[int] = []
        for j in range(n):
            if basis.col_status[j] == kBasic:
                basis_indices.append(j)
            elif basis.col_status[j] == kUpper and np.isfinite(problem.upper_bounds[j]):
                # At upper bound: basic in the augmented convention
                basis_indices.append(j)
        for i in range(m):
            if basis.row_status[i] == kBasic:
                basis_indices.append(n + i)
        for k, j in enumerate(ub_vars):
            if basis.col_status[j] != kUpper:
                basis_indices.append(n + m + k)

        if len(basis_indices) != m_aug:
            return None

        A_full = np.hstack([A_aug, np.eye(m_aug)])
        B = A_full[:, basis_indices]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return None

        # Verify: basic values must reproduce the HiGHS solution
        x_B = B_inv @ b_aug
        for pos, j in enumerate(basis_indices):
            if j < n and abs(x_B[pos] - sol.col_value[j]) > 1e-5 * max(1.0, abs(sol.col_value[j])):
                return None
        if np.any(x_B < -_FEAS_TOL):
            return None

        cond_num = float(np.linalg.cond(B_inv))
        degen_count = int(np.sum(x_B < _DEGEN_TOL))
        row_norms = np.linalg.norm(B_inv, axis=1)
        safe_norms = np.where(row_norms > 1e-12, row_norms, np.inf)
        d0 = float(np.min(x_B / safe_norms))

        return B_inv, basis_indices, cond_num, degen_count, d0

    # ------------------------------------------------------------------
    # Basis mapping (augmented indices -> HiGHS statuses) for warm start
    # ------------------------------------------------------------------

    def _augmented_to_highs_basis(self, problem: LPProblem, basis_indices: Sequence[int]):
        """Convert augmented basis indices back to a HighsBasis object.

        Returns None if the indices are inconsistent with the problem's
        dimensions (caller then falls back to a cold start).
        """
        import highspy

        m, n = problem.A.shape
        ub_vars = standard_form.finite_ub_indices(problem)
        m_aug = m + len(ub_vars)
        ub_slack_of_var = {j: n + m + k for k, j in enumerate(ub_vars)}

        basis_set = set(int(j) for j in basis_indices)
        if len(basis_set) != m_aug:
            return None
        if any(j < 0 or j >= n + m_aug for j in basis_set):
            return None

        kBasic = highspy.HighsBasisStatus.kBasic
        kLower = highspy.HighsBasisStatus.kLower
        kUpper = highspy.HighsBasisStatus.kUpper

        col_status = []
        for j in range(n):
            if j in basis_set:
                ub_slack = ub_slack_of_var.get(j)
                if ub_slack is not None and ub_slack not in basis_set:
                    # Basic through its UB row only -> at upper bound
                    col_status.append(kUpper)
                else:
                    col_status.append(kBasic)
            else:
                col_status.append(kLower)

        row_status = []
        for i in range(m):
            if (n + i) in basis_set:
                row_status.append(kBasic)
            else:
                row_status.append(kUpper)  # binding <= row

        basis_obj = highspy.HighsBasis()
        basis_obj.col_status = col_status
        basis_obj.row_status = row_status
        basis_obj.valid = True
        basis_obj.alien = True  # let HiGHS repair inconsistencies
        return basis_obj

    # ------------------------------------------------------------------
    # Status mapping
    # ------------------------------------------------------------------

    def _map_status(self, highs_status) -> SolveStatus:
        """Map HiGHS model status to SolveStatus."""
        import highspy
        mapping = {
            highspy.HighsModelStatus.kOptimal: SolveStatus.OPTIMAL,
            highspy.HighsModelStatus.kInfeasible: SolveStatus.INFEASIBLE,
            highspy.HighsModelStatus.kUnbounded: SolveStatus.UNBOUNDED,
            highspy.HighsModelStatus.kIterationLimit: SolveStatus.ITERATION_LIMIT,
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


def solve(problem: LPProblem) -> SolveState:
    """Convenience function: solve an LP with the HiGHS backend."""
    return HiGHSBackend().solve(problem)
