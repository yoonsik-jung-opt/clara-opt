"""Revised Simplex Method — dense matrix implementation with full tracing.

Solves: max c^T x  s.t.  Ax <= b,  x >= 0

Features:
    - Dense B⁻¹ maintained every iteration (product-form update)
    - Bland's rule for anti-cycling
    - IterationSnapshot recorded at every pivot
    - Sensitivity analysis (obj coeff ranging, RHS ranging) from final B⁻¹
    - Returns SolveState with complete internal trace
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
    IterationSnapshot,
    SensitivityRanges,
    SolveState,
    SolveStatus,
    VariableInfo,
)

PIVOT_TOL = 1e-10
OPTIMALITY_TOL = 1e-8
MAX_ITERATIONS = 10_000


class InfeasibleError(Exception):
    pass


class UnboundedError(Exception):
    pass


class RevisedSimplex:
    """Revised Simplex solver with full transparency.

    Internally works with the augmented system [A | I] for slack variables.
    Variables 0..n-1 are decision variables, n..n+m-1 are slacks.

    Args:
        problem: The LP problem to solve.
    """

    def __init__(self, problem: LPProblem) -> None:
        self.problem = problem

        # Add upper bound constraints for finite upper bounds
        # (simplex only handles Ax <= b, x >= 0 natively)
        A, b, con_names = self._add_upper_bound_rows(problem)

        m = A.shape[0]
        n = problem.A.shape[1]
        self.m = m
        self.n = n
        self.N = n + m

        # Full variable names: x1..xn, y1..ym (slacks)
        self.all_var_names = list(problem.var_names) + [
            f"y{i+1}" for i in range(m)
        ]

        # Objective: decision vars have cost c_j, slacks have cost 0
        # For minimize: negate c internally (simplex always maximizes)
        self.c_full = np.zeros(self.N)
        if problem.sense == "minimize":
            self.c_full[:n] = -problem.c
        else:
            self.c_full[:n] = problem.c

        # Constraint matrix: [A | I]
        self.A_full = np.hstack([A, np.eye(m)])

        # RHS
        self.b = b.copy()

        # Initial basis: slack variables (indices n, n+1, ..., n+m-1)
        self.basis = list(range(n, n + m))

        # B⁻¹ starts as identity (basis = slacks)
        self.B_inv = np.eye(m)

        # History
        self.iterations: list[IterationSnapshot] = []

    @staticmethod
    def _add_upper_bound_rows(problem: LPProblem):
        """Add x_j <= ub as explicit constraints for finite upper bounds."""
        n = problem.A.shape[1]
        extra_rows = []
        extra_rhs = []
        extra_names = []
        if problem.upper_bounds is not None:
            for j in range(n):
                ub = problem.upper_bounds[j]
                if np.isfinite(ub):
                    row = np.zeros(n)
                    row[j] = 1.0
                    extra_rows.append(row)
                    extra_rhs.append(ub)
                    extra_names.append(f"ub_{problem.var_names[j]}")

        if extra_rows:
            A = np.vstack([problem.A] + extra_rows)
            b = np.concatenate([problem.b, extra_rhs])
            names = list(problem.constraint_names) + extra_names
        else:
            A = problem.A
            b = problem.b
            names = list(problem.constraint_names)

        return A, b, names

    @classmethod
    def from_warm_start(
        cls,
        problem: LPProblem,
        basis: list[int],
        basis_inverse: np.ndarray,
    ) -> "RevisedSimplex":
        """Create a solver warm-started from an existing basis.

        solve() will start from the given basis instead of the identity,
        checking optimality and pivoting as needed.

        Args:
            problem: The (possibly modified) LP problem.
            basis: List of m column indices forming the basis.
            basis_inverse: The m×m B⁻¹ matrix from the previous solve.
        """
        solver = cls(problem)
        solver.basis = list(basis)
        solver.B_inv = basis_inverse.copy()
        return solver

    def solve(self) -> SolveState:
        """Run Revised Simplex and return a SolveState."""
        start_time = time.perf_counter()

        # Handle negative RHS via Big-M Phase I
        if np.any(self.b < -PIVOT_TOL):
            return self._solve_with_bigm(start_time)

        status = self._simplex_loop()
        elapsed = time.perf_counter() - start_time
        return self._make_state(status, elapsed)

    def _simplex_loop(self) -> SolveStatus:
        """Core simplex pivot loop. Shared by solve() and _solve_with_bigm().

        Performs pricing, ratio test, pivot, and records IterationSnapshots.
        Modifies self.B_inv, self.basis, self.iterations in place.
        """
        for iteration in range(1, MAX_ITERATIONS + 1):
            x_B = self.B_inv @ self.b

            # Pricing: find entering variable (Bland's rule)
            c_B = self.c_full[self.basis]
            y = c_B @ self.B_inv

            entering_col = -1
            entering_rc = 0.0
            for j in range(self.N):
                if j in self.basis:
                    continue
                rc_j = self.c_full[j] - y @ self.A_full[:, j]
                if rc_j > OPTIMALITY_TOL:
                    entering_col = j
                    entering_rc = rc_j
                    break

            if entering_col == -1:
                return SolveStatus.OPTIMAL

            # Direction
            d = self.B_inv @ self.A_full[:, entering_col]

            # Ratio test (Bland's rule for ties)
            leaving_row = -1
            min_ratio = float('inf')
            for i in range(self.m):
                if d[i] > PIVOT_TOL:
                    ratio = x_B[i] / d[i]
                    if ratio < min_ratio - PIVOT_TOL:
                        min_ratio = ratio
                        leaving_row = i
                    elif abs(ratio - min_ratio) <= PIVOT_TOL:
                        if self.basis[i] < self.basis[leaving_row]:
                            leaving_row = i

            if leaving_row == -1:
                return SolveStatus.UNBOUNDED

            # Pivot
            leaving_col = self.basis[leaving_row]
            pivot_element = d[leaving_row]
            self._update_basis_inverse(d, leaving_row)
            self.basis[leaving_row] = entering_col

            # Record snapshot
            x_B_new = self.B_inv @ self.b
            c_B_new = self.c_full[self.basis]
            obj_new = float(c_B_new @ x_B_new)

            self.iterations.append(IterationSnapshot(
                iteration=iteration,
                entering_var=self.all_var_names[entering_col],
                leaving_var=self.all_var_names[leaving_col],
                pivot_row=leaving_row,
                pivot_col=entering_col,
                pivot_element=pivot_element,
                objective_value=obj_new,
                basic_variables=tuple(self.all_var_names[j] for j in self.basis),
                entering_reduced_cost=entering_rc,
                leaving_ratio=min_ratio,
                basic_values=tuple(x_B_new),
            ))

        return SolveStatus.ITERATION_LIMIT

    def _solve_with_bigm(self, start_time: float) -> SolveState:
        """Handle negative RHS via Big-M: negate rows with b_i < 0, add artificials.

        For rows where b_i < 0, multiply the entire row by -1 (flips <= to >=,
        then slack is negative). We add an artificial variable for these rows.
        """
        BIG_M = 1e6
        neg_rows = [i for i in range(self.m) if self.b[i] < -PIVOT_TOL]

        # Negate rows with negative RHS so all b >= 0
        for i in neg_rows:
            self.A_full[i] *= -1
            self.b[i] *= -1

        # For negated rows, the slack variable starts negative (wrong sign).
        # We need artificial variables for these rows.
        n_art = len(neg_rows)
        if n_art == 0:
            # Shouldn't happen, but just in case
            return self._make_state(SolveStatus.INFEASIBLE, time.perf_counter() - start_time)

        # Extend the system with artificial variables
        old_N = self.N
        self.N += n_art
        self.all_var_names.extend([f"_art{k}" for k in range(n_art)])

        # Extend c_full: artificials get -BIG_M penalty (for max)
        c_ext = np.zeros(self.N)
        c_ext[:old_N] = self.c_full
        for k in range(n_art):
            c_ext[old_N + k] = -BIG_M
        self.c_full = c_ext

        # Extend A_full with artificial columns
        art_cols = np.zeros((self.m, n_art))
        for k, i in enumerate(neg_rows):
            art_cols[i, k] = 1.0
        self.A_full = np.hstack([self.A_full, art_cols])

        # Fix basis: for negated rows, replace slack with artificial
        # The slack for row i is at index (self.n + i) in the original system.
        # After negation, that slack has coefficient -1 (wrong sign).
        # The artificial has coefficient +1 (correct).
        for k, i in enumerate(neg_rows):
            self.basis[i] = old_N + k

        # Reset B_inv since basis changed
        # Extract basis matrix and compute inverse
        B = np.column_stack([self.A_full[:, j] for j in self.basis])
        try:
            self.B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return self._make_state(SolveStatus.INFEASIBLE, time.perf_counter() - start_time)

        # Run the shared simplex loop
        status = self._simplex_loop()

        # Check if any artificial variable is still in the basis with nonzero value
        x_B_final = self.B_inv @ self.b
        for i, j in enumerate(self.basis):
            if j >= old_N and abs(x_B_final[i]) > PIVOT_TOL:
                # Artificial still active → infeasible
                elapsed = time.perf_counter() - start_time
                return self._make_state(SolveStatus.INFEASIBLE, elapsed)

        # Remove artificials from system for clean state output
        # Replace any artificial still in basis (at zero) with a slack
        for i, j in enumerate(self.basis):
            if j >= old_N:
                # Find a slack not in basis to swap in
                for s in range(self.n, self.n + self.m):
                    if s not in self.basis:
                        self.basis[i] = s
                        break

        self.N = old_N
        self.c_full = self.c_full[:old_N]
        self.A_full = self.A_full[:, :old_N]
        self.all_var_names = self.all_var_names[:old_N]

        # Recompute B_inv for the clean basis
        B = np.column_stack([self.A_full[:, j] for j in self.basis])
        try:
            self.B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            pass  # keep existing B_inv

        elapsed = time.perf_counter() - start_time
        return self._make_state(status, elapsed)

    def solve_dual(self) -> SolveState:
        """Run Dual Simplex from current (dual-feasible) basis.

        Restores primal feasibility by pivoting out negative basic variables.
        Precondition: reduced costs ≤ 0 for all non-basic vars (dual feasible).
        """
        start_time = time.perf_counter()
        status = self._dual_simplex_loop()
        elapsed = time.perf_counter() - start_time
        return self._make_state(status, elapsed)

    def _dual_simplex_loop(self) -> SolveStatus:
        """Core dual simplex loop."""
        for iteration in range(1, MAX_ITERATIONS + 1):
            x_B = self.B_inv @ self.b

            # Find leaving variable: most negative x_B (Bland's rule for ties)
            leaving_row = -1
            min_val = -PIVOT_TOL
            for i in range(self.m):
                if x_B[i] < min_val:
                    min_val = x_B[i]
                    leaving_row = i
                elif leaving_row >= 0 and abs(x_B[i] - min_val) < PIVOT_TOL:
                    if self.basis[i] < self.basis[leaving_row]:
                        leaving_row = i

            if leaving_row == -1:
                return SolveStatus.OPTIMAL  # all x_B ≥ 0

            # Compute reduced costs and pivot row
            c_B = self.c_full[self.basis]
            y = c_B @ self.B_inv
            w = self.B_inv[leaving_row]  # pivot row of B⁻¹

            # Ratio test for entering variable
            entering_col = -1
            min_ratio = float('inf')
            for j in range(self.N):
                if j in self.basis:
                    continue
                d_j = w @ self.A_full[:, j]
                if d_j < -PIVOT_TOL:
                    rc_j = self.c_full[j] - y @ self.A_full[:, j]
                    ratio = rc_j / d_j
                    if ratio < min_ratio - PIVOT_TOL:
                        min_ratio = ratio
                        entering_col = j
                    elif abs(ratio - min_ratio) < PIVOT_TOL:
                        if entering_col < 0 or j < entering_col:
                            entering_col = j

            if entering_col == -1:
                return SolveStatus.INFEASIBLE  # dual unbounded

            # Pivot
            d = self.B_inv @ self.A_full[:, entering_col]
            leaving_col = self.basis[leaving_row]
            pivot_element = d[leaving_row]
            self._update_basis_inverse(d, leaving_row)
            self.basis[leaving_row] = entering_col

            # Record snapshot
            x_B_new = self.B_inv @ self.b
            c_B_new = self.c_full[self.basis]
            obj_new = float(c_B_new @ x_B_new)

            self.iterations.append(IterationSnapshot(
                iteration=iteration,
                entering_var=self.all_var_names[entering_col],
                leaving_var=self.all_var_names[leaving_col],
                pivot_row=leaving_row,
                pivot_col=entering_col,
                pivot_element=pivot_element,
                objective_value=obj_new,
                basic_variables=tuple(self.all_var_names[j] for j in self.basis),
                entering_reduced_cost=min_ratio,
                leaving_ratio=min_val,
                basic_values=tuple(x_B_new),
            ))

        return SolveStatus.ITERATION_LIMIT

    def _is_dual_feasible(self) -> bool:
        """Check if current basis is dual feasible (all rc ≤ 0 for max)."""
        c_B = self.c_full[self.basis]
        y = c_B @ self.B_inv
        for j in range(self.N):
            if j in self.basis:
                continue
            rc_j = self.c_full[j] - y @ self.A_full[:, j]
            if rc_j > OPTIMALITY_TOL:
                return False
        return True

    def _update_basis_inverse(self, d: np.ndarray, pivot_row: int) -> None:
        """Update B⁻¹ using elementary row operations (product form).

        E * B⁻¹_old = B⁻¹_new, where E is the eta matrix for the pivot.
        """
        pivot = d[pivot_row]
        # Scale pivot row
        self.B_inv[pivot_row] /= pivot
        # Eliminate other rows
        for i in range(self.m):
            if i != pivot_row:
                factor = d[i]
                self.B_inv[i] -= factor * self.B_inv[pivot_row]

    def _make_state(self, status: SolveStatus, elapsed: float) -> SolveState:
        """Build SolveState from current solver state."""
        n, m = self.n, self.m

        if status != SolveStatus.OPTIMAL:
            return SolveState(
                status=status,
                optimal_value=float('nan'),
                variables=(),
                constraints=(),
                sensitivity=SensitivityRanges({}, {}),
                engine=EngineType.INTERNAL_SIMPLEX,
                solve_time_seconds=elapsed,
                iteration_count=len(self.iterations),
                problem_name=self.problem.name,
                variable_names=tuple(self.problem.var_names),
                constraint_names=tuple(self.problem.constraint_names),
            )

        # Basic variable values
        x_B = self.B_inv @ self.b

        # Full solution vector (all N variables)
        x_full = np.zeros(self.N)
        for i, j in enumerate(self.basis):
            x_full[j] = x_B[i]

        # Optimal value (un-negate for minimize problems)
        optimal_value = float(self.c_full @ x_full)
        if self.problem.sense == "minimize":
            optimal_value = -optimal_value

        # Dual variables (shadow prices): y = c_B^T B⁻¹
        c_B = self.c_full[self.basis]
        y = c_B @ self.B_inv

        # Reduced costs for all variables
        rc_full = np.zeros(self.N)
        for j in range(self.N):
            if j in self.basis:
                rc_full[j] = 0.0
            else:
                rc_full[j] = self.c_full[j] - y @ self.A_full[:, j]

        # Sensitivity analysis
        obj_ranges = self._compute_obj_sensitivity(x_B, rc_full)
        rhs_ranges = self._compute_rhs_sensitivity()

        # Build VariableInfo for decision variables only
        basis_set = set(self.basis)
        variables = []
        for j in range(n):
            var_name = self.problem.var_names[j]
            variables.append(VariableInfo(
                name=var_name,
                value=float(x_full[j]),
                basis_status=BasisStatus.BASIC if j in basis_set else BasisStatus.NONBASIC_LOWER,
                reduced_cost=float(rc_full[j]),
                obj_coeff_range=obj_ranges.get(var_name, (float('-inf'), float('inf'))),
            ))

        # Build ConstraintInfo (only original constraints, not upper bound rows)
        orig_m = self.problem.num_constraints
        constraints = []
        for i in range(orig_m):
            con_name = self.problem.constraint_names[i]
            slack_val = float(x_full[n + i])
            constraints.append(ConstraintInfo(
                name=con_name,
                rhs=float(self.problem.b[i]) if i < len(self.problem.b) else 0.0,
                slack=slack_val,
                dual_value=float(y[i]),
                is_binding=abs(slack_val) < OPTIMALITY_TOL,
                basis_status=(
                    BasisStatus.BASIC if (n + i) in basis_set
                    else BasisStatus.NONBASIC_UPPER
                ),
                rhs_range=rhs_ranges.get(con_name, (float('-inf'), float('inf'))),
            ))

        sensitivity = SensitivityRanges(
            obj_coeff_ranges={v.name: v.obj_coeff_range for v in variables},
            rhs_ranges={c.name: c.rhs_range for c in constraints},
        )

        # Numerical diagnostics
        cond_num = float(np.linalg.cond(self.B_inv)) if self.B_inv.shape[0] > 0 else 0.0
        degen_count = int(np.sum(x_B < 1e-8))
        row_norms = np.linalg.norm(self.B_inv, axis=1)
        safe_norms = np.where(row_norms > 1e-12, row_norms, np.inf)
        d0 = float(np.min(x_B / safe_norms))

        return SolveState(
            status=SolveStatus.OPTIMAL,
            optimal_value=optimal_value,
            variables=tuple(variables),
            constraints=tuple(constraints),
            sensitivity=sensitivity,
            engine=EngineType.INTERNAL_SIMPLEX,
            solve_time_seconds=elapsed,
            iteration_count=len(self.iterations),
            basis_inverse=self.B_inv.copy(),
            iteration_history=tuple(self.iterations),
            condition_number=cond_num,
            degenerate_count=degen_count,
            basis_robustness_d0=d0,
            problem_name=self.problem.name,
            variable_names=tuple(self.problem.var_names),
            constraint_names=tuple(self.problem.constraint_names),
        )

    def _compute_obj_sensitivity(
        self, x_B: np.ndarray, rc_full: np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """Compute objective coefficient ranges for current basis.

        For basic variable x_j (at position r in basis):
            Δc_j can decrease until some non-basic reduced cost flips sign,
            or increase until some other non-basic reduced cost flips sign.

        For non-basic variable x_j:
            c_j can increase by at most |rc_j| before it enters the basis.
            c_j can decrease without limit (stays non-basic).
        """
        n, m = self.n, self.m
        ranges: dict[str, tuple[float, float]] = {}

        for j in range(n):
            var_name = self.problem.var_names[j]
            c_j = self.problem.c[j]

            if j in self.basis:
                # Basic variable: find range by examining effect on reduced costs
                basis_pos = self.basis.index(j)
                e_r = self.B_inv[basis_pos]  # row of B⁻¹

                lower = float('-inf')
                upper = float('inf')

                for k in range(self.N):
                    if k in self.basis:
                        continue
                    # rc_k = c_k - c_B^T B⁻¹ a_k
                    # If c_j changes by Δ, rc_k changes by -Δ * e_r^T a_k
                    a_k = self.A_full[:, k]
                    coeff = e_r @ a_k

                    if abs(coeff) < PIVOT_TOL:
                        continue

                    # For max: we need rc_k <= 0 to stay optimal
                    # rc_k(Δ) = rc_full[k] - Δ * coeff <= 0
                    # ⟹ Δ * coeff >= rc_full[k]
                    # coeff > 0: Δ >= rc_full[k] / coeff → lower bound
                    # coeff < 0: Δ <= rc_full[k] / coeff → upper bound
                    limit = rc_full[k] / coeff
                    if coeff > 0:
                        lower = max(lower, limit)
                    else:
                        upper = min(upper, limit)

                ranges[var_name] = (c_j + lower, c_j + upper)
            else:
                # Non-basic: can decrease without limit; increase by rc_j
                # rc_j = c_j - y^T a_j; if c_j increases, rc_j increases
                # Enters basis when rc_j > 0 (for max), i.e., Δ > -rc_full[j]
                # Since it's non-basic optimal: rc_full[j] <= 0
                ranges[var_name] = (float('-inf'), c_j - rc_full[j])

        return ranges

    def _compute_rhs_sensitivity(self) -> dict[str, tuple[float, float]]:
        """Compute RHS ranges for current basis.

        For constraint i, the RHS b_i can change by Δ while keeping
        all basic variables non-negative: x_B = B⁻¹(b + Δe_i) >= 0.

        This means B⁻¹[:,i] * Δ + x_B >= 0 for all rows.
        """
        x_B = self.B_inv @ self.b
        ranges: dict[str, tuple[float, float]] = {}

        orig_m = len(self.problem.constraint_names)
        for i in range(min(self.m, orig_m)):
            con_name = self.problem.constraint_names[i]
            col = self.B_inv[:, i]  # i-th column of B⁻¹

            lower = float('-inf')
            upper = float('inf')

            for r in range(self.m):
                if abs(col[r]) < PIVOT_TOL:
                    continue

                # x_B[r] + col[r] * Δ >= 0
                # Δ >= -x_B[r] / col[r]  if col[r] > 0
                # Δ <= -x_B[r] / col[r]  if col[r] < 0
                limit = -x_B[r] / col[r]
                if col[r] > 0:
                    lower = max(lower, limit)
                else:
                    upper = min(upper, limit)

            b_i = float(self.problem.b[i])
            ranges[con_name] = (b_i + lower, b_i + upper)

        return ranges


def solve(problem: LPProblem) -> SolveState:
    """Convenience function: solve an LP using Revised Simplex.

    Args:
        problem: The LP problem to solve.

    Returns:
        SolveState with full internal trace.
    """
    solver = RevisedSimplex(problem)
    return solver.solve()
