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
        m, n = problem.A.shape
        self.m = m  # number of constraints
        self.n = n  # number of decision variables
        self.N = n + m  # total variables (decision + slack)

        # Full variable names: x1..xn, y1..ym (slacks)
        self.all_var_names = list(problem.var_names) + [
            f"y{i+1}" for i in range(m)
        ]

        # Objective: decision vars have cost c_j, slacks have cost 0
        self.c_full = np.zeros(self.N)
        self.c_full[:n] = problem.c

        # Constraint matrix: [A | I]
        self.A_full = np.hstack([problem.A, np.eye(m)])

        # RHS
        self.b = problem.b.copy()

        # Initial basis: slack variables (indices n, n+1, ..., n+m-1)
        self.basis = list(range(n, n + m))

        # B⁻¹ starts as identity (basis = slacks)
        self.B_inv = np.eye(m)

        # History
        self.iterations: list[IterationSnapshot] = []

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

        # Check initial feasibility (b >= 0 required for standard form)
        if np.any(self.b < -PIVOT_TOL):
            return self._make_state(SolveStatus.INFEASIBLE, time.perf_counter() - start_time)

        status = SolveStatus.OPTIMAL
        for iteration in range(1, MAX_ITERATIONS + 1):
            # Step 1: Compute basic variable values  x_B = B⁻¹ b
            x_B = self.B_inv @ self.b

            # Step 2: Compute reduced costs for all non-basic variables
            # c̄_j = c_j - c_B^T B⁻¹ a_j
            c_B = self.c_full[self.basis]
            y = c_B @ self.B_inv  # dual variables (simplex multipliers)

            # Find entering variable (Bland's rule: smallest index with negative reduced cost)
            # For maximization: enter if reduced cost > 0 (can improve objective)
            entering_col = -1
            entering_rc = 0.0
            for j in range(self.N):
                if j in self.basis:
                    continue
                rc_j = self.c_full[j] - y @ self.A_full[:, j]
                if rc_j > OPTIMALITY_TOL:
                    # Bland's rule: take the first eligible variable by index
                    entering_col = j
                    entering_rc = rc_j
                    break

            if entering_col == -1:
                # No improving variable → optimal
                status = SolveStatus.OPTIMAL
                break

            # Step 3: Compute direction  d = B⁻¹ a_entering
            a_entering = self.A_full[:, entering_col]
            d = self.B_inv @ a_entering

            # Step 4: Minimum ratio test (Bland's rule for ties: smallest basis index)
            leaving_row = -1
            min_ratio = float('inf')
            for i in range(self.m):
                if d[i] > PIVOT_TOL:
                    ratio = x_B[i] / d[i]
                    if ratio < min_ratio - PIVOT_TOL:
                        min_ratio = ratio
                        leaving_row = i
                    elif abs(ratio - min_ratio) <= PIVOT_TOL:
                        # Bland's rule tie-breaking: prefer row with smaller basis index
                        if self.basis[i] < self.basis[leaving_row]:
                            leaving_row = i

            if leaving_row == -1:
                status = SolveStatus.UNBOUNDED
                break

            # Record snapshot before pivot
            leaving_col = self.basis[leaving_row]
            pivot_element = d[leaving_row]

            # Step 5: Update B⁻¹ via elementary row operations
            self._update_basis_inverse(d, leaving_row)

            # Update basis
            self.basis[leaving_row] = entering_col

            # Compute new x_B and objective for snapshot
            x_B_new = self.B_inv @ self.b
            c_B_new = self.c_full[self.basis]
            obj_new = float(c_B_new @ x_B_new)

            snapshot = IterationSnapshot(
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
            )
            self.iterations.append(snapshot)
        else:
            status = SolveStatus.ITERATION_LIMIT

        elapsed = time.perf_counter() - start_time
        return self._make_state(status, elapsed)

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

        # Optimal value
        optimal_value = float(self.c_full @ x_full)

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

        # Build ConstraintInfo
        constraints = []
        for i in range(m):
            con_name = self.problem.constraint_names[i]
            slack_val = float(x_full[n + i])
            constraints.append(ConstraintInfo(
                name=con_name,
                rhs=float(self.problem.b[i]),
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

        for i in range(self.m):
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
