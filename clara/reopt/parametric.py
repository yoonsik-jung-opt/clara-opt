"""Parametric LP solver for Type RC compound changes.

Traces the optimal path from old (θ=0) to new (θ=1) problem:
    max (c₀ + θΔc)ᵀx  s.t.  Ax ≤ b₀ + θΔb,  x ≥ 0

At each breakpoint θ*, the basis changes — the solution structure
qualitatively shifts. Between breakpoints, the solution varies linearly.

Theory: Gal & Greenberg (1997), Advances in Sensitivity Analysis.
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
from clara.reopt.types import ParametricBreakpoint, ParametricResult

PIVOT_TOL = 1e-10
OPTIMALITY_TOL = 1e-8
MAX_PIVOTS = 100


class ParametricLPSolver:
    """Trace the optimal path from old to new problem via parametric LP."""

    def solve(
        self,
        old_state: SolveState,
        old_problem: LPProblem,
        delta_b: np.ndarray,
        delta_c: np.ndarray,
    ) -> ParametricResult:
        """Trace the parametric path from θ=0 to θ=1.

        Args:
            old_state: Solved state at θ=0 (must have B⁻¹).
            old_problem: The original problem.
            delta_b: RHS change vector.
            delta_c: Objective change vector.
        """
        start = time.perf_counter()
        from clara.engine.simplex import RevisedSimplex as _RS
        _, n = old_problem.A.shape

        # Build augmented system matching B_inv dimensions (includes UB rows)
        A_aug, b0, _ = _RS._add_upper_bound_rows(old_problem)
        m = A_aug.shape[0]  # augmented constraint count
        N = n + m

        A_full = np.hstack([A_aug, np.eye(m)])

        # Full cost vectors (decision + slack)
        c0_full = np.zeros(N)
        c0_full[:n] = old_problem.c
        dc_full = np.zeros(N)
        dc_full[:n] = delta_c

        # Pad delta_b to augmented size (UB rows have delta=0)
        db_orig = np.asarray(delta_b, dtype=float)
        db = np.zeros(m)
        db[:len(db_orig)] = db_orig

        # Extract basis from old state
        B_inv = old_state.basis_inverse.copy()
        basis = self._extract_basis(old_state, old_problem)

        var_names = list(old_problem.var_names) + [f"y{i+1}" for i in range(m)]
        breakpoints: list[ParametricBreakpoint] = []
        current_theta = 0.0
        prev_obj = old_state.optimal_value
        monotone = True

        for _ in range(MAX_PIVOTS):
            if current_theta >= 1.0 - PIVOT_TOL:
                break

            # Find next breakpoint
            theta_p, leaving_row = self._find_primal_breakpoint(
                current_theta, B_inv, b0, db, m
            )
            theta_d, entering_col = self._find_dual_breakpoint(
                current_theta, B_inv, A_full, c0_full, dc_full, basis, N
            )

            theta_next = min(theta_p, theta_d)
            if theta_next >= 1.0 - PIVOT_TOL:
                break  # No breakpoint before θ=1

            theta_next = min(theta_next, 1.0)

            # Compute objective at breakpoint
            b_theta = b0 + theta_next * db
            c_theta = c0_full[:n] + theta_next * dc_full[:n]
            x_B = B_inv @ b_theta
            c_B = np.array([c0_full[j] + theta_next * dc_full[j] for j in basis])
            obj_at_bp = float(c_B @ x_B)

            if obj_at_bp < prev_obj - OPTIMALITY_TOL:
                monotone = False
            prev_obj = obj_at_bp

            # Pivot
            if theta_next == theta_p and leaving_row >= 0:
                # Primal breakpoint: basic var hits 0 → dual ratio test
                leaving_var = var_names[basis[leaving_row]]
                entering_col_bp, entering_var = self._dual_ratio_test(
                    leaving_row, B_inv, A_full, c0_full, dc_full, basis, theta_next, N, var_names
                )
                if entering_col_bp < 0:
                    break  # Infeasible

                d = B_inv @ A_full[:, entering_col_bp]
                self._update_basis_inverse(B_inv, d, leaving_row, m)
                basis[leaving_row] = entering_col_bp
                bp_type = "primal"
            else:
                # Dual breakpoint: reduced cost hits 0 → primal ratio test
                entering_var = var_names[entering_col]
                leaving_row_bp, leaving_var = self._primal_ratio_test(
                    entering_col, B_inv, A_full, b0, db, basis, theta_next, m, var_names
                )
                if leaving_row_bp < 0:
                    break  # Unbounded

                d = B_inv @ A_full[:, entering_col]
                self._update_basis_inverse(B_inv, d, leaving_row_bp, m)
                basis[leaving_row_bp] = entering_col
                bp_type = "dual"

            explanation = self._make_explanation(
                bp_type, theta_next, leaving_var, entering_var, var_names, basis
            )

            breakpoints.append(ParametricBreakpoint(
                theta=theta_next,
                breakpoint_type=bp_type,
                leaving_var=leaving_var,
                entering_var=entering_var,
                objective_value=obj_at_bp,
                basic_variables=tuple(var_names[j] for j in basis),
                explanation=explanation,
            ))

            current_theta = theta_next

        # Build final state at θ=1
        new_state = self._build_final_state(
            basis, B_inv, old_problem, b0, db, c0_full, dc_full, n, m, N,
            time.perf_counter() - start, len(breakpoints),
        )

        elapsed = time.perf_counter() - start
        return ParametricResult(
            new_state=new_state,
            breakpoints=tuple(breakpoints),
            theta_start=0.0,
            theta_end=1.0,
            num_pivots=len(breakpoints),
            path_monotone=monotone,
            solve_time_seconds=elapsed,
        )

    def _find_primal_breakpoint(
        self, current_theta: float, B_inv: np.ndarray,
        b0: np.ndarray, db: np.ndarray, m: int,
    ) -> tuple[float, int]:
        """Find θ where a basic variable hits 0."""
        b_theta = b0 + current_theta * db
        x_B = B_inv @ b_theta
        slope = B_inv @ db

        best_theta = float("inf")
        leaving_row = -1

        for i in range(m):
            if slope[i] < -PIVOT_TOL:
                theta_i = current_theta + (-x_B[i] / slope[i])
                if theta_i > current_theta + PIVOT_TOL and theta_i < best_theta:
                    best_theta = theta_i
                    leaving_row = i

        return best_theta, leaving_row

    def _find_dual_breakpoint(
        self, current_theta: float, B_inv: np.ndarray,
        A_full: np.ndarray, c0_full: np.ndarray, dc_full: np.ndarray,
        basis: list[int], N: int,
    ) -> tuple[float, int]:
        """Find θ where a reduced cost hits 0 (from negative toward positive)."""
        c_B = np.array([c0_full[j] + current_theta * dc_full[j] for j in basis])
        dc_B = np.array([dc_full[j] for j in basis])
        y = c_B @ B_inv
        dy = dc_B @ B_inv

        best_theta = float("inf")
        entering_col = -1

        for j in range(N):
            if j in basis:
                continue
            a_j = A_full[:, j]
            c_j_theta = c0_full[j] + current_theta * dc_full[j]
            rc_j = c_j_theta - y @ a_j
            rc_slope = dc_full[j] - dy @ a_j

            if rc_slope > PIVOT_TOL and rc_j < -PIVOT_TOL:
                theta_j = current_theta + (-rc_j / rc_slope)
                if theta_j > current_theta + PIVOT_TOL and theta_j < best_theta:
                    best_theta = theta_j
                    entering_col = j

        return best_theta, entering_col

    def _dual_ratio_test(
        self, leaving_row: int, B_inv: np.ndarray, A_full: np.ndarray,
        c0_full: np.ndarray, dc_full: np.ndarray, basis: list[int],
        theta: float, N: int, var_names: list[str],
    ) -> tuple[int, str]:
        """Dual ratio test: select entering variable at primal breakpoint."""
        w = B_inv[leaving_row]
        c_B = np.array([c0_full[j] + theta * dc_full[j] for j in basis])
        y = c_B @ B_inv

        best_ratio = float("inf")
        entering = -1

        for j in range(N):
            if j in basis:
                continue
            d_j = w @ A_full[:, j]
            if d_j < -PIVOT_TOL:
                c_j = c0_full[j] + theta * dc_full[j]
                rc_j = c_j - y @ A_full[:, j]
                ratio = rc_j / d_j
                if ratio < best_ratio - PIVOT_TOL:
                    best_ratio = ratio
                    entering = j
                elif abs(ratio - best_ratio) < PIVOT_TOL and (entering < 0 or j < entering):
                    entering = j

        name = var_names[entering] if entering >= 0 else ""
        return entering, name

    def _primal_ratio_test(
        self, entering_col: int, B_inv: np.ndarray, A_full: np.ndarray,
        b0: np.ndarray, db: np.ndarray, basis: list[int],
        theta: float, m: int, var_names: list[str],
    ) -> tuple[int, str]:
        """Primal ratio test: select leaving variable at dual breakpoint."""
        d = B_inv @ A_full[:, entering_col]
        b_theta = b0 + theta * db
        x_B = B_inv @ b_theta

        best_ratio = float("inf")
        leaving = -1

        for i in range(m):
            if d[i] > PIVOT_TOL:
                ratio = x_B[i] / d[i]
                if ratio < best_ratio - PIVOT_TOL:
                    best_ratio = ratio
                    leaving = i
                elif abs(ratio - best_ratio) < PIVOT_TOL:
                    if leaving < 0 or basis[i] < basis[leaving]:
                        leaving = i

        name = var_names[basis[leaving]] if leaving >= 0 else ""
        return leaving, name

    def _update_basis_inverse(
        self, B_inv: np.ndarray, d: np.ndarray, pivot_row: int, m: int
    ) -> None:
        """Standard eta update (same as RevisedSimplex)."""
        pivot = d[pivot_row]
        B_inv[pivot_row] /= pivot
        for i in range(m):
            if i != pivot_row:
                B_inv[i] -= d[i] * B_inv[pivot_row]

    def _make_explanation(
        self, bp_type: str, theta: float,
        leaving: str, entering: str,
        var_names: list[str], basis: list[int],
    ) -> str:
        pct = theta * 100
        if bp_type == "primal":
            return (
                f"At θ = {theta:.4f} ({pct:.1f}%), {leaving} hits zero and "
                f"leaves the basis. {entering} enters."
            )
        return (
            f"At θ = {theta:.4f} ({pct:.1f}%), reduced cost of {entering} "
            f"reaches zero — enters basis. {leaving} leaves."
        )

    def _extract_basis(self, state: SolveState, problem: LPProblem) -> list[int]:
        """Extract basis indices by matching B⁻¹ columns to [A|I]."""
        from clara.engine.simplex import RevisedSimplex as _RS
        n = problem.num_variables
        B_inv = state.basis_inverse
        B = np.linalg.inv(B_inv)
        A_aug, _, _ = _RS._add_upper_bound_rows(problem)
        m_aug = A_aug.shape[0]
        A_full = np.hstack([A_aug, np.eye(m_aug)])

        basis = []
        for col_idx in range(m_aug):
            b_col = B[:, col_idx]
            best_j = min(range(n + m_aug), key=lambda j: np.linalg.norm(A_full[:, j] - b_col))
            basis.append(best_j)
        return basis

    def _build_final_state(
        self, basis, B_inv, problem, b0, db, c0_full, dc_full,
        n, m, N, elapsed, num_breakpoints,
    ) -> SolveState:
        """Build SolveState at θ=1."""
        b_final = b0 + db
        c_final = c0_full[:n] + dc_full[:n]
        x_B = B_inv @ b_final

        x_full = np.zeros(N)
        for i, j in enumerate(basis):
            x_full[j] = x_B[i]

        opt_val = float(c_final @ x_full[:n])

        c_full_final = np.zeros(N)
        c_full_final[:n] = c_final
        c_B = np.array([c_full_final[j] for j in basis])
        y = c_B @ B_inv

        basis_set = set(basis)
        A_full = np.hstack([problem.A, np.eye(m)])

        variables = []
        for j in range(n):
            rc = 0.0 if j in basis_set else float(c_full_final[j] - y @ A_full[:, j])
            variables.append(VariableInfo(
                name=problem.var_names[j],
                value=float(x_full[j]),
                basis_status=BasisStatus.BASIC if j in basis_set else BasisStatus.NONBASIC_LOWER,
                reduced_cost=rc,
                obj_coeff_range=(float("-inf"), float("inf")),
            ))

        constraints = []
        for i in range(m):
            slack = float(x_full[n + i])
            constraints.append(ConstraintInfo(
                name=problem.constraint_names[i],
                rhs=float(b_final[i]),
                slack=slack,
                dual_value=float(y[i]),
                is_binding=abs(slack) < 1e-8,
                basis_status=BasisStatus.BASIC if (n + i) in basis_set else BasisStatus.NONBASIC_UPPER,
                rhs_range=(float("-inf"), float("inf")),
            ))

        return SolveState(
            status=SolveStatus.OPTIMAL,
            optimal_value=opt_val,
            variables=tuple(variables),
            constraints=tuple(constraints),
            sensitivity=SensitivityRanges({}, {}),
            engine=EngineType.INTERNAL_SIMPLEX,
            solve_time_seconds=elapsed,
            iteration_count=num_breakpoints,
            basis_inverse=B_inv.copy(),
            problem_name=problem.name,
            variable_names=tuple(problem.var_names),
            constraint_names=tuple(problem.constraint_names),
        )
