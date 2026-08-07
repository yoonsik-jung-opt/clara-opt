"""MIP opportunity cost bound analyzer.

Computes LP-relaxation bound (Theorem 2):
    OC ≤ z'*_LP - c'^T x*

where z'*_LP is the changed LP relaxation optimal and x* is the old MIP solution.
Enables automatic skip/reoptimize decisions for MIP parameter changes.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from clara.engine.highs_backend import HiGHSBackend
from clara.model.problem import LPProblem
from clara.model.solve_state import SolveState
from clara.reopt.types import MIPBoundResult, ParameterChange


class MIPBoundAnalyzer:
    """Compute opportunity cost bounds for MIP reoptimization decisions.

    Args:
        epsilon: Threshold for skip decision. If bound_ratio ≤ epsilon, skip.
    """

    def __init__(self, epsilon: float = 0.05) -> None:
        self.epsilon = epsilon

    def analyze(
        self,
        mip_state: SolveState,
        lp_state: SolveState,
        old_problem: LPProblem,
        new_problem: LPProblem,
        change: ParameterChange,
    ) -> MIPBoundResult:
        """Analyze opportunity cost of not reoptimizing a MIP.

        Args:
            mip_state: Original MIP solution (x*).
            lp_state: Original LP relaxation solution (x̄).
            old_problem: Original problem.
            new_problem: Changed problem.
            change: Detected parameter change.

        Returns:
            MIPBoundResult with bound, decision, and supporting evidence.
        """
        is_min = old_problem.sense == "minimize"
        n = old_problem.num_variables
        m = old_problem.num_constraints

        # Extract solutions
        x_mip = np.array([v.value for v in mip_state.variables[:n]])
        z_mip = mip_state.optimal_value
        z_lp = lp_state.optimal_value

        # Integrality gap
        gap = abs(z_lp - z_mip)
        gap_ratio = gap / max(abs(z_lp), 1e-10)

        # New objective evaluated at old MIP solution
        c_new = new_problem.c[:n]
        c_new_x_old = float(c_new @ x_mip)
        if is_min:
            c_new_x_old_val = c_new_x_old  # minimize: higher = worse
        else:
            c_new_x_old_val = c_new_x_old

        # Feasibility check
        feasible = True
        max_violation = 0.0
        if change.delta_b is not None:
            b_new = new_problem.b[:m]
            Ax = old_problem.A[:m, :n] @ x_mip
            violations = Ax - b_new  # positive = violated for <=
            max_violation = float(np.max(violations)) if len(violations) > 0 else 0.0
            if max_violation > 1e-6:
                feasible = False

        if not feasible:
            return MIPBoundResult(
                decision="infeasible",
                reason=f"Old MIP solution violates constraints (max violation: {max_violation:.4f})",
                bound=float("inf"), bound_ratio=float("inf"),
                bound_corrected=None, bound_corrected_ratio=None,
                original_integrality_gap=gap, gap_ratio=gap_ratio,
                z_mip_old=z_mip, z_lp_old=z_lp, z_lp_new=None,
                c_new_x_old=c_new_x_old_val,
                old_solution_feasible=False, max_violation=max_violation,
                within_lp_sensitivity=False, lp_solve_needed=False,
                oguz_relative_bound=None, delta=None,
            )

        # LP sensitivity check
        within_sens = self._check_lp_sensitivity(lp_state, change, old_problem)

        # Get z'*_LP
        z_lp_new: Optional[float] = None
        lp_solve_needed = False

        if within_sens:
            # LP basis unchanged: evaluate new c at old LP solution
            x_lp = np.array([v.value for v in lp_state.variables[:n]])
            z_lp_new = float(c_new @ x_lp)
            if is_min:
                pass  # minimize: z_lp_new is the value under new c
            # Note: for RHS changes within sensitivity, z_lp_new = old LP obj
            # with recomputed x_B. Simplified: use c_new^T x_lp_old
        else:
            # Need to solve changed LP relaxation
            lp_solve_needed = True
            lp_problem = new_problem.as_lp()
            try:
                lp_new_state = HiGHSBackend().solve(lp_problem)
                if lp_new_state.is_optimal:
                    z_lp_new = lp_new_state.optimal_value
            except Exception:
                pass

        if z_lp_new is None:
            # Couldn't get LP relaxation value — conservative reoptimize
            return MIPBoundResult(
                decision="reoptimize",
                reason="Could not solve changed LP relaxation",
                bound=float("inf"), bound_ratio=float("inf"),
                bound_corrected=None, bound_corrected_ratio=None,
                original_integrality_gap=gap, gap_ratio=gap_ratio,
                z_mip_old=z_mip, z_lp_old=z_lp, z_lp_new=None,
                c_new_x_old=c_new_x_old_val,
                old_solution_feasible=feasible, max_violation=max_violation,
                within_lp_sensitivity=within_sens, lp_solve_needed=lp_solve_needed,
                oguz_relative_bound=None, delta=None,
            )

        # Compute bound B = z'*_LP - c'^T x*
        if is_min:
            # For min: OC = c'^T x* - z'*_MIP, bound B = c'^T x* - z'*_LP
            bound = c_new_x_old_val - z_lp_new
        else:
            # For max: OC = z'*_MIP - c'^T x*, bound B = z'*_LP - c'^T x*
            bound = z_lp_new - c_new_x_old_val

        bound = max(bound, 0.0)  # bound should be non-negative
        bound_ratio = bound / max(abs(z_lp_new), 1e-10)

        # Gap-corrected estimate (Proposition 6)
        gap_est = gap * abs(z_lp_new) / max(abs(z_lp), 1e-10)
        bound_corrected = max(bound - gap_est, 0.0)
        bound_corrected_ratio = bound_corrected / max(abs(z_lp_new), 1e-10)

        # Oguz bound
        oguz, delta_val = self._compute_oguz_bound(change, old_problem)

        # Decision
        threshold_bound = bound_corrected if bound_corrected is not None else bound
        threshold_ratio = threshold_bound / max(abs(z_lp_new), 1e-10)

        if threshold_ratio <= self.epsilon:
            decision = "skip"
            reason = (f"Bound ratio {threshold_ratio:.1%} ≤ threshold {self.epsilon:.0%}. "
                      f"Max opportunity cost is small.")
        else:
            decision = "reoptimize"
            reason = (f"Bound ratio {threshold_ratio:.1%} > threshold {self.epsilon:.0%}. "
                      f"Reoptimization recommended.")

        return MIPBoundResult(
            decision=decision, reason=reason,
            bound=bound, bound_ratio=bound_ratio,
            bound_corrected=bound_corrected, bound_corrected_ratio=bound_corrected_ratio,
            original_integrality_gap=gap, gap_ratio=gap_ratio,
            z_mip_old=z_mip, z_lp_old=z_lp, z_lp_new=z_lp_new,
            c_new_x_old=c_new_x_old_val,
            old_solution_feasible=feasible, max_violation=max_violation,
            within_lp_sensitivity=within_sens, lp_solve_needed=lp_solve_needed,
            oguz_relative_bound=oguz, delta=delta_val,
        )

    def _check_lp_sensitivity(
        self, lp_state: SolveState, change: ParameterChange, old_problem: LPProblem
    ) -> bool:
        """Check if all changes fall within LP sensitivity ranges."""
        if change.delta_c is not None:
            for j, var_name in enumerate(old_problem.var_names):
                if j < len(change.delta_c) and abs(change.delta_c[j]) > 1e-10:
                    if var_name not in lp_state.sensitivity.obj_coeff_ranges:
                        return False
                    lo, hi = lp_state.sensitivity.obj_coeff_ranges[var_name]
                    new_c = old_problem.c[j] + change.delta_c[j]
                    if new_c < lo - 1e-8 or new_c > hi + 1e-8:
                        return False

        if change.delta_b is not None:
            for i, con_name in enumerate(old_problem.constraint_names):
                if i < len(change.delta_b) and abs(change.delta_b[i]) > 1e-10:
                    if con_name not in lp_state.sensitivity.rhs_ranges:
                        return False
                    lo, hi = lp_state.sensitivity.rhs_ranges[con_name]
                    new_b = old_problem.b[i] + change.delta_b[i]
                    if new_b < lo - 1e-8 or new_b > hi + 1e-8:
                        return False

        return True

    def _compute_oguz_bound(
        self, change: ParameterChange, old_problem: LPProblem
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute Oguz bound 2δ/(1+δ) if applicable."""
        if change.delta_c is None:
            return None, None

        delta = 0.0
        for j in range(len(old_problem.var_names)):
            if j < len(change.delta_c) and abs(change.delta_c[j]) > 1e-10:
                if abs(old_problem.c[j]) < 1e-10:
                    return None, None  # can't compute relative change
                rel = abs(change.delta_c[j] / old_problem.c[j])
                delta = max(delta, rel)

        if delta == 0:
            return 0.0, 0.0

        oguz = 2 * delta / (1 + delta)
        return oguz, delta
