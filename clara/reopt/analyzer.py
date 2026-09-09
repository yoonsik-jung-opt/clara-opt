"""Impact Analyzer — decides whether reoptimization is needed.

Three-step decision pipeline:
    Step 1: Sensitivity range check — does the change break the current basis?
            (a pass is executed through the certified recompute path)
    Step 2: Oguz bound (Type C only) — certified upper bound on the
            opportunity cost, tightened with the Wendell objective tolerance
    Step 3: Reoptimization method recommendation

References:
    Oguz (2000), Management Science 46(7) — 2θ/(1+θ) bound
    Wendell (1985), Management Science 31 — simultaneous tolerance
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from clara.model.problem import LPProblem
from clara.model.solve_state import SolveState
from clara.reopt.types import ChangeType, ParameterChange, ReoptDecision


class ImpactAnalyzer:
    """Decide whether reoptimization is needed and recommend a method.

    Uses mathematical guarantees (Oguz bound, sensitivity ranges)
    rather than heuristics wherever possible.

    Args:
        oguz_threshold: If Oguz bound < this, recommend skipping reoptimization.
            Default 5% — "losing at most 5% is acceptable."
        force_reopt_types: Change types that always require full re-solve.
    """

    def __init__(
        self,
        oguz_threshold: float = 0.05,
        force_reopt_types: Optional[set[ChangeType]] = None,
    ) -> None:
        self.oguz_threshold = oguz_threshold
        self.force_reopt_types = force_reopt_types or {
            ChangeType.TYPE_V, ChangeType.TYPE_X,
            ChangeType.TYPE_M, ChangeType.TYPE_MULTI,
        }

    def analyze(
        self,
        state: SolveState,
        change: ParameterChange,
        old_problem: LPProblem,
    ) -> ReoptDecision:
        """Analyze the impact of a parameter change on the current solution.

        Args:
            state: Current SolveState (with sensitivity ranges).
            change: Detected parameter change (from ChangeDetector).
            old_problem: Original problem (for objective coefficients).

        Returns:
            ReoptDecision with recommendation and supporting evidence.
        """
        # Structural changes → always re-solve
        if change.change_type in self.force_reopt_types:
            return ReoptDecision(
                should_reoptimize=True,
                reason=f"{change.summary}. Structural change requires full re-solve.",
                recommended_method="scratch",
            )

        # Type C: skip candidates (all changes within the objective
        # sensitivity ranges, or Oguz bound below threshold) are routed
        # through the certified recompute path, which verifies dual
        # feasibility of the retained basis under the new objective and
        # falls back to a warm start otherwise.  One-at-a-time ranges do
        # not certify simultaneous changes, and the Oguz proxy is a
        # heuristic, so neither may keep a stale solution on its own.
        if change.change_type == ChangeType.TYPE_C:
            raw, tightened, oguz_expl = self._oguz_bound(change, old_problem, state)
            within, violated, sens_expl = self._sensitivity_check(state, change, old_problem)

            if within:
                return ReoptDecision(
                    should_reoptimize=False,
                    reason=f"All objective coefficient changes within sensitivity ranges. {oguz_expl}",
                    oguz_bound=tightened,
                    within_sensitivity=True,
                    recommended_method="recompute",
                )

            if tightened < self.oguz_threshold:
                return ReoptDecision(
                    should_reoptimize=False,
                    reason=(
                        f"Oguz bound {tightened:.1%} below threshold {self.oguz_threshold:.0%}. "
                        f"Max opportunity cost is small. {oguz_expl}"
                    ),
                    oguz_bound=tightened,
                    within_sensitivity=False,
                    recommended_method="recompute",
                )

            return ReoptDecision(
                should_reoptimize=True,
                reason=f"Sensitivity ranges violated for: {', '.join(violated)}. {oguz_expl}",
                oguz_bound=tightened,
                within_sensitivity=False,
                recommended_method="warm_start",
            )

        # Type R: sensitivity check only
        if change.change_type == ChangeType.TYPE_R:
            within, violated, sens_expl = self._sensitivity_check(state, change, old_problem)

            if within:
                return ReoptDecision(
                    should_reoptimize=False,
                    reason=(
                        f"All RHS changes within sensitivity ranges. "
                        f"Recompute x_B = B⁻¹ * b_new (no pivots). {sens_expl}"
                    ),
                    within_sensitivity=True,
                    recommended_method="recompute",
                )

            return ReoptDecision(
                should_reoptimize=True,
                reason=f"RHS sensitivity ranges violated for: {', '.join(violated)}. {sens_expl}",
                within_sensitivity=False,
                recommended_method="warm_start",
            )

        # Type RC: check both, recommend parametric if violated
        if change.change_type == ChangeType.TYPE_RC:
            within, violated, sens_expl = self._sensitivity_check(state, change, old_problem)

            if within:
                return ReoptDecision(
                    should_reoptimize=False,
                    reason=f"All changes within sensitivity ranges (both RHS and obj). {sens_expl}",
                    within_sensitivity=True,
                    recommended_method="recompute",
                )

            return ReoptDecision(
                should_reoptimize=True,
                reason=(
                    f"Compound change — sensitivity violated for: {', '.join(violated)}. "
                    f"Parametric LP recommended for accurate path tracing. {sens_expl}"
                ),
                within_sensitivity=False,
                recommended_method="parametric_lp",
            )

        # Type A or unknown → warm_start
        return ReoptDecision(
            should_reoptimize=True,
            reason=f"{change.summary}. Reoptimization recommended.",
            recommended_method="warm_start",
        )

    def _oguz_bound(
        self,
        change: ParameterChange,
        old_problem: LPProblem,
        state: SolveState,
    ) -> tuple[float, float, str]:
        """Oguz bound on the relative opportunity cost of keeping x*.

        Let theta = max_j |dc_j| / |c_j| be the componentwise maximum
        relative change of the objective coefficients.  For a program
        with nonnegative objective coefficients whose feasible set is
        unchanged, the opportunity cost of not reoptimizing satisfies
        (Oguz, 2000)

            OC / |z'*| <= 2 theta / (1 + theta)   (maximize)
            OC / |z'*| <= 2 theta / (1 - theta)   (minimize, theta < 1).

        LP tightening.  Let tau be the Wendell (1985) objective
        tolerance of the retained basis: every coefficient may move
        simultaneously by at most tau |c_j| with the basis still
        optimal.  Clipping each relative change to [-tau, tau] gives an
        objective c'' for which x* is optimal; relative to c'', the new
        objective c' deviates upward by at most (theta - tau)/(1 + tau)
        and downward by at most (theta - tau)/(1 - tau), and Oguz's
        argument with base c'' yields

            OC / |z'*| <= 2 (theta - tau) / ((1 - tau)(1 + theta))   (max)
            OC / |z'*| <= 2 (theta - tau) / ((1 + tau)(1 - theta))   (min)

        and 0 when theta <= tau.  The bound is certified; it is not
        applicable (returned as 1.0) when some objective coefficient is
        negative or a zero coefficient is perturbed.

        Returns:
            (raw_bound, tightened_bound, explanation)
        """
        c = np.asarray(old_problem.c, dtype=float)
        n = len(old_problem.var_names)
        dc = change.delta_c if change.delta_c is not None else np.zeros(n)
        is_max = old_problem.sense != "minimize"

        if np.any(c < -1e-12):
            return 1.0, 1.0, ("Oguz bound not applicable: objective has negative "
                              "coefficients.")
        theta = 0.0
        for j in range(n):
            if abs(dc[j]) > 1e-10:
                if abs(c[j]) > 1e-10:
                    theta = max(theta, abs(dc[j] / c[j]))
                else:
                    theta = float("inf")

        def oguz(t: float, base_tol: float = 0.0) -> float:
            if math.isinf(t):
                return 1.0
            if t <= base_tol:
                return 0.0
            if is_max:
                return 2 * (t - base_tol) / ((1 - base_tol) * (1 + t))
            if t >= 1.0:
                return 1.0
            return min(1.0, 2 * (t - base_tol) / ((1 + base_tol) * (1 - t)))

        raw_bound = oguz(theta)
        tau = self._wendell_objective_tolerance(old_problem, state)
        if math.isinf(theta) or not math.isfinite(tau) or tau >= 1.0:
            tightened_bound = raw_bound
        else:
            tightened_bound = min(raw_bound, oguz(theta, tau))

        explanation = (
            f"Oguz bound: theta = {theta:.4f}, raw bound = {raw_bound:.4f} "
            f"({raw_bound:.1%} max opportunity cost). "
            f"Wendell objective tolerance tau = {tau:.4f}, "
            f"tightened bound = {tightened_bound:.4f} ({tightened_bound:.1%})."
        )
        return raw_bound, tightened_bound, explanation

    @staticmethod
    def _wendell_objective_tolerance(problem: LPProblem, state: SolveState) -> float:
        """Wendell (1985) maximum tolerance for the objective coefficients.

        The largest tau such that the retained basis stays optimal when
        every c_j moves independently within tau |c_j|:
            tau = min_{j nonbasic} |cbar_j| / (|c_j| + sum_i |c_B_i| |(B^-1 a_j)_i|).
        Requires the reconstructed basis inverse; returns inf when it is
        unavailable or no nonbasic column bounds the tolerance.
        """
        if state.basis_inverse is None or state.basis_indices is None:
            return float("inf")
        from clara.engine import standard_form
        A_aug, _, _ = standard_form.add_upper_bound_rows(problem)
        m_aug = A_aug.shape[0]
        B_inv = state.basis_inverse
        if B_inv.shape[0] != m_aug:
            return float("inf")
        n = problem.num_variables
        A_full = np.hstack([A_aug, np.eye(m_aug)])
        c_full = np.zeros(n + m_aug)
        c_full[:n] = problem.c
        basis = list(state.basis_indices)
        c_B = c_full[basis]
        y = c_B @ B_inv
        sign = 1.0 if problem.sense != "minimize" else -1.0
        basis_set = set(basis)
        tau = float("inf")
        for j in range(n + m_aug):
            if j in basis_set:
                continue
            rc = c_full[j] - y @ A_full[:, j]
            col = B_inv @ A_full[:, j]
            denom = abs(c_full[j]) + float(np.sum(np.abs(c_B) * np.abs(col)))
            if denom > 1e-12:
                tau = min(tau, max(-sign * rc, 0.0) / denom)
        return tau

    def _sensitivity_check(
        self,
        state: SolveState,
        change: ParameterChange,
        old_problem: LPProblem,
    ) -> tuple[bool, list[str], str]:
        """Check if all changes fall within sensitivity ranges.

        Returns:
            (within_range, violated_parameters, explanation)
        """
        violated: list[str] = []

        # Check RHS changes (Type R or RC)
        if change.delta_b is not None:
            for i, con_name in enumerate(old_problem.constraint_names):
                if abs(change.delta_b[i]) < 1e-10:
                    continue
                if con_name not in state.sensitivity.rhs_ranges:
                    violated.append(con_name)
                    continue
                lo, hi = state.sensitivity.rhs_ranges[con_name]
                new_b = old_problem.b[i] + change.delta_b[i]
                if new_b < lo - 1e-8 or new_b > hi + 1e-8:
                    violated.append(con_name)

        # Check obj coeff changes (Type C or RC)
        if change.delta_c is not None:
            for j, var_name in enumerate(old_problem.var_names):
                if abs(change.delta_c[j]) < 1e-10:
                    continue
                if var_name not in state.sensitivity.obj_coeff_ranges:
                    violated.append(var_name)
                    continue
                lo, hi = state.sensitivity.obj_coeff_ranges[var_name]
                new_c = old_problem.c[j] + change.delta_c[j]
                if new_c < lo - 1e-8 or new_c > hi + 1e-8:
                    violated.append(var_name)

        within = len(violated) == 0
        if within:
            explanation = "All changed parameters are within sensitivity ranges."
        else:
            explanation = f"Parameters outside ranges: {', '.join(violated)}."

        return within, violated, explanation
