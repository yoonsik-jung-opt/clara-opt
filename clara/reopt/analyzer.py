"""Impact Analyzer — decides whether reoptimization is needed.

Three-step decision pipeline:
    Step 1: Oguz bound (Type C only) — quick upper bound on opportunity cost
    Step 2: Sensitivity range check — does the change break the current basis?
    Step 3: Reoptimization method recommendation

References:
    Oguz (2000), Management Science 46(7) — 2δ/(1+δ) bound
    Wendell (1985), Management Science 31 — tolerance tightening
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

        # Type C: Oguz bound first, then sensitivity
        if change.change_type == ChangeType.TYPE_C:
            raw, tightened, oguz_expl = self._oguz_bound(change, old_problem, state)
            within, violated, sens_expl = self._sensitivity_check(state, change, old_problem)

            if within:
                return ReoptDecision(
                    should_reoptimize=False,
                    reason=f"All objective coefficient changes within sensitivity ranges. {oguz_expl}",
                    oguz_bound=tightened,
                    within_sensitivity=True,
                    recommended_method="none",
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
                    recommended_method="none",
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
        """Compute Oguz bound and Wendell-tightened bound.

        Returns:
            (raw_bound, tightened_bound, explanation)
        """
        # δ = max relative change across all variables
        delta = 0.0
        for j in range(len(old_problem.var_names)):
            if change.delta_c is not None and abs(change.delta_c[j]) > 1e-10:
                if abs(old_problem.c[j]) > 1e-10:
                    rel = abs(change.delta_c[j] / old_problem.c[j])
                    delta = max(delta, rel)
                else:
                    delta = float("inf")

        if math.isinf(delta):
            raw_bound = 1.0
        else:
            raw_bound = 2 * delta / (1 + delta) if delta > 0 else 0.0

        # Wendell tolerance: smallest relative obj coeff change that breaks basis
        alpha = float("inf")
        for j, name in enumerate(old_problem.var_names):
            if name in state.sensitivity.obj_coeff_ranges:
                lo, hi = state.sensitivity.obj_coeff_ranges[name]
                c_j = old_problem.c[j]
                if abs(c_j) > 1e-10:
                    tol_down = abs((c_j - lo) / c_j) if not math.isinf(lo) else float("inf")
                    tol_up = abs((hi - c_j) / c_j) if not math.isinf(hi) else float("inf")
                    alpha = min(alpha, tol_down, tol_up)

        if math.isinf(alpha):
            tightened_bound = raw_bound
        elif delta <= alpha:
            tightened_bound = 0.0
        else:
            delta_prime = (delta - alpha) / (1 + alpha)
            tightened_bound = 2 * delta_prime / (1 + delta_prime)

        explanation = (
            f"Oguz bound: δ = {delta:.4f}, raw bound = {raw_bound:.4f} "
            f"({raw_bound:.1%} max opportunity cost). "
            f"Wendell tolerance α = {alpha:.4f}, "
            f"tightened bound = {tightened_bound:.4f} ({tightened_bound:.1%})."
        )

        return raw_bound, tightened_bound, explanation

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
