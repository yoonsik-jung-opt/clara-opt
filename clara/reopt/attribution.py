"""Objective change attribution — decompose Δz into parameter contributions.

First-order decomposition (exact when basis preserved):
    Δz ≈ yᵀΔb + Δcᵀx + Δc_Bᵀ B⁻¹ Δb

Shapley value decomposition (when basis changes):
    shapley_b = [(z_b - z_old) + (z_new - z_c)] / 2
    shapley_c = [(z_c - z_old) + (z_new - z_b)] / 2
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clara.engine.simplex import RevisedSimplex
from clara.model.problem import LPProblem
from clara.model.solve_state import SolveState
from clara.reopt.types import AttributionResult, ParameterChange


class ChangeAttributor:
    """Decompose objective change into parameter contributions."""

    def attribute(
        self,
        old_state: SolveState,
        new_state: SolveState,
        old_problem: LPProblem,
        new_problem: LPProblem,
        change: ParameterChange,
        compute_shapley: bool = True,
    ) -> AttributionResult:
        """Attribute the objective change to individual parameters.

        Args:
            old_state: SolveState before change (with B⁻¹).
            new_state: SolveState after change.
            old_problem: Original problem.
            new_problem: Modified problem.
            change: Detected parameter change.
            compute_shapley: Whether to compute Shapley decomposition.
        """
        z_old = old_state.optimal_value
        z_new = new_state.optimal_value
        delta_z = z_new - z_old

        n = old_problem.num_variables
        m = old_problem.num_constraints

        # Extract old solution values
        x_old = np.array([v.value for v in old_state.variables[:n]])
        y_old = np.array([c.dual_value for c in old_state.constraints[:m]])

        # Delta vectors
        delta_b = change.delta_b if change.delta_b is not None else np.zeros(m)
        delta_c = change.delta_c if change.delta_c is not None else np.zeros(n)

        # Adjust for minimize: engine stores negated values internally
        if old_problem.sense == "minimize":
            y_old = -y_old
            delta_c_internal = -delta_c
        else:
            delta_c_internal = delta_c

        # First-order effects
        rhs_effect = float(y_old @ delta_b[:m])
        obj_effect = float(delta_c[:n] @ x_old)

        # Interaction: Δc_Bᵀ B⁻¹ Δb
        interaction = 0.0
        if old_state.basis_inverse is not None and np.any(delta_b != 0) and np.any(delta_c != 0):
            B_inv = old_state.basis_inverse
            # Get basis indices
            basis_decision = [j for j, v in enumerate(old_state.variables[:n])
                              if v.basis_status.name == "BASIC"]
            dc_B = np.zeros(B_inv.shape[0])
            for k, j in enumerate(basis_decision):
                if k < len(dc_B) and j < n:
                    dc_B[k] = delta_c_internal[j]
            interaction = float(dc_B @ (B_inv @ delta_b[:B_inv.shape[0]]))
            if old_problem.sense == "minimize":
                interaction = -interaction

        residual = delta_z - (rhs_effect + obj_effect + interaction)
        basis_preserved = abs(residual) < 1e-4

        # Per-parameter contributions
        rhs_contribs = {}
        for i in range(min(m, len(old_state.constraints))):
            name = old_problem.constraint_names[i] if i < len(old_problem.constraint_names) else f"C{i+1}"
            rhs_contribs[name] = float(y_old[i] * delta_b[i]) if i < len(delta_b) else 0.0

        obj_contribs = {}
        for j in range(n):
            name = old_problem.var_names[j]
            obj_contribs[name] = float(delta_c[j] * x_old[j])

        # Shapley values
        shapley_b, shapley_c, z_b_only, z_c_only = None, None, None, None
        if compute_shapley and np.any(delta_b != 0) and np.any(delta_c != 0):
            z_b_only = self._solve_variant(old_problem, delta_b=delta_b)
            z_c_only = self._solve_variant(old_problem, delta_c=delta_c)

            if z_b_only is not None and z_c_only is not None:
                shapley_b = ((z_b_only - z_old) + (z_new - z_c_only)) / 2
                shapley_c = ((z_c_only - z_old) + (z_new - z_b_only)) / 2

        return AttributionResult(
            delta_z=delta_z,
            rhs_effect=rhs_effect,
            obj_effect=obj_effect,
            interaction_effect=interaction,
            first_order_residual=residual,
            rhs_contributions=rhs_contribs,
            obj_contributions=obj_contribs,
            shapley_b=shapley_b,
            shapley_c=shapley_c,
            z_b_only=z_b_only,
            z_c_only=z_c_only,
            basis_preserved=basis_preserved,
            nonlinearity=abs(residual) / max(abs(delta_z), 1e-10),
        )

    def _solve_variant(
        self, old_problem: LPProblem,
        delta_b: Optional[np.ndarray] = None,
        delta_c: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Solve a variant of old_problem with only b or c changed."""
        b_new = old_problem.b.copy()
        c_new = old_problem.c.copy()
        if delta_b is not None:
            b_new = b_new + delta_b[:len(b_new)]
        if delta_c is not None:
            c_new = c_new + delta_c[:len(c_new)]

        variant = LPProblem(
            c=c_new, A=old_problem.A, b=b_new,
            var_names=list(old_problem.var_names),
            constraint_names=list(old_problem.constraint_names),
            name=old_problem.name,
            sense=old_problem.sense,
        )
        try:
            state = RevisedSimplex(variant).solve()
            return state.optimal_value if state.is_optimal else None
        except Exception:
            return None
