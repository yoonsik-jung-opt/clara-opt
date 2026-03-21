"""Diff Report — compares two SolveStates and explains what changed.

Final piece of the reoptimization pipeline:
    Change Detector → Impact Analyzer → Reoptimizer → **Diff Report**
"""

from __future__ import annotations

from typing import Optional

from clara.model.solve_state import BasisStatus, SolveState
from clara.reopt.types import (
    ConstraintChange,
    DiffReport,
    ParameterChange,
    ReoptResult,
    VariableChange,
)


class DiffReporter:
    """Compare two SolveStates and generate an explanation of changes."""

    CHANGE_TOL = 1e-6

    def diff(
        self,
        old_state: SolveState,
        new_state: SolveState,
        change: ParameterChange,
        reopt_result: ReoptResult,
    ) -> DiffReport:
        """Generate a diff report comparing old and new solutions."""
        # Objective
        old_obj = old_state.optimal_value
        new_obj = new_state.optimal_value
        obj_delta = new_obj - old_obj
        obj_delta_pct = (obj_delta / abs(old_obj) * 100) if abs(old_obj) > 1e-10 else 0.0

        # Variable changes
        var_changes, entered, left = self._compute_variable_changes(old_state, new_state)

        # Constraint changes
        con_changes, binding_new, nonbinding_new = self._compute_constraint_changes(
            old_state, new_state, change
        )

        # Bottleneck
        old_bn = self._find_bottleneck(old_state)
        new_bn = self._find_bottleneck(new_state)

        # Summary
        summary = self._make_summary(
            change, reopt_result, obj_delta, obj_delta_pct,
            entered, left, binding_new, nonbinding_new, old_bn, new_bn,
        )

        return DiffReport(
            change=change,
            reopt_result=reopt_result,
            old_objective=old_obj,
            new_objective=new_obj,
            objective_delta=obj_delta,
            objective_delta_pct=obj_delta_pct,
            variable_changes=tuple(var_changes),
            variables_entered_basis=tuple(entered),
            variables_left_basis=tuple(left),
            constraint_changes=tuple(con_changes),
            became_binding=tuple(binding_new),
            became_nonbinding=tuple(nonbinding_new),
            old_bottleneck=old_bn,
            new_bottleneck=new_bn,
            bottleneck_shifted=old_bn != new_bn,
            summary=summary,
        )

    def _compute_variable_changes(
        self, old_state: SolveState, new_state: SolveState
    ) -> tuple[list[VariableChange], list[str], list[str]]:
        old_vars = {v.name: v for v in old_state.variables}
        new_vars = {v.name: v for v in new_state.variables}
        changes = []
        entered = []
        left = []

        for name in old_vars:
            if name not in new_vars:
                continue
            ov, nv = old_vars[name], new_vars[name]
            delta = nv.value - ov.value
            if abs(delta) < self.CHANGE_TOL and ov.basis_status == nv.basis_status:
                continue

            delta_pct = (delta / abs(ov.value) * 100) if abs(ov.value) > 1e-10 else 0.0
            bc = ov.basis_status != nv.basis_status

            changes.append(VariableChange(
                name=name,
                old_value=ov.value, new_value=nv.value,
                delta=delta, delta_pct=delta_pct,
                old_basis=ov.basis_status, new_basis=nv.basis_status,
                basis_changed=bc,
            ))

            if ov.basis_status != BasisStatus.BASIC and nv.basis_status == BasisStatus.BASIC:
                entered.append(name)
            elif ov.basis_status == BasisStatus.BASIC and nv.basis_status != BasisStatus.BASIC:
                left.append(name)

        changes.sort(key=lambda vc: abs(vc.delta), reverse=True)
        return changes, entered, left

    def _compute_constraint_changes(
        self, old_state: SolveState, new_state: SolveState, change: ParameterChange
    ) -> tuple[list[ConstraintChange], list[str], list[str]]:
        old_cons = {c.name: c for c in old_state.constraints}
        new_cons = {c.name: c for c in new_state.constraints}
        changes = []
        binding_new = []
        nonbinding_new = []

        for name in old_cons:
            if name not in new_cons:
                continue
            oc, nc = old_cons[name], new_cons[name]

            new_rhs = nc.rhs
            bb = not oc.is_binding and nc.is_binding
            bn = oc.is_binding and not nc.is_binding

            changes.append(ConstraintChange(
                name=name,
                old_rhs=oc.rhs, new_rhs=new_rhs,
                old_slack=oc.slack, new_slack=nc.slack,
                old_dual=oc.dual_value, new_dual=nc.dual_value,
                old_binding=oc.is_binding, new_binding=nc.is_binding,
                became_binding=bb, became_nonbinding=bn,
            ))

            if bb:
                binding_new.append(name)
            if bn:
                nonbinding_new.append(name)

        return changes, binding_new, nonbinding_new

    def _find_bottleneck(self, state: SolveState) -> Optional[str]:
        binding = [c for c in state.constraints if c.is_binding]
        if not binding:
            return None
        return max(binding, key=lambda c: abs(c.dual_value)).name

    def _make_summary(
        self, change, result, obj_delta, obj_delta_pct,
        entered, left, binding_new, nonbinding_new, old_bn, new_bn,
    ) -> str:
        parts = []
        parts.append(f"Parameter change: {change.summary}.")
        parts.append(f"Reoptimization: {result.summary}")

        direction = "increased" if obj_delta > 0 else "decreased"
        parts.append(
            f"Objective {direction} by {abs(obj_delta_pct):.1f}% ({abs(obj_delta):.4f})."
        )

        if entered:
            parts.append(f"Variables entered basis: {', '.join(entered)}.")
        if left:
            parts.append(f"Variables left basis: {', '.join(left)}.")
        if not entered and not left:
            parts.append("Solution structure unchanged (same basis).")

        if binding_new:
            parts.append(f"Newly binding: {', '.join(binding_new)}.")
        if nonbinding_new:
            parts.append(f"No longer binding: {', '.join(nonbinding_new)}.")

        if old_bn and new_bn and old_bn != new_bn:
            parts.append(f"Bottleneck shifted from {old_bn} to {new_bn}.")

        return "\n".join(parts)
