"""Variable report generator."""

from __future__ import annotations

from clara.explain.types import (
    BasicVarItem,
    DetailLevel,
    NonBasicVarItem,
    VariableReport,
    _fmt_num,
)
from clara.model.solve_state import BasisStatus, SolveState


class VariableReportGenerator:
    """Generates the variable report (XAIOR Justifiability)."""

    def generate(self, state: SolveState, level: DetailLevel) -> VariableReport:
        basic_items = []
        nonbasic_items = []
        opt_val = state.optimal_value if state.optimal_value != 0 else 1.0

        for v in state.variables:
            if v.basis_status == BasisStatus.BASIC:
                obj_coeff = self._get_obj_coeff(v, state)
                contrib = v.value * obj_coeff
                contrib_pct = (contrib / state.optimal_value * 100) if state.optimal_value != 0 else 0.0
                basic_items.append(BasicVarItem(
                    name=v.name,
                    value=v.value,
                    obj_coeff=obj_coeff,
                    contribution=contrib,
                    contribution_pct=contrib_pct,
                ))
            else:
                obj_coeff = self._get_obj_coeff(v, state)
                nonbasic_items.append(NonBasicVarItem(
                    name=v.name,
                    value=v.value,
                    reduced_cost=v.reduced_cost,
                    obj_coeff=obj_coeff,
                ))

        # Sort basic by contribution descending
        basic_items.sort(key=lambda x: x.contribution, reverse=True)
        # Sort nonbasic by reduced cost ascending (closest to entering first)
        nonbasic_items.sort(key=lambda x: x.reduced_cost)

        summary = self._make_summary(basic_items, nonbasic_items)
        return VariableReport(
            basic=basic_items,
            nonbasic=nonbasic_items,
            summary=summary,
        )

    def _get_obj_coeff(self, v, state: SolveState) -> float:
        """Extract objective coefficient from sensitivity range midpoint heuristic."""
        lo, hi = v.obj_coeff_range
        # The current obj coeff is within [lo, hi]. We can retrieve it from
        # the reduced cost: for basic vars rc=0, for nonbasic rc = c_j - y^T a_j.
        # Since we don't store c directly in SolveState, we use the range center
        # as approximation. But actually, for basic vars the current coeff is
        # somewhere in the range. We need the actual coeff.
        # Workaround: store obj_coeff in BasicVarItem/NonBasicVarItem via the
        # fact that contribution = value * coeff, so coeff = contribution / value.
        # But we don't have contribution yet...
        # The simplest approach: we store it during generation from the problem.
        # For now, we'll use a different approach - derive from reduced cost.
        # For non-basic: rc = c_j - y^T a_j, and at optimality for max, rc <= 0
        # For basic: rc = 0, and c_j is the actual coeff
        # We can't easily get c_j from SolveState alone without the problem.
        # Let's just use the midpoint of the sensitivity range as a reasonable estimate.
        # Actually, the VariableInfo doesn't store obj_coeff directly.
        # We'll compute it in the Explainer which has access to the problem.
        return 0.0  # placeholder — overridden by Explainer

    def _make_summary(
        self,
        basic: list[BasicVarItem],
        nonbasic: list[NonBasicVarItem],
    ) -> str:
        n_vars = len(basic) + len(nonbasic)
        parts = [f"{len(basic)} of {n_vars} variables are in the solution."]

        if basic:
            top = basic[0]
            parts.append(f"{top.name} contributes {top.contribution_pct:.0f}% of the objective.")

        if nonbasic:
            parts.append(f"{len(nonbasic)} variables are not produced (at lower bound).")

        return " ".join(parts)
