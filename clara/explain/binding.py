"""Binding constraint report generator."""

from __future__ import annotations

from clara.explain.types import (
    BindingItem,
    BindingReport,
    DetailLevel,
    NonBindingItem,
    _fmt_num,
)
from clara.model.solve_state import SolveState


class BindingReportGenerator:
    """Generates the binding constraint report (XAIOR Understandability)."""

    def generate(self, state: SolveState, level: DetailLevel) -> BindingReport:
        binding_items = []
        nonbinding_items = []

        for c in state.constraints:
            if c.is_binding:
                binding_items.append(BindingItem(
                    name=c.name,
                    rhs=c.rhs,
                    dual_value=c.dual_value,
                    rank=0,
                ))
            else:
                util = ((c.rhs - c.slack) / c.rhs * 100) if c.rhs != 0 else 0.0
                nonbinding_items.append(NonBindingItem(
                    name=c.name,
                    rhs=c.rhs,
                    slack=c.slack,
                    utilization_pct=util,
                ))

        # Sort binding by shadow price descending, assign ranks
        binding_items.sort(key=lambda x: x.dual_value, reverse=True)
        for i, item in enumerate(binding_items):
            item.rank = i + 1

        # Sort nonbinding by utilization descending
        nonbinding_items.sort(key=lambda x: x.utilization_pct, reverse=True)

        summary = self._make_summary(binding_items, nonbinding_items, state)
        return BindingReport(
            binding=binding_items,
            nonbinding=nonbinding_items,
            summary=summary,
        )

    def _make_summary(
        self,
        binding: list[BindingItem],
        nonbinding: list[NonBindingItem],
        state: SolveState,
    ) -> str:
        n_total = len(binding) + len(nonbinding)
        parts = [f"{len(binding)} of {n_total} constraints are binding."]

        if binding:
            top = binding[0]
            parts.append(
                f"{top.name} is the tightest bottleneck "
                f"(shadow price {_fmt_num(top.dual_value)})."
            )

        if nonbinding:
            nb = nonbinding[0]
            parts.append(
                f"{nb.name} has {_fmt_num(nb.slack)} units of unused capacity "
                f"({nb.utilization_pct:.0f}% utilized)."
            )

        return " ".join(parts)
