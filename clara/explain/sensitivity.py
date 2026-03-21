"""Sensitivity report generator."""

from __future__ import annotations

import math

from clara.explain.types import (
    DetailLevel,
    ObjSensItem,
    RhsSensItem,
    SensitivityReport,
    _fmt_num,
)
from clara.model.solve_state import SolveState


class SensitivityReportGenerator:
    """Generates the sensitivity report (XAIOR Actionability)."""

    def generate(
        self, state: SolveState, level: DetailLevel, obj_coeffs: dict[str, float] | None = None
    ) -> SensitivityReport:
        obj_items = self._build_obj_items(state, obj_coeffs or {})
        rhs_items = self._build_rhs_items(state)

        # Identify bottleneck, most fragile, most robust
        binding_rhs = [r for r in rhs_items if r.is_binding]
        bottleneck = max(binding_rhs, key=lambda r: abs(r.dual_value)).name if binding_rhs else None

        finite_obj = [o for o in obj_items if not math.isinf(o.range_width)]
        most_fragile = min(finite_obj, key=lambda o: o.range_width).name if finite_obj else None
        most_robust = max(finite_obj, key=lambda o: o.range_width).name if finite_obj else None

        summary = self._make_summary(obj_items, rhs_items, bottleneck, most_fragile, most_robust)

        return SensitivityReport(
            obj_items=obj_items,
            rhs_items=rhs_items,
            bottleneck=bottleneck,
            most_robust=most_robust,
            most_fragile=most_fragile,
            summary=summary,
        )

    def _build_obj_items(
        self, state: SolveState, obj_coeffs: dict[str, float]
    ) -> list[ObjSensItem]:
        items = []
        for v in state.variables:
            lo, hi = v.obj_coeff_range
            current = obj_coeffs.get(v.name, 0.0)
            width = hi - lo if not (math.isinf(hi) or math.isinf(lo)) else float("inf")

            if current != 0:
                pct_dec = (current - lo) / abs(current) * 100 if not math.isinf(lo) else float("inf")
                pct_inc = (hi - current) / abs(current) * 100 if not math.isinf(hi) else float("inf")
            else:
                pct_dec = float("inf") if math.isinf(lo) else float("inf")
                pct_inc = float("inf") if math.isinf(hi) else float("inf")

            items.append(ObjSensItem(
                name=v.name,
                current=current,
                lower=lo,
                upper=hi,
                range_width=width,
                pct_decrease=pct_dec,
                pct_increase=pct_inc,
            ))

        # Sort by range_width ascending (most fragile first)
        items.sort(key=lambda o: o.range_width if not math.isinf(o.range_width) else float("inf"))
        return items

    def _build_rhs_items(self, state: SolveState) -> list[RhsSensItem]:
        items = []
        for c in state.constraints:
            lo, hi = c.rhs_range
            width = hi - lo if not (math.isinf(hi) or math.isinf(lo)) else float("inf")
            items.append(RhsSensItem(
                name=c.name,
                current=c.rhs,
                lower=lo,
                upper=hi,
                dual_value=c.dual_value,
                is_binding=c.is_binding,
                range_width=width,
            ))

        # Sort by dual_value descending
        items.sort(key=lambda r: abs(r.dual_value), reverse=True)
        return items

    def _make_summary(
        self,
        obj_items: list[ObjSensItem],
        rhs_items: list[RhsSensItem],
        bottleneck: str | None,
        most_fragile: str | None,
        most_robust: str | None,
    ) -> str:
        parts = []

        if most_robust:
            r = next(o for o in obj_items if o.name == most_robust)
            parts.append(
                f"Most robust: {r.name}'s profit can range "
                f"[{_fmt_num(r.lower, 2)}, {_fmt_num(r.upper, 2)}] "
                f"(width {_fmt_num(r.range_width, 2)}) "
                f"without changing the solution structure."
            )

        if bottleneck:
            b = next(r for r in rhs_items if r.name == bottleneck)
            parts.append(
                f"Bottleneck: Expanding {b.name} capacity has the highest "
                f"marginal return ({_fmt_num(b.dual_value)} per unit)."
            )

        if most_fragile:
            f = next(o for o in obj_items if o.name == most_fragile)
            pct = f"{f.pct_decrease:.0f}%" if not math.isinf(f.pct_decrease) else "no limit"
            parts.append(
                f"Most fragile: {f.name}'s profit has the narrowest range "
                f"[{_fmt_num(f.lower, 2)}, {_fmt_num(f.upper, 2)}] "
                f"\u2014 can only decrease by {pct} before the solution changes."
            )

        return " ".join(parts)
