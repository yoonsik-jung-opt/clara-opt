"""Explanation report data structures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class DetailLevel(Enum):
    BRIEF = auto()
    DETAILED = auto()


@dataclass
class BindingItem:
    name: str
    rhs: float
    dual_value: float
    rank: int


@dataclass
class NonBindingItem:
    name: str
    rhs: float
    slack: float
    utilization_pct: float


@dataclass
class BasicVarItem:
    name: str
    value: float
    obj_coeff: float
    contribution: float
    contribution_pct: float


@dataclass
class NonBasicVarItem:
    name: str
    value: float
    reduced_cost: float
    obj_coeff: float


@dataclass
class ObjSensItem:
    name: str
    current: float
    lower: float
    upper: float
    range_width: float
    pct_decrease: float
    pct_increase: float


@dataclass
class RhsSensItem:
    name: str
    current: float
    lower: float
    upper: float
    dual_value: float
    is_binding: bool
    range_width: float


def _fmt_num(v: float, places: int = 4) -> str:
    if math.isinf(v):
        return "no limit"
    return f"{v:.{places}f}"


def _json_num(v: float) -> float | None:
    if math.isinf(v) or math.isnan(v):
        return None
    return v


# ============================================================
# Sub-reports
# ============================================================

@dataclass
class BindingReport:
    binding: list[BindingItem]
    nonbinding: list[NonBindingItem]
    summary: str

    def to_text(self, level: DetailLevel) -> str:
        lines = ["BINDING CONSTRAINTS"]
        if level == DetailLevel.BRIEF:
            lines.append(self.summary)
            return "\n".join(lines)

        n_total = len(self.binding) + len(self.nonbinding)
        lines.append("\u2500" * 60)

        if self.binding:
            lines.append(f"Binding constraints ({len(self.binding)} of {n_total}):")
            lines.append("")
            for item in self.binding:
                lines.append(f"  {item.rank}. {item.name} (RHS = {_fmt_num(item.rhs)}): Fully utilized.")
                lines.append(f"     Shadow price: {_fmt_num(item.dual_value)}")
                lines.append(f"     \u2192 Each additional unit of {item.name} capacity increases")
                lines.append(f"       objective by {_fmt_num(item.dual_value)}.")
                lines.append("")

        if self.nonbinding:
            lines.append(f"Non-binding constraints ({len(self.nonbinding)} of {n_total}):")
            lines.append("")
            for item in self.nonbinding:
                lines.append(f"  {item.name} (RHS = {_fmt_num(item.rhs)}): Slack = {_fmt_num(item.slack)}")
                lines.append(f"     \u2192 {item.utilization_pct:.1f}% utilized. Has {_fmt_num(item.slack)} units of unused capacity.")
                lines.append("")

        if not self.binding:
            lines.append("No constraints are binding \u2014 check if the problem is unbounded.")
        elif not self.nonbinding:
            lines.append("All constraints are binding \u2014 the solution is fully constrained.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "binding": [
                {"name": i.name, "rhs": i.rhs, "dual_value": i.dual_value, "rank": i.rank}
                for i in self.binding
            ],
            "nonbinding": [
                {"name": i.name, "rhs": i.rhs, "slack": i.slack, "utilization_pct": i.utilization_pct}
                for i in self.nonbinding
            ],
        }


@dataclass
class VariableReport:
    basic: list[BasicVarItem]
    nonbasic: list[NonBasicVarItem]
    summary: str

    def to_text(self, level: DetailLevel) -> str:
        lines = ["VARIABLE STATUS"]
        if level == DetailLevel.BRIEF:
            lines.append(self.summary)
            return "\n".join(lines)

        n_vars = len(self.basic) + len(self.nonbasic)
        lines.append("\u2500" * 60)

        if self.basic:
            lines.append(f"Variables in solution ({len(self.basic)} of {n_vars}):")
            lines.append("")
            for v in self.basic:
                lines.append(f"  {v.name} = {_fmt_num(v.value)} (profit = {_fmt_num(v.obj_coeff)})")
                lines.append(f"     Contribution: {_fmt_num(v.contribution)} ({v.contribution_pct:.1f}% of objective)")
                lines.append("")

        if self.nonbasic:
            lines.append(f"Variables not in solution ({len(self.nonbasic)} of {n_vars}):")
            lines.append("")
            for v in self.nonbasic:
                rc_abs = abs(v.reduced_cost)
                lines.append(f"  {v.name} = {_fmt_num(v.value)}")
                lines.append(f"     Reduced cost: {_fmt_num(v.reduced_cost)}")
                lines.append(f"     \u2192 {v.name}'s profit must increase by {_fmt_num(rc_abs)}")
                lines.append(f"       before it enters the solution.")
                lines.append("")

        if not self.nonbasic:
            lines.append("All variables are in the solution.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "basic": [
                {"name": v.name, "value": v.value, "obj_coeff": v.obj_coeff,
                 "contribution": v.contribution, "contribution_pct": v.contribution_pct}
                for v in self.basic
            ],
            "nonbasic": [
                {"name": v.name, "value": v.value, "reduced_cost": v.reduced_cost,
                 "obj_coeff": v.obj_coeff}
                for v in self.nonbasic
            ],
        }


@dataclass
class SensitivityReport:
    obj_items: list[ObjSensItem]
    rhs_items: list[RhsSensItem]
    bottleneck: Optional[str]
    most_robust: Optional[str]
    most_fragile: Optional[str]
    summary: str

    def to_text(self, level: DetailLevel) -> str:
        lines = ["SENSITIVITY ANALYSIS"]
        if level == DetailLevel.BRIEF:
            lines.append(self.summary)
            return "\n".join(lines)

        lines.append("\u2500" * 60)

        # Obj coeff ranges
        lines.append("Objective coefficient ranges (current basis stays optimal):")
        lines.append("")
        for item in self.obj_items:
            lines.append(f"  {item.name} (current = {_fmt_num(item.current)}):")
            lines.append(f"     Allowable range: [{_fmt_num(item.lower)}, {_fmt_num(item.upper)}]")
            pct_dec = f"{item.pct_decrease:.0f}%" if not math.isinf(item.pct_decrease) else "no limit"
            pct_inc = f"{item.pct_increase:.0f}%" if not math.isinf(item.pct_increase) else "no limit"
            lines.append(f"     \u2192 Can decrease by {pct_dec} or increase by {pct_inc} without")
            lines.append(f"       changing the solution.")
            lines.append("")

        # RHS ranges
        lines.append("RHS ranges (current basis stays feasible):")
        lines.append("")
        for item in self.rhs_items:
            if item.is_binding:
                lines.append(f"  {item.name} (current = {_fmt_num(item.current)}, shadow price = {_fmt_num(item.dual_value)}):")
                lines.append(f"     Allowable range: [{_fmt_num(item.lower)}, {_fmt_num(item.upper)}]")
                lines.append(f"     \u2192 Within this range, each unit change in {item.name} changes")
                lines.append(f"       the objective by {_fmt_num(item.dual_value)}.")
            else:
                slack = item.current - item.lower if not math.isinf(item.lower) else float("inf")
                lines.append(f"  {item.name} (current = {_fmt_num(item.current)}, shadow price = {_fmt_num(item.dual_value)}):")
                lines.append(f"     \u2192 Non-binding. RHS can decrease by {_fmt_num(slack)} without")
                lines.append(f"       affecting the solution.")
            lines.append("")

        # Recommendations
        lines.append("Recommendations:")
        if self.bottleneck:
            dual = next((r.dual_value for r in self.rhs_items if r.name == self.bottleneck), 0)
            lines.append(f"  \u2192 Bottleneck: {self.bottleneck} has the highest shadow price ({_fmt_num(dual)}).")
            lines.append(f"    Expanding this constraint first gives the best return.")
        if self.most_fragile:
            width = next((o.range_width for o in self.obj_items if o.name == self.most_fragile), 0)
            lines.append(f"  \u2192 Most fragile: {self.most_fragile} has the narrowest allowable range")
            lines.append(f"    (width {_fmt_num(width, 2)}). Monitor this parameter closely.")
        if self.most_robust:
            width = next((o.range_width for o in self.obj_items if o.name == self.most_robust), 0)
            lines.append(f"  \u2192 Most robust: {self.most_robust} has the widest allowable range")
            lines.append(f"    (width {_fmt_num(width, 2)}). This parameter is stable.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "bottleneck": self.bottleneck,
            "most_robust": self.most_robust,
            "most_fragile": self.most_fragile,
            "obj_coeff_ranges": [
                {"name": o.name, "current": o.current, "lower": _json_num(o.lower),
                 "upper": _json_num(o.upper), "range_width": _json_num(o.range_width),
                 "pct_decrease": _json_num(o.pct_decrease), "pct_increase": _json_num(o.pct_increase)}
                for o in self.obj_items
            ],
            "rhs_ranges": [
                {"name": r.name, "current": r.current, "lower": _json_num(r.lower),
                 "upper": _json_num(r.upper), "dual_value": r.dual_value,
                 "is_binding": r.is_binding, "range_width": _json_num(r.range_width)}
                for r in self.rhs_items
            ],
        }


# ============================================================
# Top-level report
# ============================================================

@dataclass
class ExplanationReport:
    binding: BindingReport
    variable: VariableReport
    sensitivity: SensitivityReport
    header: str
    bnb: object = None  # Optional BnBReport (avoid circular import)

    def to_text(self, level: DetailLevel = DetailLevel.DETAILED) -> str:
        sep = "\u2550" * 60
        parts = [
            sep,
            self.header,
            sep,
            "",
        ]
        if self.bnb is not None:
            parts.append(self.bnb.to_text(level))
            parts.append("")
        parts.extend([
            self.binding.to_text(level),
            "",
            self.variable.to_text(level),
            "",
            self.sensitivity.to_text(level),
        ])
        return "\n".join(parts)

    def to_dict(self) -> dict:
        d = {
            "header": self.header,
            "binding_report": self.binding.to_dict(),
            "variable_report": self.variable.to_dict(),
            "sensitivity_report": self.sensitivity.to_dict(),
        }
        if self.bnb is not None:
            d["bnb_report"] = self.bnb.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
