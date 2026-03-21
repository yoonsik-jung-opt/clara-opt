"""Branch-and-Bound explanation report generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from clara.explain.types import DetailLevel, _fmt_num
from clara.model.solve_state import SolveState


@dataclass
class BnBReport:
    """B&B tree explanation report."""
    summary: str
    root_lp_bound: float
    mip_optimal: float
    gap_closed_pct: float
    nodes_explored: int
    nodes_by_reason: dict[str, int]
    optimal_node_id: Optional[int]
    optimal_depth: int
    max_depth: int

    def to_text(self, level: DetailLevel) -> str:
        lines = ["BRANCH-AND-BOUND SUMMARY"]
        if level == DetailLevel.BRIEF:
            lines.append(self.summary)
            return "\n".join(lines)

        lines.append("\u2500" * 60)
        lines.append(f"Root LP relaxation: {_fmt_num(self.root_lp_bound)}")
        lines.append(f"MIP optimal: {_fmt_num(self.mip_optimal)} (gap: {self.gap_closed_pct:.2f}%)")

        reasons = ", ".join(
            f"{count} {reason}" for reason, count in self.nodes_by_reason.items() if count > 0
        )
        lines.append(f"Nodes explored: {self.nodes_explored} ({reasons})")

        if self.optimal_node_id is not None:
            lines.append(f"Solution found at node {self.optimal_node_id} (depth {self.optimal_depth})")
        lines.append(f"Max tree depth: {self.max_depth}")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "root_lp_bound": self.root_lp_bound,
            "mip_optimal": self.mip_optimal,
            "gap_closed_pct": self.gap_closed_pct,
            "nodes_explored": self.nodes_explored,
            "nodes_by_reason": self.nodes_by_reason,
            "optimal_node_id": self.optimal_node_id,
            "optimal_depth": self.optimal_depth,
            "max_depth": self.max_depth,
        }


class BnBReportGenerator:
    """Generate B&B explanation from bnb_history."""

    def generate(self, state: SolveState, level: DetailLevel) -> Optional[BnBReport]:
        if state.bnb_history is None:
            return None

        history = state.bnb_history
        if not history:
            return None

        root_bound = history[0].lp_relaxation_value
        mip_opt = state.optimal_value

        # Gap
        if root_bound != 0 and not math.isnan(root_bound):
            gap = abs(root_bound - mip_opt) / abs(root_bound) * 100
        else:
            gap = 0.0

        # Count by fathom reason
        reasons: dict[str, int] = {"bound": 0, "infeasible": 0, "integer_feasible": 0, "branched": 0}
        for snap in history:
            if snap.is_fathomed and snap.fathom_reason:
                reasons[snap.fathom_reason] = reasons.get(snap.fathom_reason, 0) + 1
            elif not snap.is_fathomed:
                reasons["branched"] += 1

        # Find optimal node
        int_nodes = [s for s in history if s.fathom_reason == "integer_feasible"]
        optimal_node = None
        optimal_depth = 0
        if int_nodes:
            best = max(int_nodes, key=lambda s: s.lp_relaxation_value)
            optimal_node = best.node_id
            optimal_depth = best.depth

        max_depth = max(s.depth for s in history)

        summary = (
            f"MIP solved: {_fmt_num(mip_opt)} "
            f"(LP relaxation: {_fmt_num(root_bound)}, gap: {gap:.1f}%). "
            f"Explored {len(history)} nodes. "
            f"Solution found at node {optimal_node} (depth {optimal_depth})."
        )

        return BnBReport(
            summary=summary,
            root_lp_bound=root_bound,
            mip_optimal=mip_opt,
            gap_closed_pct=gap,
            nodes_explored=len(history),
            nodes_by_reason=reasons,
            optimal_node_id=optimal_node,
            optimal_depth=optimal_depth,
            max_depth=max_depth,
        )
