"""Explainer — orchestrates report generation from SolveState.

Usage:
    explainer = Explainer()
    report = explainer.explain(state, problem=problem)
    print(report.to_text())
    print(report.to_json())
"""

from __future__ import annotations

from typing import Optional

from clara.explain.binding import BindingReportGenerator
from clara.explain.sensitivity import SensitivityReportGenerator
from clara.explain.types import (
    BasicVarItem,
    DetailLevel,
    ExplanationReport,
    NonBasicVarItem,
    VariableReport,
    _fmt_num,
)
from clara.explain.variable import VariableReportGenerator
from clara.model.problem import LPProblem
from clara.model.solve_state import BasisStatus, SolveState


class Explainer:
    """Rule-based explanation generator. No LLM dependency."""

    def __init__(self) -> None:
        self._binding = BindingReportGenerator()
        self._sensitivity = SensitivityReportGenerator()

    def explain(
        self,
        state: SolveState,
        level: DetailLevel = DetailLevel.DETAILED,
        problem: Optional[LPProblem] = None,
    ) -> ExplanationReport:
        """Generate a complete explanation report from a SolveState.

        Args:
            state: Immutable solver state snapshot.
            level: BRIEF or DETAILED.
            problem: Optional LPProblem for objective coefficients.

        Returns:
            ExplanationReport containing all three sub-reports.
        """
        # Extract obj coefficients from problem if available
        obj_coeffs = {}
        if problem is not None:
            for j, name in enumerate(problem.var_names):
                obj_coeffs[name] = float(problem.c[j])

        header = self._make_header(state)
        binding = self._binding.generate(state, level)
        variable = self._make_variable_report(state, level, obj_coeffs)
        sensitivity = self._sensitivity.generate(state, level, obj_coeffs)

        return ExplanationReport(
            binding=binding,
            variable=variable,
            sensitivity=sensitivity,
            header=header,
        )

    def _make_header(self, state: SolveState) -> str:
        name = state.problem_name or "unnamed"
        return (
            f"CLARA \u2014 {name}\n"
            f"Optimal value: {_fmt_num(state.optimal_value)} ({state.status.name})\n"
            f"Solved in {state.iteration_count} iterations, "
            f"{state.solve_time_seconds:.3f}s ({state.engine.name})"
        )

    def _make_variable_report(
        self,
        state: SolveState,
        level: DetailLevel,
        obj_coeffs: dict[str, float],
    ) -> VariableReport:
        """Build variable report with actual obj coefficients."""
        basic_items = []
        nonbasic_items = []
        opt_val = state.optimal_value if state.optimal_value != 0 else 1.0

        for v in state.variables:
            coeff = obj_coeffs.get(v.name, 0.0)

            if v.basis_status == BasisStatus.BASIC:
                contrib = v.value * coeff
                contrib_pct = (contrib / state.optimal_value * 100) if state.optimal_value != 0 else 0.0
                basic_items.append(BasicVarItem(
                    name=v.name,
                    value=v.value,
                    obj_coeff=coeff,
                    contribution=contrib,
                    contribution_pct=contrib_pct,
                ))
            else:
                nonbasic_items.append(NonBasicVarItem(
                    name=v.name,
                    value=v.value,
                    reduced_cost=v.reduced_cost,
                    obj_coeff=coeff,
                ))

        basic_items.sort(key=lambda x: x.contribution, reverse=True)
        nonbasic_items.sort(key=lambda x: x.reduced_cost)

        n_vars = len(basic_items) + len(nonbasic_items)
        parts = [f"{len(basic_items)} of {n_vars} variables are in the solution."]
        if basic_items:
            top = basic_items[0]
            parts.append(f"{top.name} contributes {top.contribution_pct:.0f}% of the objective.")
        if nonbasic_items:
            parts.append(f"{len(nonbasic_items)} variables are not produced (at lower bound).")
        summary = " ".join(parts)

        return VariableReport(
            basic=basic_items,
            nonbasic=nonbasic_items,
            summary=summary,
        )
