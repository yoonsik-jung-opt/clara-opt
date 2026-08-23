"""Supply chain production planning case study for CLARA paper §6.6.

Demonstrates the full CLARA pipeline on a realistic problem:
  Solve → Explain → Sensitivity Region → Attribute → Reoptimize → Diff Report
"""

import numpy as np

from clara.engine import HiGHSBackend
from clara.explain.explainer import Explainer
from clara.explain.types import DetailLevel
from clara.model.problem import LPProblem
from clara.reopt.analyzer import ImpactAnalyzer
from clara.reopt.attribution import ChangeAttributor
from clara.reopt.detector import ChangeDetector
from clara.reopt.diff_report import DiffReporter
from clara.reopt.reoptimizer import Reoptimizer
from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer
from clara.reopt.types import ReoptDecision


PRODUCTS = ["Widget-A", "Widget-B", "Gadget-C", "Gadget-D", "Premium-E"]
N_PRODUCTS = 5
N_PERIODS = 4
N_VARS = N_PRODUCTS * N_PERIODS  # 20


def create_base_problem():
    """Production planning: max profit over 4 quarters, 5 products, 3 resources.

    Profit = revenue - cost per unit. Resources are tight → interesting trade-offs.
    """

    # Profit per unit (revenue - cost), varies by quarter
    profit = np.array([
        [8, 7, 6, 9],     # Widget-A: cheap, moderate profit
        [12, 11, 10, 13],  # Widget-B: mid-range
        [18, 16, 17, 20],  # Gadget-C: higher profit, more resources
        [22, 21, 20, 24],  # Gadget-D: premium segment
        [35, 33, 32, 38],  # Premium-E: highest profit, most resources
    ])

    labor = [2, 3, 4, 5, 8]
    machine = [1, 2, 3, 2, 5]
    material = [3, 4, 6, 7, 10]

    # Tight capacities — forces trade-offs between products
    labor_cap = [200, 210, 190, 220]
    machine_cap = [120, 125, 115, 130]
    material_cap = [350, 360, 330, 380]

    # Max demand (upper bounds) — can't sell unlimited
    demand = np.array([
        [40, 45, 38, 50],
        [30, 35, 28, 40],
        [20, 22, 18, 25],
        [15, 18, 14, 20],
        [8, 10, 7, 12],
    ])

    n_cons = 3 * N_PERIODS  # 12
    A = np.zeros((n_cons, N_VARS))
    b = np.zeros(n_cons)
    con_names = []

    for t in range(N_PERIODS):
        for ri, (res, cap, res_name) in enumerate(zip(
            [labor, machine, material],
            [labor_cap, machine_cap, material_cap],
            ["labor", "machine", "material"],
        )):
            row = t * 3 + ri
            for p in range(N_PRODUCTS):
                col = p * N_PERIODS + t
                A[row, col] = res[p]
            b[row] = cap[t]
            con_names.append(f"{res_name}_Q{t+1}")

    var_names = [f"{PRODUCTS[p]}_Q{t+1}" for p in range(N_PRODUCTS) for t in range(N_PERIODS)]
    ub = demand.flatten().astype(float)

    return LPProblem(
        c=profit.flatten(), A=A, b=b,
        var_names=var_names, constraint_names=con_names,
        name="supply_chain", sense="maximize",
        upper_bounds=ub,
    ), profit, demand


def create_scenario_labor(base, labor_q3_delta=-40):
    """Scenario 1: Q3 labor capacity changes."""
    b_new = base.b.copy()
    b_new[2 * 3] += labor_q3_delta  # labor_Q3 row
    return LPProblem(
        c=base.c.copy(), A=base.A.copy(), b=b_new,
        var_names=list(base.var_names), constraint_names=list(base.constraint_names),
        name="supply_chain_labor_q3", sense=base.sense,
        upper_bounds=base.upper_bounds.copy() if base.upper_bounds is not None else None,
    )


def create_scenario_cost(base, multiplier=1.2):
    """Scenario 2: Premium-E profit changes."""
    c_new = base.c.copy()
    for t in range(N_PERIODS):
        c_new[4 * N_PERIODS + t] *= multiplier
    return LPProblem(
        c=c_new, A=base.A.copy(), b=base.b.copy(),
        var_names=list(base.var_names), constraint_names=list(base.constraint_names),
        name="supply_chain_premium_cost", sense=base.sense,
        upper_bounds=base.upper_bounds.copy() if base.upper_bounds is not None else None,
    )


def var_label(j):
    p, t = divmod(j, N_PERIODS)
    return f"{PRODUCTS[p]} Q{t+1}"


def main():
    base, profit, demand = create_base_problem()

    print("=" * 60)
    print("CLARA Case Study: Production Planning")
    print("=" * 60)
    print(f"Variables: {base.num_variables}, Constraints: {base.num_constraints}")
    print(f"Products: {', '.join(PRODUCTS)}")
    print(f"Periods: Q1-Q4, Sense: {base.sense}")

    # ========== STEP 1: Solve ==========
    state = HiGHSBackend().solve(base)
    print(f"\n--- Base Solution ---")
    print(f"Optimal profit: ${state.optimal_value:,.2f}")
    print(f"κ(B): {state.condition_number:.2e}")
    print(f"d₀: {state.basis_robustness_d0:.4f}")
    print(f"Degenerate vars: {state.degenerate_count}")
    print(f"Iterations: {state.iteration_count}")

    # Print production plan
    print(f"\nProduction plan:")
    for j in range(N_VARS):
        val = state.variables[j].value
        p, t = divmod(j, N_PERIODS)
        min_d = demand[p, t]
        max_d = demand[p, t]
        extra = " *" if val > 0.1 else ""
        if val > 0.1:
            print(f"  {var_label(j)}: {val:.1f} (max demand {max_d}){extra}")

    # ========== STEP 2: Explain ==========
    report = Explainer().explain(state, level=DetailLevel.BRIEF, problem=base)
    print(f"\n--- Explanation ---")
    print(f"Binding constraints: {len(report.binding.binding)}/{base.num_constraints}")
    for item in report.binding.binding:
        print(f"  {item.name}: shadow price = {abs(item.dual_value):.2f} (rank {item.rank})")
    if report.sensitivity.bottleneck:
        print(f"Bottleneck: {report.sensitivity.bottleneck}")
    if report.sensitivity.most_fragile:
        print(f"Most fragile: {report.sensitivity.most_fragile}")

    # ========== STEP 3: Sensitivity Region ==========
    region = SimultaneousRegionAnalyzer().analyze(state, base)
    print(f"\n--- Simultaneous Sensitivity Region ---")
    print(f"Chebyshev radius r* = {region.chebyshev_radius:.4f}")
    print(f"Min OAT tolerance = {region.min_oat_tolerance:.4f}")
    print(f"Simultaneity ratio ρ = {region.simultaneity_ratio:.4f}")

    # ========== STEP 4: Scenario 1 — Q3 labor shortage ==========
    print(f"\n{'=' * 60}")
    print("SCENARIO 1: Q3 labor capacity 480 → 400 hours (-16.7%)")
    print("=" * 60)

    pert1 = create_scenario_labor(base, labor_q3_delta=-40)
    change1 = ChangeDetector().detect(base, pert1)
    print(f"Change type: {change1.change_type.name}")

    # Solve perturbed
    state1 = HiGHSBackend().solve(pert1)

    # Attribution
    attr1 = ChangeAttributor().attribute(state, state1, base, pert1, change1, compute_shapley=False)
    print(f"Δz: ${abs(attr1.delta_z):+,.2f}")
    print(f"RHS contribution: {abs(attr1.rhs_effect)/max(abs(attr1.delta_z),1e-10)*100:.1f}%")
    print(f"η (nonlinearity): {attr1.nonlinearity:.4f}")
    print(f"Basis preserved: {attr1.basis_preserved}")

    # Top contributing constraints
    sorted_rhs = sorted(attr1.rhs_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_rhs[:3]:
        if abs(val) > 0.01:
            print(f"  {name}: ${val:+.2f}")

    # Reoptimize
    decision1 = ImpactAnalyzer().analyze(state, change1, base)
    print(f"Impact analysis: should_reoptimize={decision1.should_reoptimize}, method={decision1.recommended_method}")

    forced1 = ReoptDecision(should_reoptimize=True, reason="bench", recommended_method="warm_start")
    result1 = Reoptimizer().reoptimize(state, pert1, change1, forced1, old_problem=base)
    scratch1 = HiGHSBackend().solve(pert1)
    print(f"Warm-start: method={result1.method_used}, pivots={result1.pivots}")
    print(f"Scratch: pivots={scratch1.iteration_count}")
    print(f"New optimal profit: ${result1.new_state.optimal_value:,.2f}")
    match1 = abs(result1.new_state.optimal_value - scratch1.optimal_value) < 0.01
    print(f"Match: {match1}")

    # Diff report
    diff1 = DiffReporter().diff(state, result1.new_state, change1, result1)
    print(f"Variables changed: {len(diff1.variable_changes)}")
    for vc in diff1.variable_changes[:5]:
        print(f"  {vc.name}: {vc.old_value:.1f} → {vc.new_value:.1f} ({vc.delta:+.1f})")
    if diff1.bottleneck_shifted:
        print(f"Bottleneck shift: {diff1.old_bottleneck} → {diff1.new_bottleneck}")

    # ========== STEP 5: Scenario 2 — Premium-E cost +20% ==========
    print(f"\n{'=' * 60}")
    print("SCENARIO 2: Premium-E production cost +20% all quarters")
    print("=" * 60)

    pert2 = create_scenario_cost(base, multiplier=1.2)
    change2 = ChangeDetector().detect(base, pert2)
    print(f"Change type: {change2.change_type.name}")

    state2 = HiGHSBackend().solve(pert2)
    attr2 = ChangeAttributor().attribute(state, state2, base, pert2, change2, compute_shapley=False)
    print(f"Δz: ${abs(attr2.delta_z):+,.2f}")
    print(f"OBJ contribution: {abs(attr2.obj_effect)/max(abs(attr2.delta_z),1e-10)*100:.1f}%")
    print(f"η: {attr2.nonlinearity:.4f}")
    print(f"Basis preserved: {attr2.basis_preserved}")

    decision2 = ImpactAnalyzer().analyze(state, change2, base)
    print(f"Impact: should_reoptimize={decision2.should_reoptimize}, method={decision2.recommended_method}")

    if decision2.should_reoptimize:
        forced2 = ReoptDecision(should_reoptimize=True, reason="bench", recommended_method="warm_start")
        result2 = Reoptimizer().reoptimize(state, pert2, change2, forced2, old_problem=base)
        print(f"Warm-start: method={result2.method_used}, pivots={result2.pivots}")
        print(f"New profit: ${result2.new_state.optimal_value:,.2f}")
    else:
        print(f"SKIP reoptimization — old plan remains optimal")
        print(f"Profit with old plan: ${state2.optimal_value:,.2f}")

    print(f"\n{'=' * 60}")
    print("CASE STUDY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
