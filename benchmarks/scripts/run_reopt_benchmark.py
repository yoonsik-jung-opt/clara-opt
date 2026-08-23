"""Run Albici reoptimization comparison benchmark.

Usage:
    python benchmarks/scripts/run_reopt_benchmark.py

Compares reoptimization methods (recompute, warm_start, dual, parametric)
against scratch re-solve for each Albici scenario.

Output: benchmarks/results/reopt_comparison.csv
"""

import csv
import time
from pathlib import Path

import numpy as np

from clara.engine import HiGHSBackend
from clara.model.problem import LPProblem
from clara.reopt.analyzer import ImpactAnalyzer
from clara.reopt.detector import ChangeDetector
from clara.reopt.reoptimizer import Reoptimizer

RESULTS_DIR = Path(__file__).parent.parent / "results"


def albici_base():
    return LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )


SCENARIOS = {
    "base→b1": {
        "new_problem": lambda: LPProblem(
            c=[3, 4, 5], A=albici_base().A, b=[1300, 1200, 1800, 1000],
            var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
        ),
        "expected_type": "TYPE_R",
        "expected_optimal": 33100 / 9,
    },
    "base→b2": {
        "new_problem": lambda: LPProblem(
            c=[3, 4, 5], A=albici_base().A, b=[1500, 1300, 2400, 800],
            var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
        ),
        "expected_type": "TYPE_R",
        "expected_optimal": 4300.0,
    },
    "base→cost": {
        "new_problem": lambda: LPProblem(
            c=[8, 6, 7], A=albici_base().A, b=albici_base().b,
            var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
        ),
        "expected_type": "TYPE_C",
        "expected_optimal": 24800 / 3,
    },
    "base→columns": {
        "new_problem": lambda: LPProblem(
            c=[3, 4, 5, 3, 5],
            A=[[1, 2, 1, 2, 3], [1, 0, 3, 0, 1], [2, 1, 2, 1, 0], [0, 2, 3, 0, 1]],
            b=[1200, 1400, 2000, 800],
            var_names=["x1", "x2", "x3", "x4", "x5"],
            constraint_names=["S1", "S2", "S3", "S4"],
        ),
        "expected_type": "TYPE_V",
        "expected_optimal": 33700 / 9,
    },
    "compound (RC)": {
        "new_problem": lambda: LPProblem(
            c=[8, 6, 7], A=albici_base().A, b=[1500, 1300, 2400, 800],
            var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
        ),
        "expected_type": "TYPE_RC",
        "expected_optimal": 10000.0,
    },
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    old_problem = albici_base()
    base_state = HiGHSBackend().solve(old_problem)

    results = []
    print(f"{'Scenario':<16} {'Type':<8} {'Method':<16} {'Pivots':>6} "
          f"{'Scratch':>7} {'Speedup':>8} {'Optimal':>12} {'Match':>5}")
    print("-" * 85)

    for scenario_name, info in SCENARIOS.items():
        new_problem = info["new_problem"]()

        # Full pipeline
        change = ChangeDetector().detect(old_problem, new_problem)
        decision = ImpactAnalyzer().analyze(base_state, change, old_problem)
        result = Reoptimizer().reoptimize(
            base_state, new_problem, change, decision, old_problem=old_problem
        )

        # Scratch solve for comparison
        scratch_state = HiGHSBackend().solve(new_problem)
        scratch_pivots = scratch_state.iteration_count

        # Check optimal value
        opt = result.new_state.optimal_value
        expected = info["expected_optimal"]
        match = "✓" if abs(opt - expected) < 0.01 else "✗"

        speedup_str = ""
        if result.pivots == 0:
            speedup_str = "∞"
        elif scratch_pivots > 0:
            speedup_str = f"{scratch_pivots / max(result.pivots, 1):.1f}x"

        print(f"{scenario_name:<16} {change.change_type.name:<8} {result.method_used:<16} "
              f"{result.pivots:>6} {scratch_pivots:>7} {speedup_str:>8} "
              f"{opt:>12.4f} {match:>5}")

        results.append({
            "scenario": scenario_name,
            "change_type": change.change_type.name,
            "method": result.method_used,
            "pivots": result.pivots,
            "scratch_pivots": scratch_pivots,
            "speedup": speedup_str,
            "optimal": opt,
            "expected": expected,
            "match": match,
        })

    # Save CSV
    csv_path = RESULTS_DIR / "reopt_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")
    matched = sum(1 for r in results if r["match"] == "✓")
    print(f"All optimal values correct: {matched}/{len(results)}")


if __name__ == "__main__":
    main()
