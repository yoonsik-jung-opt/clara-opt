"""Exp 3: Warm-start effectiveness — pivot reduction and speedup.

Usage: python benchmarks/scripts/run_warmstart_benchmark.py
Output: benchmarks/results/exp3_warmstart.csv
"""

import csv
import time
from pathlib import Path

import numpy as np

PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"
NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"


def load_problem(filepath):
    p = Path(filepath)
    if p.suffix.lower() == ".mps":
        from clara.io.mps_parser import read_mps
        return read_mps(p)
    from clara.io.lp_parser import read_lp
    return read_lp(p)


def find_base_file(base_name):
    for ext in [".mps", ".lp"]:
        for d in [NETLIB_DIR, RANDOM_DIR]:
            p = d / f"{base_name}{ext}"
            if p.exists():
                return p
    return None


def process_pair(base_file, pert_file, base_name, pert_info):
    from clara.engine.simplex import RevisedSimplex
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.analyzer import ImpactAnalyzer
    from clara.reopt.reoptimizer import Reoptimizer

    old_problem = load_problem(base_file)
    new_problem = load_problem(pert_file)

    old_state = RevisedSimplex(old_problem).solve()
    if not old_state.is_optimal:
        return None

    change = ChangeDetector().detect(old_problem, new_problem)
    if change is None:
        return None

    decision = ImpactAnalyzer().analyze(old_state, change, old_problem)

    # Warm-start reoptimize
    result = Reoptimizer().reoptimize(old_state, new_problem, change, decision, old_problem=old_problem)

    # Scratch solve
    t0 = time.perf_counter()
    scratch = RevisedSimplex(new_problem).solve()
    t_scratch = time.perf_counter() - t0
    if not scratch.is_optimal:
        return None

    scratch_pivots = scratch.iteration_count
    ws_pivots = result.pivots
    reduction = (scratch_pivots - ws_pivots) / max(scratch_pivots, 1)
    speedup = scratch_pivots / max(ws_pivots, 1)
    opt_match = abs(result.new_state.optimal_value - scratch.optimal_value) < 1e-4

    return {
        "base_instance": base_name,
        "perturbation": pert_info.get("perturbation", ""),
        "change_type": change.change_type.name,
        "magnitude": pert_info.get("b_magnitude", ""),
        "n_vars": old_problem.num_variables,
        "n_cons": old_problem.num_constraints,
        "method_used": result.method_used,
        "warmstart_pivots": ws_pivots,
        "scratch_pivots": scratch_pivots,
        "pivot_reduction": f"{reduction:.2f}",
        "speedup": f"{speedup:.1f}",
        "time_warmstart": f"{result.reopt_time_seconds:.6f}",
        "time_scratch": f"{t_scratch:.6f}",
        "optimal_match": opt_match,
        "z_warmstart": f"{result.new_state.optimal_value:.6f}",
        "z_scratch": f"{scratch.optimal_value:.6f}",
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    base_dirs = sorted(d for d in PERTURBATIONS_DIR.iterdir() if d.is_dir())[:20]

    print(f"Exp 3: Warm-start on {len(base_dirs)} base instances...")

    for base_dir in base_dirs:
        base_name = base_dir.name
        base_file = find_base_file(base_name)
        if not base_file:
            continue

        manifest = base_dir / "manifest.csv"
        if not manifest.exists():
            continue

        with open(manifest) as fh:
            perts = list(csv.DictReader(fh))

        for pi in perts:
            pf = base_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue
            try:
                r = process_pair(base_file, pf, base_name, pi)
                if r:
                    results.append(r)
            except Exception:
                pass

        if results:
            print(f"  [{base_name}] {len(results)} total")

    if not results:
        print("No results.")
        return

    out = RESULTS_DIR / "exp3_warmstart.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    speedups = [float(r["speedup"]) for r in results if r["method_used"] != "scratch"]
    matched = sum(1 for r in results if r["optimal_match"])
    print(f"\nDone: {len(results)} pairs → {out}")
    print(f"Optimal match: {matched}/{len(results)}")
    if speedups:
        print(f"Speedup (non-scratch): mean={np.mean(speedups):.1f}x, max={max(speedups):.1f}x")


if __name__ == "__main__":
    main()
