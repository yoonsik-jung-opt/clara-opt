import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 3: Warm-start effectiveness — pivot reduction and speedup.

Usage: python benchmarks/scripts/run_warmstart_benchmark.py
Output: benchmarks/results/exp3_warmstart.csv
"""

import argparse
import csv
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"
NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"


def process_single(task):
    import time
    from pathlib import Path

    base_file = task["base_file"]
    pert_file = task["pert_file"]
    base_name = task["base_name"]
    pert_info = task["pert_info"]

    try:
        from clara.engine.simplex import RevisedSimplex
        from clara.reopt.detector import ChangeDetector
        from clara.reopt.analyzer import ImpactAnalyzer
        from clara.reopt.reoptimizer import Reoptimizer

        def load_problem(filepath):
            p = Path(filepath)
            if p.suffix.lower() == ".mps":
                from clara.io.mps_parser import read_mps
                return read_mps(p)
            from clara.io.lp_parser import read_lp
            return read_lp(p)

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
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use random LP instances (match=✓, n≤50)
    tasks = []
    rand_manifest = RANDOM_DIR / "manifest.csv"
    bases = []
    if rand_manifest.exists():
        with open(rand_manifest) as fh:
            bases = [r["instance"] for r in csv.DictReader(fh)
                     if r.get("match", "").strip() == "✓" and int(r["n_vars"]) <= 50]

    for base_name in bases:
        base_file = RANDOM_DIR / f"{base_name}.lp"
        if not base_file.exists():
            continue
        pert_dir = PERTURBATIONS_DIR / base_name
        pert_manifest = pert_dir / "manifest.csv"
        if not pert_manifest.exists():
            continue
        with open(pert_manifest) as fh:
            perts = [r for r in csv.DictReader(fh)
                     if r.get("type", "") in ("R", "C")]
        for pi in perts:
            pf = pert_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue
            tasks.append({
                "base_file": str(base_file),
                "pert_file": str(pf),
                "base_name": base_name,
                "pert_info": dict(pi),
            })

    print(f"Exp 3: Warm-start on {len(tasks)} pairs (workers={args.workers})...")

    n_workers = args.workers
    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                results.append(r)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(tasks)}] {len(results)} successful")

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
    print(f"\nDone: {len(results)} pairs -> {out}")
    print(f"Optimal match: {matched}/{len(results)}")
    if speedups:
        print(f"Speedup (non-scratch): mean={np.mean(speedups):.1f}x, max={max(speedups):.1f}x")


if __name__ == "__main__":
    main()
