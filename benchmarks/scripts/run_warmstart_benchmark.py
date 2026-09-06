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
        import numpy as np
        from clara.engine import HiGHSBackend
        from clara.reopt.detector import ChangeDetector
        from clara.reopt.reoptimizer import Reoptimizer
        from clara.reopt.types import ReoptDecision

        def load_problem(filepath):
            p = Path(filepath)
            if p.suffix.lower() == ".mps":
                from clara.io.mps_parser import read_mps
                return read_mps(p)
            from clara.io.lp_parser import read_lp
            return read_lp(p)

        old_problem = load_problem(base_file)
        new_problem = load_problem(pert_file)

        old_state = HiGHSBackend().solve(old_problem)
        if not old_state.is_optimal:
            return None

        change = ChangeDetector().detect(old_problem, new_problem)
        if change is None:
            return None

        # Compute magnitude
        magnitude = 0.0
        if change.delta_c is not None:
            magnitude = max(magnitude, float(np.max(np.abs(change.delta_c))))
        if change.delta_b is not None:
            magnitude = max(magnitude, float(np.max(np.abs(change.delta_b))))

        # Force warm-start (bypass ImpactAnalyzer to avoid "none" method)
        forced_decision = ReoptDecision(
            should_reoptimize=True,
            reason="forced by benchmark",
            recommended_method="warm_start",
        )

        # Warm-start reoptimize
        result = Reoptimizer().reoptimize(
            old_state, new_problem, change, forced_decision, old_problem=old_problem
        )

        # Scratch solve. Time the same span as the warm start (model
        # build + simplex run, reported by the backend as
        # solve_time_seconds) so the two sides are comparable; ranging
        # and basis-inverse reconstruction are excluded on both sides.
        scratch = HiGHSBackend().solve(new_problem)
        t_scratch = scratch.solve_time_seconds
        if not scratch.is_optimal:
            return None

        z_warm = result.new_state.optimal_value
        z_scratch_val = scratch.optimal_value
        scratch_pivots = scratch.iteration_count
        ws_pivots = result.pivots
        reduction = (scratch_pivots - ws_pivots) / max(scratch_pivots, 1)
        speedup = scratch_pivots / max(ws_pivots, 1)
        opt_match = abs(z_warm - z_scratch_val) < max(abs(z_scratch_val) * 1e-6, 1e-4)

        return {
            "base_instance": base_name,
            "perturbation": pert_info.get("perturbation", ""),
            "change_type": change.change_type.name,
            "magnitude": f"{magnitude:.6f}",
            "n_vars": old_problem.num_variables,
            "n_cons": old_problem.num_constraints,
            "method_used": result.method_used,
            "warmstart_pivots": ws_pivots,
            "scratch_pivots": scratch_pivots,
            "pivot_reduction": f"{reduction:.2f}",
            "speedup": f"{speedup:.1f}",
            "time_warmstart": f"{result.reopt_time_seconds:.6f}",
            "time_scratch": f"{t_scratch:.6f}",
            "run_time_warmstart": f"{(result.new_state.simplex_time_seconds or 0.0):.6f}",
            "run_time_scratch": f"{(scratch.simplex_time_seconds or 0.0):.6f}",
            "optimal_match": opt_match,
            "z_warmstart": f"{z_warm:.6f}",
            "z_scratch": f"{z_scratch_val:.6f}",
        }
    except Exception as e:
        return {"base_instance": base_name, "error": str(e)[:200]}


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

    errors = [r for r in results if "error" in r]
    results = [r for r in results if "error" not in r]
    if errors:
        print(f"\nWARNING: {len(errors)} pairs failed; first error: {errors[0]['error']}")
    if not results:
        print("No successful results; nothing written.")
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
