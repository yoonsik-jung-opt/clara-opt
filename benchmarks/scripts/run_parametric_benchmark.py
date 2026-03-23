import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 4: Parametric LP path analysis for Type RC compound changes.

Usage: python benchmarks/scripts/run_parametric_benchmark.py
Output: benchmarks/results/exp4_parametric.csv, exp4_breakpoints.csv
"""

import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np

PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"
NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"


def process_single(task):
    from pathlib import Path

    base_file = task["base_file"]
    pert_file = task["pert_file"]
    base_name = task["base_name"]
    pi = task["pert_info"]

    try:
        from clara.engine.simplex import RevisedSimplex
        from clara.reopt.detector import ChangeDetector
        from clara.reopt.parametric import ParametricLPSolver

        def load_problem(filepath):
            p = Path(filepath)
            if p.suffix.lower() == ".mps":
                from clara.io.mps_parser import read_mps
                return read_mps(p)
            from clara.io.lp_parser import read_lp
            return read_lp(p)

        old_problem = load_problem(base_file)
        old_state = RevisedSimplex(old_problem).solve()
        if not old_state.is_optimal or old_state.basis_inverse is None:
            return None

        new_problem = load_problem(pert_file)
        change = ChangeDetector().detect(old_problem, new_problem)
        if change is None or change.delta_b is None or change.delta_c is None:
            return None

        pr = ParametricLPSolver().solve(
            old_state, old_problem, change.delta_b, change.delta_c
        )

        # Scratch verify
        scratch = RevisedSimplex(new_problem).solve()
        z_scratch = scratch.optimal_value if scratch.is_optimal else float("nan")
        opt_match = abs(pr.new_state.optimal_value - z_scratch) < 1e-4 if scratch.is_optimal else False

        main_result = {
            "base_instance": base_name,
            "perturbation": pi["perturbation"],
            "magnitude": pi.get("b_magnitude", ""),
            "n_vars": old_problem.num_variables,
            "n_cons": old_problem.num_constraints,
            "num_breakpoints": pr.num_breakpoints,
            "theta_first": f"{pr.breakpoints[0].theta:.4f}" if pr.breakpoints else "",
            "num_pivots": pr.num_pivots,
            "path_monotone": pr.path_monotone,
            "optimal_match": opt_match,
            "z_parametric": f"{pr.new_state.optimal_value:.6f}",
            "z_scratch": f"{z_scratch:.6f}",
        }

        bp_results = []
        for k, bp in enumerate(pr.breakpoints):
            bp_results.append({
                "base_instance": base_name,
                "perturbation": pi["perturbation"],
                "bp_index": k,
                "theta": f"{bp.theta:.6f}",
                "bp_type": bp.breakpoint_type,
                "leaving": bp.leaving_var or "",
                "entering": bp.entering_var or "",
                "objective": f"{bp.objective_value:.6f}",
            })

        return {"main": main_result, "breakpoints": bp_results}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use random LP instances (match=✓, n≤30) for reliability
    tasks = []
    rand_manifest = RANDOM_DIR / "manifest.csv"
    if rand_manifest.exists():
        with open(rand_manifest) as fh:
            bases = [r["instance"] for r in csv.DictReader(fh)
                     if r.get("match", "").strip() == "✓" and int(r["n_vars"]) <= 30]
    else:
        bases = []

    for base_name in bases:
        base_file = RANDOM_DIR / f"{base_name}.lp"
        if not base_file.exists():
            continue
        pert_dir = PERTURBATIONS_DIR / base_name
        pert_manifest = pert_dir / "manifest.csv"
        if not pert_manifest.exists():
            continue
        with open(pert_manifest) as fh:
            perts = [r for r in csv.DictReader(fh) if r.get("type", "") == "RC"]
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

    print(f"Exp 4: Parametric LP on {len(tasks)} RC pairs (workers={args.workers})...")

    n_workers = args.workers
    results = []
    bp_results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                results.append(r["main"])
                bp_results.extend(r["breakpoints"])
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(tasks)}] {len(results)} successful")

    if results:
        out = RESULTS_DIR / "exp4_parametric.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        matched = sum(1 for r in results if r["optimal_match"])
        mismatched = [r for r in results if not r["optimal_match"]]
        print(f"Done: {len(results)} RC pairs -> {out}")
        print(f"Optimal match: {matched}/{len(results)}")
        for r in mismatched[:5]:
            print(f"  MISMATCH: {r['base_instance']} {r['perturbation']}: "
                  f"z_param={r['z_parametric']}, z_scratch={r['z_scratch']}, "
                  f"bp={r['num_breakpoints']}")
    else:
        print("No RC perturbations found.")

    if bp_results:
        out2 = RESULTS_DIR / "exp4_breakpoints.csv"
        with open(out2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=bp_results[0].keys())
            w.writeheader()
            w.writerows(bp_results)
        print(f"Breakpoints: {len(bp_results)} -> {out2}")


if __name__ == "__main__":
    main()
