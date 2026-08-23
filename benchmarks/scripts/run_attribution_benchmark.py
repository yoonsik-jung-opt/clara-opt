import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 7: Attribution decomposition benchmark for Type RC changes.

Usage: python benchmarks/scripts/run_attribution_benchmark.py
Output: benchmarks/results/exp7_attribution.csv
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
        from clara.engine import HiGHSBackend
        from clara.reopt.attribution import ChangeAttributor
        from clara.reopt.detector import ChangeDetector

        def load_problem(filepath):
            p = Path(filepath)
            if p.suffix.lower() == ".mps":
                from clara.io.mps_parser import read_mps
                return read_mps(p)
            from clara.io.lp_parser import read_lp
            return read_lp(p)

        old_problem = load_problem(base_file)
        old_state = HiGHSBackend().solve(old_problem)
        if not old_state.is_optimal:
            return {"base_instance": base_name, "error": f"old: {old_state.status.name}"}

        new_problem = load_problem(pert_file)
        new_state = HiGHSBackend().solve(new_problem)
        if not new_state.is_optimal:
            # HiGHS fallback
            from clara.engine.highs_backend import HiGHSBackend
            new_state = HiGHSBackend().solve(new_problem)
            if not new_state.is_optimal:
                return {"base_instance": base_name, "error": f"new: {new_state.status.name}"}

        change = ChangeDetector().detect(old_problem, new_problem)
        if change is None:
            return {"base_instance": base_name, "error": "no change detected"}

        attr = ChangeAttributor().attribute(
            old_state, new_state, old_problem, new_problem, change
        )
        dz = attr.delta_z
        adz = max(abs(dz), 1e-10)

        top_rhs = max(attr.rhs_contributions.items(), key=lambda x: abs(x[1]), default=("", 0))
        top_obj = max(attr.obj_contributions.items(), key=lambda x: abs(x[1]), default=("", 0))

        return {
            "base_instance": base_name,
            "perturbation": pi["perturbation"],
            "magnitude": pi.get("b_magnitude", ""),
            "n_vars": old_problem.num_variables,
            "n_cons": old_problem.num_constraints,
            "delta_z": f"{dz:.6f}",
            "rhs_effect": f"{attr.rhs_effect:.6f}",
            "obj_effect": f"{attr.obj_effect:.6f}",
            "interaction_effect": f"{attr.interaction_effect:.6f}",
            "rhs_pct": f"{abs(attr.rhs_effect)/adz*100:.1f}",
            "obj_pct": f"{abs(attr.obj_effect)/adz*100:.1f}",
            "interaction_pct": f"{abs(attr.interaction_effect)/adz*100:.1f}",
            "shapley_b": f"{attr.shapley_b:.6f}" if attr.shapley_b is not None else "",
            "shapley_c": f"{attr.shapley_c:.6f}" if attr.shapley_c is not None else "",
            "basis_preserved": attr.basis_preserved,
            "nonlinearity": f"{attr.nonlinearity:.6f}",
            "top_rhs_param": top_rhs[0],
            "top_rhs_value": f"{top_rhs[1]:.6f}",
            "top_obj_param": top_obj[0],
            "top_obj_value": f"{top_obj[1]:.6f}",
        }
    except Exception as e:
        import traceback
        return {"base_instance": base_name, "perturbation": pi.get("perturbation",""),
                "error": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use random LP instances (match=✓, n≤50) for reliability
    tasks = []
    rand_manifest = RANDOM_DIR / "manifest.csv"
    if rand_manifest.exists():
        with open(rand_manifest) as fh:
            bases = [r["instance"] for r in csv.DictReader(fh)
                     if r.get("match", "").strip() == "✓" and int(r["n_vars"]) <= 50]
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

    print(f"Exp 7: Attribution on {len(tasks)} RC pairs (workers={args.workers})...")

    n_workers = args.workers
    all_results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                all_results.append(r)
            if (i + 1) % 20 == 0:
                valid = [x for x in all_results if "error" not in x]
                print(f"  [{i+1}/{len(tasks)}] {len(valid)} successful, {len(all_results)-len(valid)} errors")

    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]

    if errors:
        print(f"\nErrors: {len(errors)} instances failed")
        for e in errors[:3]:
            print(f"  {e.get('base_instance','?')}: {e.get('error','?')}")

    if not valid:
        print("No valid results.")
        return

    out = RESULTS_DIR / "exp7_attribution.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=valid[0].keys())
        w.writeheader()
        w.writerows(valid)

    preserved = sum(1 for r in valid if r.get("basis_preserved"))
    print(f"\nDone: {len(valid)} RC pairs -> {out}")
    print(f"Basis preserved: {preserved}/{len(valid)}")
    rhs_pcts = [float(r["rhs_pct"]) for r in valid]
    obj_pcts = [float(r["obj_pct"]) for r in valid]
    print(f"Mean RHS contribution: {np.mean(rhs_pcts):.1f}%")
    print(f"Mean OBJ contribution: {np.mean(obj_pcts):.1f}%")


if __name__ == "__main__":
    main()
