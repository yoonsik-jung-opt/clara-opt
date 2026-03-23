import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 2: Reoptimization decision quality — does CLARA correctly predict when to reoptimize?

Usage: python benchmarks/scripts/run_reopt_decision.py
Output: benchmarks/results/exp2_reopt_decision.csv
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
LOSS_THRESHOLD = 0.01


def process_single(task):
    import numpy as np
    from pathlib import Path

    base_file = task["base_file"]
    pert_file = task["pert_file"]
    base_name = task["base_name"]
    pert_info = task["pert_info"]

    try:
        from clara.engine.simplex import RevisedSimplex
        from clara.reopt.detector import ChangeDetector
        from clara.reopt.analyzer import ImpactAnalyzer

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

        # Ground truth: scratch solve new problem
        new_state = RevisedSimplex(new_problem).solve()
        if not new_state.is_optimal:
            return None

        z_scratch = new_state.optimal_value
        x_old = np.array([v.value for v in old_state.variables])
        z_old_in_new = float(new_problem.c @ x_old)
        if old_problem.sense == "minimize":
            z_old_in_new = -z_old_in_new  # engine internally negates

        actual_loss = abs(z_scratch - z_old_in_new) / max(abs(z_scratch), 1e-10)
        oracle = actual_loss > LOSS_THRESHOLD

        return {
            "base_instance": base_name,
            "perturbation": pert_info.get("perturbation", ""),
            "change_type": change.change_type.name,
            "magnitude": pert_info.get("b_magnitude", ""),
            "should_reoptimize": decision.should_reoptimize,
            "recommended_method": decision.recommended_method or "",
            "oguz_bound": f"{decision.oguz_bound:.6f}" if decision.oguz_bound is not None else "",
            "within_sensitivity": decision.within_sensitivity,
            "actual_loss": f"{actual_loss:.6f}",
            "oracle_decision": oracle,
            "correct": decision.should_reoptimize == oracle,
            "delta_max_b": pert_info.get("delta_b_max", ""),
            "delta_max_c": pert_info.get("delta_c_max", ""),
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_dirs = sorted(PERTURBATIONS_DIR.iterdir()) if PERTURBATIONS_DIR.exists() else []
    # Limit to manageable subset for initial run
    base_dirs = [d for d in base_dirs if d.is_dir()][:20]

    print(f"Exp 2: Decision quality on {len(base_dirs)} base instances (workers={args.workers})...")

    tasks = []
    for base_dir in base_dirs:
        base_name = base_dir.name
        # Find the base file
        base_file = None
        for ext in [".mps", ".lp"]:
            p = NETLIB_DIR / f"{base_name}{ext}"
            if p.exists():
                base_file = str(p)
                break
        if base_file is None:
            p = RANDOM_DIR / f"{base_name}.lp"
            if p.exists():
                base_file = str(p)
        if base_file is None:
            continue

        manifest = base_dir / "manifest.csv"
        if not manifest.exists():
            continue

        with open(manifest) as fh:
            perts = list(csv.DictReader(fh))

        for pert_info in perts:
            pert_file = base_dir / f"{pert_info['instance']}.lp"
            if not pert_file.exists():
                continue
            tasks.append({
                "base_file": base_file,
                "pert_file": str(pert_file),
                "base_name": base_name,
                "pert_info": dict(pert_info),
            })

    n_workers = args.workers
    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                results.append(r)
            if (i + 1) % 50 == 0:
                correct = sum(1 for res in results if res["correct"])
                print(f"  [{i+1}/{len(tasks)}] {len(results)} processed, accuracy={correct}/{len(results)}")

    if not results:
        print("No results generated.")
        return

    out = RESULTS_DIR / "exp2_reopt_decision.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    correct = sum(1 for r in results if r["correct"])
    print(f"\nDone: {len(results)} pairs -> {out}")
    print(f"Decision accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")


if __name__ == "__main__":
    main()
