"""Exp 2: Reoptimization decision quality — does CLARA correctly predict when to reoptimize?

Usage: python benchmarks/scripts/run_reopt_decision.py
Output: benchmarks/results/exp2_reopt_decision.csv
"""

import csv
import time
from pathlib import Path

import numpy as np

PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"
NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
LOSS_THRESHOLD = 0.01


def load_problem(filepath):
    p = Path(filepath)
    if p.suffix.lower() == ".mps":
        from clara.io.mps_parser import read_mps
        return read_mps(p)
    from clara.io.lp_parser import read_lp
    return read_lp(p)


def find_base_file(base_name):
    """Find the base LP/MPS file."""
    for ext in [".mps", ".lp"]:
        p = NETLIB_DIR / f"{base_name}{ext}"
        if p.exists():
            return p
    p = RANDOM_DIR / f"{base_name}.lp"
    if p.exists():
        return p
    return None


def process_pair(base_file, pert_file, base_name, pert_info):
    from clara.engine.simplex import RevisedSimplex
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.analyzer import ImpactAnalyzer

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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total_pairs = 0

    base_dirs = sorted(PERTURBATIONS_DIR.iterdir()) if PERTURBATIONS_DIR.exists() else []
    # Limit to manageable subset for initial run
    base_dirs = [d for d in base_dirs if d.is_dir()][:20]

    print(f"Exp 2: Decision quality on {len(base_dirs)} base instances...")

    for base_dir in base_dirs:
        base_name = base_dir.name
        base_file = find_base_file(base_name)
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

            total_pairs += 1
            try:
                r = process_pair(base_file, pert_file, base_name, pert_info)
                if r:
                    results.append(r)
            except Exception as e:
                pass

        if len(results) % 50 == 0 and results:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [{base_name}] {len(results)} processed, accuracy={correct}/{len(results)}")

    if not results:
        print("No results generated.")
        return

    out = RESULTS_DIR / "exp2_reopt_decision.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    correct = sum(1 for r in results if r["correct"])
    print(f"\nDone: {len(results)} pairs → {out}")
    print(f"Decision accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")


if __name__ == "__main__":
    main()
