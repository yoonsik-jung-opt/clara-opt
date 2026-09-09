import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 9: Ablation of the warm-start method-selection rule.

All arms start HiGHS from the same retained basis on the same perturbed
problem; they differ only in the simplex strategy forced:

  rule          CLARA's rule -- primal simplex when the retained basis is
                still primal feasible, dual simplex when it is primal
                infeasible but dual feasible (Reoptimizer._warm_start)
  always_dual   simplex_strategy = 1 (HiGHS default)
  always_primal simplex_strategy = 4
  choose        simplex_strategy = 0 (HiGHS decides)
  cold          no starting basis, HiGHS default strategy

For each arm the HiGHS simplex run time (SolveState.simplex_time_seconds)
and iteration count are recorded, so the arms are compared at the solver
level over the same span. Arms are interleaved per pair.

Usage: python benchmarks/scripts/run_method_ablation.py [--workers N]
Output: benchmarks/results/exp9_method_ablation.csv
"""

import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np

PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"

ARMS = ("rule", "always_dual", "always_primal", "choose", "cold")


def process_single(task):
    try:
        from clara.engine import HiGHSBackend
        from clara.engine.highs_backend import (
            SIMPLEX_STRATEGY_CHOOSE, SIMPLEX_STRATEGY_DUAL, SIMPLEX_STRATEGY_PRIMAL,
        )
        from clara.io.lp_parser import read_lp
        from clara.reopt.detector import ChangeDetector
        from clara.reopt.reoptimizer import Reoptimizer
        from clara.reopt.types import ReoptDecision

        old_problem = read_lp(task["base_file"])
        new_problem = read_lp(task["pert_file"])
        backend = HiGHSBackend()
        old_state = backend.solve(old_problem)
        if not old_state.is_optimal or old_state.basis_inverse is None:
            return None
        change = ChangeDetector().detect(old_problem, new_problem)
        if change is None:
            return None
        basis = list(old_state.basis_indices)

        def timed(state):
            return (state.simplex_time_seconds or 0.0), int(state.iteration_count), state.optimal_value

        fixed = {
            "always_dual": SIMPLEX_STRATEGY_DUAL,
            "always_primal": SIMPLEX_STRATEGY_PRIMAL,
            "choose": SIMPLEX_STRATEGY_CHOOSE,
        }
        row = {
            "base_instance": task["base_name"],
            "perturbation": task["pert_info"].get("perturbation", ""),
            "change_type": change.change_type.name,
            "n_vars": old_problem.num_variables,
            "n_cons": old_problem.num_constraints,
        }
        z_ref = None
        for arm in ARMS:
            if arm == "rule":
                dec = ReoptDecision(should_reoptimize=True, reason="ablation",
                                    recommended_method="warm_start")
                res = Reoptimizer().reoptimize(old_state, new_problem, change, dec,
                                               old_problem=old_problem)
                t, it, z = timed(res.new_state)
                row["rule_method"] = res.method_used
            elif arm == "cold":
                t, it, z = timed(backend.solve(new_problem))
            else:
                t, it, z = timed(backend.solve(new_problem, initial_basis=basis,
                                               simplex_strategy=fixed[arm]))
            if z_ref is None:
                z_ref = z
            row[f"{arm}_time"] = f"{t:.6f}"
            row[f"{arm}_iters"] = it
            row[f"{arm}_match"] = abs(z - z_ref) < max(abs(z_ref) * 1e-6, 1e-4)
        return row
    except Exception as e:  # pragma: no cover - benchmark robustness
        return {"base_instance": task["base_name"], "error": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    with open(RANDOM_DIR / "manifest.csv") as fh:
        bases = [r["instance"] for r in csv.DictReader(fh)
                 if r.get("match", "").strip() == "✓" and int(r["n_vars"]) <= 50]
    for base_name in bases:
        base_file = RANDOM_DIR / f"{base_name}.lp"
        pert_manifest = PERTURBATIONS_DIR / base_name / "manifest.csv"
        if not base_file.exists() or not pert_manifest.exists():
            continue
        with open(pert_manifest) as fh:
            perts = [r for r in csv.DictReader(fh) if r.get("type", "") in ("R", "C")]
        for pi in perts:
            pf = PERTURBATIONS_DIR / base_name / f"{pi['instance']}.lp"
            if pf.exists():
                tasks.append({"base_file": str(base_file), "pert_file": str(pf),
                              "base_name": base_name, "pert_info": dict(pi)})

    print(f"Exp 9: method-selection ablation on {len(tasks)} pairs (workers={args.workers})...")
    results = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(tasks)}]")
    errors = [r for r in results if "error" in r]
    results = [r for r in results if "error" not in r]
    if errors:
        print(f"WARNING: {len(errors)} pairs failed; first error: {errors[0]['error']}")
    if not results:
        return
    out = RESULTS_DIR / "exp9_method_ablation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Done: {len(results)} pairs -> {out}")
    summarize(results)


def summarize(results):
    def med(x):
        return float(np.median(x)) if len(x) else float("nan")

    groups = {"ALL": results}
    for ct in ("TYPE_R", "TYPE_C"):
        groups[ct] = [r for r in results if r["change_type"] == ct]
    print("\nmedian pair-level speedup vs cold (simplex run time) | median iterations")
    header = f"{'group':8} {'n':>5} " + " ".join(f"{a:>14}" for a in ARMS)
    print(header)
    for name, rows in groups.items():
        if not rows:
            continue
        cells = []
        for arm in ARMS:
            t_arm = np.array([float(r[f"{arm}_time"]) for r in rows])
            t_cold = np.array([float(r["cold_time"]) for r in rows])
            ok = (t_arm > 0) & (t_cold > 0)
            sp = med(t_cold[ok] / t_arm[ok])
            it = med([int(r[f"{arm}_iters"]) for r in rows])
            cells.append(f"{sp:6.2f}x {it:5.0f}it")
        print(f"{name:8} {len(rows):5d} " + " ".join(f"{c:>14}" for c in cells))
    mism = {arm: sum(1 for r in results if str(r[f"{arm}_match"]) != "True") for arm in ARMS}
    print("objective mismatches:", mism)


if __name__ == "__main__":
    main()
