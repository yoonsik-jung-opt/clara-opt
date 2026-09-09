import os
os.environ["OMP_NUM_THREADS"] = "1"

"""Small-magnitude Oğuz threshold validation (Online Supp. §7).

Exercises the screening step (Step 2) of the impact analyzer in the
regime where it actually fires: perturbation magnitudes
delta in {0.5%, 1%, 2%}, producing screening values
OC-hat = 2d/(1+d) in {0.995%, 1.98%, 3.92%}, which straddle the
epsilon = 1% and epsilon = 5% thresholds. Ground truth is a scratch
solve of each perturbed instance (relative value-change tolerance
1e-4). Decision counts are hardware-independent.

Usage: python benchmarks/scripts/run_oguz_threshold_validation.py
Output: benchmarks/results/exp_oguz_threshold.csv
"""

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent
RANDOM = SCRIPTS.parent / "instances" / "random"
RESULTS = SCRIPTS.parent / "results"

MAGNITUDES = [0.005, 0.01, 0.02]
TYPES = ["R", "C"]
SEEDS = [42, 123, 456]
THRESHOLDS = [0.01, 0.05]
ORACLE_TOL = 1e-4


def process_base(base_name):
    sys.path.insert(0, str(SCRIPTS))
    from generate_perturbations import generate_perturbation
    from clara.io.lp_parser import read_lp
    from clara.engine import HiGHSBackend
    from clara.model.problem import LPProblem
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.analyzer import ImpactAnalyzer

    try:
        problem = read_lp(RANDOM / f"{base_name}.lp")
        old_state = HiGHSBackend().solve(problem)
        if not old_state.is_optimal:
            return []
    except Exception:
        return []

    rows = []
    for ptype in TYPES:
        for mag in MAGNITUDES:
            b_mag = mag if ptype == "R" else 0.0
            c_mag = mag if ptype == "C" else 0.0
            for seed in SEEDS:
                out = generate_perturbation(
                    problem, f"{ptype}_{mag}", ptype, b_mag, c_mag, seed)
                if out is None:
                    continue
                new_c, new_b, delta_c, delta_b = out
                new_problem = LPProblem(
                    c=new_c, A=problem.A.copy(), b=new_b,
                    var_names=list(problem.var_names),
                    constraint_names=list(problem.constraint_names),
                    name=f"{base_name}_{ptype}_{mag}_s{seed}",
                    sense=problem.sense,
                    lower_bounds=problem.lower_bounds.copy(),
                    upper_bounds=problem.upper_bounds.copy(),
                )
                change = ChangeDetector().detect(problem, new_problem)
                if change is None:
                    continue

                try:
                    new_state = HiGHSBackend().solve(new_problem)
                    if not new_state.is_optimal:
                        continue
                    rel = (abs(new_state.optimal_value - old_state.optimal_value)
                           / max(abs(old_state.optimal_value), 1.0))
                    oracle = rel > ORACLE_TOL
                    # Opportunity cost of keeping the stale solution x*
                    # under the new objective (the quantity the Oguz
                    # bound certifies; for Type R x* may be infeasible,
                    # in which case the entry is left blank).
                    oc = ""
                    if ptype == "C":
                        x_old = np.array([v.value for v in old_state.variables])
                        z_stale = float(new_problem.c @ x_old)
                        sgn = 1.0 if problem.sense != "minimize" else -1.0
                        oc_val = sgn * (new_state.optimal_value - z_stale) / max(abs(new_state.optimal_value), 1e-12)
                        oc = f"{oc_val:.6e}"
                except Exception:
                    continue

                for eps in THRESHOLDS:
                    decision = ImpactAnalyzer(oguz_threshold=eps).analyze(
                        old_state, change, problem)
                    rec = bool(decision.should_reoptimize)
                    skip_source = ""
                    if not rec:
                        skip_source = ("sensitivity_range"
                                       if decision.within_sensitivity
                                       else "screening")
                    rows.append({
                        "base_instance": base_name,
                        "type": ptype,
                        "magnitude": mag,
                        "seed": seed,
                        "epsilon": eps,
                        "oguz_bound": (f"{decision.oguz_bound:.6f}"
                                       if decision.oguz_bound is not None else ""),
                        "should_reoptimize": rec,
                        "skip_source": skip_source,
                        "oracle_decision": oracle,
                        "rel_value_change": f"{rel:.6e}",
                        "opportunity_cost": oc,
                    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RANDOM / "manifest.csv") as fh:
        bases = [r["instance"] for r in csv.DictReader(fh)
                 if r.get("match", "").strip() == "✓"]

    print(f"Oğuz threshold validation on {len(bases)} bases "
          f"({len(TYPES)} types x {len(MAGNITUDES)} magnitudes x "
          f"{len(SEEDS)} seeds x {len(THRESHOLDS)} thresholds)...")

    all_rows = []
    with mp.Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_base, bases)):
            all_rows.extend(rows)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(bases)}] {len(all_rows)} trials")

    all_rows.sort(key=lambda r: (r["base_instance"], r["type"],
                                 r["magnitude"], r["seed"], r["epsilon"]))
    out = RESULTS / "exp_oguz_threshold.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Done: {len(all_rows)} decision rows -> {out}\n")

    # Summary
    for eps in THRESHOLDS:
        rr = [r for r in all_rows if r["epsilon"] == eps]
        tp = sum(1 for r in rr if r["should_reoptimize"] and r["oracle_decision"])
        fp = sum(1 for r in rr if r["should_reoptimize"] and not r["oracle_decision"])
        fn = sum(1 for r in rr if not r["should_reoptimize"] and r["oracle_decision"])
        tn = sum(1 for r in rr if not r["should_reoptimize"] and not r["oracle_decision"])
        sr = sum(1 for r in rr if r["skip_source"] == "sensitivity_range")
        sc = sum(1 for r in rr if r["skip_source"] == "screening")
        fn_screen = sum(1 for r in rr if r["skip_source"] == "screening"
                        and r["oracle_decision"])
        acc = (tp + tn) / len(rr) if rr else 0
        sc_rows = [r for r in rr if r["skip_source"] == "screening"]
        max_val = max((float(r["rel_value_change"]) for r in sc_rows), default=0.0)
        max_oc = max((float(r["opportunity_cost"]) for r in sc_rows
                      if r["opportunity_cost"]), default=0.0)
        viol = sum(1 for r in sc_rows if r["opportunity_cost"]
                   and float(r["opportunity_cost"]) > eps + 1e-9)
        print(f"eps={eps:.0%}: pairs={len(rr)} TP={tp} TN={tn} FP={fp} FN={fn} "
              f"acc={acc:.1%} | skips: range={sr} screening={sc} "
              f"(screening FN={fn_screen}; max value change {max_val:.4%}; "
              f"max opportunity cost {max_oc:.4%}; contract violations {viol})")


if __name__ == "__main__":
    main()
