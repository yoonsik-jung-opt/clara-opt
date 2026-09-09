import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 12: Cost of a certified skip versus a solver call.

On every pair that the decision experiment (exp2) skips and whose
retained basis passes the certificate (exp2b, skip_method_used ==
'recompute'), measures

  recompute_time   certificate + zero-pivot recompute (Reoptimizer,
                   method 'recompute'; no solver call)
  warm_time        HiGHSBackend.solve with the retained basis as an
                   advanced starting basis (model build + run), i.e.
                   solve_time_seconds
  cold_time        HiGHSBackend.solve from scratch (solve_time_seconds)

Each measurement is the best of --reps repetitions, interleaved per pair.

Usage: python benchmarks/scripts/run_skip_cost.py [--reps 3]
Output: benchmarks/results/exp12_skip_cost.csv
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RANDOM = ROOT / "benchmarks" / "instances" / "random"
PERT = ROOT / "benchmarks" / "instances" / "perturbations"
RESULTS = ROOT / "benchmarks" / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--source", default=str(RESULTS / "exp2b_skip_realized_loss.csv"))
    args = parser.parse_args()

    from clara.io.lp_parser import read_lp
    from clara.engine import HiGHSBackend
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.reoptimizer import Reoptimizer
    from clara.reopt.types import ReoptDecision

    rows = [r for r in csv.DictReader(open(args.source))
            if r["skip_method_used"] == "recompute"]
    print(f"Exp 12: skip cost on {len(rows)} certified-recompute pairs (reps={args.reps})...")
    cache = {}
    out = []
    backend = HiGHSBackend()
    for k, r in enumerate(rows):
        base = r["base_instance"]
        if base not in cache:
            p = read_lp(RANDOM / f"{base}.lp")
            cache[base] = (p, backend.solve(p))
        old_p, old_s = cache[base]
        cand = sorted((PERT / base).glob(f"{base}_{r['perturbation']}*.lp"))
        alt = [c for c in cand if c.stem == r["perturbation"]]
        new_p = read_lp(alt[0] if alt else cand[0])
        change = ChangeDetector().detect(old_p, new_p)
        dec = ReoptDecision(should_reoptimize=False, reason="skip cost",
                            recommended_method="recompute")
        basis = list(old_s.basis_indices)
        t_rec = t_warm = t_cold = float("inf")
        method = ""
        for _ in range(args.reps):
            t0 = time.perf_counter()
            res = Reoptimizer().reoptimize(old_s, new_p, change, dec, old_problem=old_p)
            t_rec = min(t_rec, time.perf_counter() - t0)
            method = res.method_used
            ws = backend.solve(new_p, initial_basis=basis)
            t_warm = min(t_warm, ws.solve_time_seconds)
            cs = backend.solve(new_p)
            t_cold = min(t_cold, cs.solve_time_seconds)
        out.append({
            "base_instance": base, "perturbation": r["perturbation"],
            "change_type": r["change_type"], "n_vars": old_p.num_variables,
            "n_cons": old_p.num_constraints, "method_used": method,
            "recompute_time": f"{t_rec:.6f}", "warm_time": f"{t_warm:.6f}",
            "cold_time": f"{t_cold:.6f}",
        })
        if (k + 1) % 100 == 0:
            print(f"  [{k+1}/{len(rows)}]")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "exp12_skip_cost.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Done: {len(out)} pairs -> {path}")

    rec = np.array([float(r["recompute_time"]) for r in out]) * 1e3
    warm = np.array([float(r["warm_time"]) for r in out]) * 1e3
    cold = np.array([float(r["cold_time"]) for r in out]) * 1e3
    sizes = np.array([int(r["n_cons"]) for r in out])
    print(f"\nmedian ms: recompute {np.median(rec):.3f} | warm-start call {np.median(warm):.3f} "
          f"| cold {np.median(cold):.3f}")
    print(f"median per-pair ratio warm/recompute {np.median(warm / rec):.1f}x, "
          f"cold/recompute {np.median(cold / rec):.1f}x")
    for m in sorted(set(sizes)):
        k = sizes == m
        print(f"  m={m:4d} n={k.sum():4d} recompute {np.median(rec[k]):.3f} ms  "
              f"warm {np.median(warm[k]):.3f} ms  ratio {np.median(warm[k] / rec[k]):.1f}x")


if __name__ == "__main__":
    main()
