import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 10: Cost of the simultaneous sensitivity region as a function of m.

For random LPs of increasing size (same generator as the benchmark
suite) and the Netlib instances, records

  simplex_time      HiGHS simplex run on the base problem
  solve_time        full HiGHSBackend.solve (model build, simplex,
                    ranging, B^-1 reconstruction, exact rhs ranges)
  chebyshev_time    SimultaneousRegionAnalyzer.analyze: OAT extraction,
                    the Chebyshev-center LP of S (m_aug+1 variables,
                    3 m_aug rows) and, on degenerate bases, the
                    face-restricted Chebyshev LP

so the region cost can be compared with a single simplex solve.

Usage: python benchmarks/scripts/run_chebyshev_timing.py [--max-m 2000] [--reps 3]
Output: benchmarks/results/exp10_chebyshev_timing.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "scripts"))
RESULTS_DIR = ROOT / "benchmarks" / "results"
NETLIB_DIR = ROOT / "benchmarks" / "netlib" / "mps"

SIZES = [(10, 10), (20, 40), (50, 100), (100, 200), (200, 400),
         (500, 1000), (1000, 2000), (2000, 4000)]
SEEDS = (42, 123, 456)


def time_instance(problem, reps):
    from clara.engine import HiGHSBackend
    from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer

    backend = HiGHSBackend()
    analyzer = SimultaneousRegionAnalyzer()
    best = {}
    state = None
    for _ in range(reps):
        t0 = time.perf_counter()
        state = backend.solve(problem)
        t_solve = time.perf_counter() - t0
        if not state.is_optimal or state.basis_inverse is None:
            return None, None
        t0 = time.perf_counter()
        region = analyzer.analyze(state, problem)
        t_cheb = time.perf_counter() - t0
        rec = {"simplex_time": state.simplex_time_seconds or 0.0,
               "solve_time": t_solve, "chebyshev_time": t_cheb}
        for k, v in rec.items():
            best[k] = min(best.get(k, float("inf")), v)
    m_aug = state.basis_inverse.shape[0]
    best.update({
        "m_aug": m_aug,
        "degenerate_basics": state.degenerate_count,
        "n_degenerate_params": region.n_degenerate_params,
        "chebyshev_radius": region.chebyshev_radius,
        "chebyshev_radius_face": region.chebyshev_radius_face,
        "simultaneity_ratio": region.simultaneity_ratio,
    })
    return best, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-m", type=int, default=2000)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    from generate_random_lp import generate_lp
    from clara.model.problem import LPProblem
    from clara.io.mps_parser import read_mps

    rows = []
    for n, m in SIZES:
        if m > args.max_m:
            continue
        for seed in SEEDS:
            c, A, b, ub = generate_lp(n, m, 0.3, seed)
            problem = LPProblem(c=c, A=A, b=b, upper_bounds=ub, name=f"rand_n{n}_m{m}_s{seed}")
            rec, _ = time_instance(problem, args.reps)
            if rec is None:
                print(f"  rand n={n} m={m} seed={seed}: not optimal, skipped")
                continue
            rec.update({"instance": problem.name, "family": "random", "n": n, "m": m})
            rows.append(rec)
            print(f"  {problem.name:22} m_aug={rec['m_aug']:5d} simplex={rec['simplex_time']*1e3:9.2f} ms "
                  f"solve={rec['solve_time']*1e3:9.2f} ms  chebyshev={rec['chebyshev_time']*1e3:9.2f} ms")
    for path in sorted(NETLIB_DIR.glob("*.mps")):
        problem = read_mps(path)
        rec, _ = time_instance(problem, args.reps)
        if rec is None:
            continue
        rec.update({"instance": path.stem, "family": "netlib",
                    "n": problem.num_variables, "m": problem.num_constraints})
        rows.append(rec)
        print(f"  {path.stem:22} m_aug={rec['m_aug']:5d} simplex={rec['simplex_time']*1e3:9.2f} ms "
              f"solve={rec['solve_time']*1e3:9.2f} ms  chebyshev={rec['chebyshev_time']*1e3:9.2f} ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "exp10_chebyshev_timing.csv"
    fields = ["instance", "family", "n", "m", "m_aug", "degenerate_basics", "n_degenerate_params",
              "simplex_time", "solve_time", "chebyshev_time", "chebyshev_radius",
              "chebyshev_radius_face", "simultaneity_ratio"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"\nDone: {len(rows)} instances -> {out}")

    print("\nrandom: median over seeds (ms)")
    print(f"{'n':>5} {'m':>5} {'m_aug':>6} {'simplex':>9} {'solve':>9} {'chebyshev':>10} {'cheb/simplex':>13}")
    for n, m in SIZES:
        rs = [r for r in rows if r["family"] == "random" and r["n"] == n and r["m"] == m]
        if not rs:
            continue
        s = np.median([r["simplex_time"] for r in rs]) * 1e3
        so = np.median([r["solve_time"] for r in rs]) * 1e3
        ch = np.median([r["chebyshev_time"] for r in rs]) * 1e3
        print(f"{n:5d} {m:5d} {rs[0]['m_aug']:6d} {s:9.2f} {so:9.2f} {ch:10.2f} {ch/max(s,1e-9):13.1f}")


if __name__ == "__main__":
    main()
