"""Exp 1: Cross-validation — Internal Simplex vs HiGHS.

Usage: python benchmarks/scripts/run_cross_validation.py
Output: benchmarks/results/exp1_cross_validation.csv
"""

import csv
import math
import time
from pathlib import Path

import numpy as np

NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_problem(filepath):
    p = Path(filepath)
    if p.suffix.lower() == ".mps":
        from clara.io.mps_parser import read_mps
        return read_mps(p)
    from clara.io.lp_parser import read_lp
    return read_lp(p)


def run_instance(filepath):
    from clara.engine.simplex import RevisedSimplex
    from clara.engine.highs_backend import HiGHSBackend

    problem = load_problem(filepath)
    n, m = problem.num_variables, problem.num_constraints

    # Internal
    t0 = time.perf_counter()
    try:
        si = RevisedSimplex(problem).solve()
        ti = time.perf_counter() - t0
        if not si.is_optimal or ti > 60:
            return None
    except Exception:
        return None

    # HiGHS
    t0 = time.perf_counter()
    sh = HiGHSBackend().solve(problem)
    th = time.perf_counter() - t0
    if not sh.is_optimal:
        return None

    zi, zh = si.optimal_value, sh.optimal_value
    opt_gap = abs(zi - zh) / max(abs(zh), 1e-10)

    # Solution diff
    vi = {v.name: v.value for v in si.variables}
    vh = {v.name: v.value for v in sh.variables}
    sol_diff = max(abs(vi.get(k, 0) - vh.get(k, 0)) for k in vi)

    # Sensitivity match
    obj_match = _range_match(si.sensitivity.obj_coeff_ranges, sh.sensitivity.obj_coeff_ranges)
    rhs_match = _range_match(si.sensitivity.rhs_ranges, sh.sensitivity.rhs_ranges)

    return {
        "instance": Path(filepath).stem,
        "n_vars": n, "n_cons": problem.num_constraints,
        "z_internal": f"{zi:.6f}", "z_highs": f"{zh:.6f}",
        "opt_gap": f"{opt_gap:.2e}",
        "sol_diff": f"{sol_diff:.2e}",
        "obj_range_match": f"{obj_match:.1%}",
        "rhs_range_match": f"{rhs_match:.1%}",
        "time_internal": f"{ti:.4f}", "time_highs": f"{th:.4f}",
        "iterations": si.iteration_count,
    }


def _range_match(ranges_a, ranges_b):
    if not ranges_a or not ranges_b:
        return 0.0
    matches = 0
    total = 0
    for k in ranges_a:
        if k not in ranges_b:
            continue
        total += 1
        lo_a, hi_a = ranges_a[k]
        lo_b, hi_b = ranges_b[k]
        if _close(lo_a, lo_b, 0.01) and _close(hi_a, hi_b, 0.01):
            matches += 1
    return matches / max(total, 1)


def _close(a, b, tol):
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) / max(abs(b), 1) < tol


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []

    # Netlib (skip blend)
    for f in sorted(NETLIB_DIR.glob("*.mps")):
        if f.stem != "blend":
            files.append(f)

    # Random (solvable only)
    manifest = RANDOM_DIR / "manifest.csv"
    if manifest.exists():
        with open(manifest) as fh:
            for row in csv.DictReader(fh):
                if row["match"] == "✓":
                    lp = RANDOM_DIR / f"{row['instance']}.lp"
                    if lp.exists():
                        files.append(lp)

    print(f"Exp 1: Cross-validation on {len(files)} instances...")
    results = []
    for i, f in enumerate(files):
        r = run_instance(f)
        if r:
            results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}] {len(results)} successful")

    out = RESULTS_DIR / "exp1_cross_validation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"Done: {len(results)} instances → {out}")
    gaps = [float(r["opt_gap"]) for r in results]
    print(f"Opt gap: max={max(gaps):.2e}, mean={np.mean(gaps):.2e}")


if __name__ == "__main__":
    main()
