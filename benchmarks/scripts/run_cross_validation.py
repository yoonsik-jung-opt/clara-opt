import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 1: Cross-validation — Internal Simplex vs HiGHS.

Usage: python benchmarks/scripts/run_cross_validation.py
Output: benchmarks/results/exp1_cross_validation.csv
"""

import argparse
import csv
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
RESULTS_DIR = Path(__file__).parent.parent / "results"


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


def process_single(task):
    import math
    import time
    from pathlib import Path

    filepath = task["filepath"]

    try:
        from clara.engine.simplex import RevisedSimplex
        from clara.engine.highs_backend import HiGHSBackend

        p = Path(filepath)
        if p.suffix.lower() == ".mps":
            from clara.io.mps_parser import read_mps
            problem = read_mps(p)
        else:
            from clara.io.lp_parser import read_lp
            problem = read_lp(p)

        n = problem.num_variables

        # HiGHS always runs
        t0 = time.perf_counter()
        try:
            sh = HiGHSBackend().solve(problem)
            th = time.perf_counter() - t0
            h_status = sh.status.name
            zh = sh.optimal_value if sh.is_optimal else None
        except Exception:
            th = 0
            h_status = "ERROR"
            zh = None

        # Internal
        t0 = time.perf_counter()
        try:
            si = RevisedSimplex(problem).solve()
            ti = time.perf_counter() - t0
            i_status = si.status.name
            if ti > 60:
                i_status = "TIMEOUT"
            zi = si.optimal_value if si.is_optimal else None
        except Exception:
            ti = 0
            i_status = "ERROR"
            zi = None
            si = None

        opt_gap = abs(zi - zh) / max(abs(zh), 1e-10) if zi is not None and zh is not None else None

        # Solution diff & sensitivity match (only if both optimal)
        sol_diff = None
        obj_match = rhs_match = None
        if si and si.is_optimal and sh and sh.is_optimal:
            vi = {v.name: v.value for v in si.variables}
            vh = {v.name: v.value for v in sh.variables}
            sol_diff = max(abs(vi.get(k, 0) - vh.get(k, 0)) for k in vi) if vi else 0

            def _range_match_inner(ranges_a, ranges_b):
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
                    def _close_inner(a, b, tol):
                        if math.isinf(a) and math.isinf(b):
                            return (a > 0) == (b > 0)
                        if math.isinf(a) or math.isinf(b):
                            return False
                        return abs(a - b) / max(abs(b), 1) < tol
                    if _close_inner(lo_a, lo_b, 0.01) and _close_inner(hi_a, hi_b, 0.01):
                        matches += 1
                return matches / max(total, 1)

            obj_match = _range_match_inner(si.sensitivity.obj_coeff_ranges, sh.sensitivity.obj_coeff_ranges)
            rhs_match = _range_match_inner(si.sensitivity.rhs_ranges, sh.sensitivity.rhs_ranges)

        match = ""
        if zi is not None and zh is not None:
            match = "✓" if abs(zi - zh) < 0.01 else "✗"
        elif zi is None:
            match = i_status

        return {
            "instance": Path(filepath).stem,
            "n_vars": n, "n_cons": problem.num_constraints,
            "z_internal": f"{zi:.6f}" if zi is not None else "",
            "z_highs": f"{zh:.6f}" if zh is not None else "",
            "internal_status": i_status,
            "highs_status": h_status,
            "match": match,
            "opt_gap": f"{opt_gap:.2e}" if opt_gap is not None else "",
            "sol_diff": f"{sol_diff:.2e}" if sol_diff is not None else "",
            "obj_range_match": f"{obj_match:.1%}" if obj_match is not None else "",
            "rhs_range_match": f"{rhs_match:.1%}" if rhs_match is not None else "",
            "time_internal": f"{ti:.4f}", "time_highs": f"{th:.4f}",
            "iterations": si.iteration_count if si else 0,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

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

    tasks = [{"filepath": str(f)} for f in files]
    n_workers = args.workers

    print(f"Exp 1: Cross-validation on {len(tasks)} instances (workers={n_workers})...")
    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                results.append(r)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(tasks)}] {len(results)} successful")

    if not results:
        print("No results generated.")
        return

    out = RESULTS_DIR / "exp1_cross_validation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"Done: {len(results)} instances -> {out}")
    gaps = [float(r["opt_gap"]) for r in results if r["opt_gap"]]
    if gaps:
        print(f"Opt gap: max={max(gaps):.2e}, mean={np.mean(gaps):.2e}")


if __name__ == "__main__":
    main()
