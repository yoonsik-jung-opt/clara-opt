import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 8: Simultaneous sensitivity region benchmark.

Usage: python benchmarks/scripts/run_region_benchmark.py
Output: benchmarks/results/exp8_region.csv, exp8_albici_projections.json
"""

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def process_single(task):
    import time
    from pathlib import Path

    filepath = task["filepath"]

    try:
        from clara.engine.simplex import RevisedSimplex
        from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer

        p = Path(filepath)
        if p.suffix.lower() == ".mps":
            from clara.io.mps_parser import read_mps
            problem = read_mps(p)
        else:
            from clara.io.lp_parser import read_lp
            problem = read_lp(p)

        name = p.stem

        state = RevisedSimplex(problem).solve()
        if not state.is_optimal or state.basis_inverse is None:
            return None

        analyzer = SimultaneousRegionAnalyzer()
        t0 = time.perf_counter()
        region = analyzer.analyze(state, problem)
        t_analyze = time.perf_counter() - t0

        n_basic = sum(1 for v in state.variables if v.basis_status.name == "BASIC")
        n_nonbasic = len(state.variables) - n_basic

        overest = 1 / region.simultaneity_ratio if region.simultaneity_ratio > 1e-10 else float("inf")

        return {
            "instance": name,
            "n_vars": problem.num_variables,
            "n_cons": problem.num_constraints,
            "chebyshev_radius": f"{region.chebyshev_radius:.6f}",
            "min_oat_tolerance": f"{region.min_oat_tolerance:.6f}",
            "simultaneity_ratio": f"{region.simultaneity_ratio:.6f}",
            "overestimation_factor": f"{overest:.2f}" if overest < 1e6 else "inf",
            "n_primal_cons": state.basis_inverse.shape[0],
            "n_dual_cons": n_nonbasic,
            "analyze_time_seconds": f"{t_analyze:.6f}",
        }
    except Exception:
        return None


def _albici_projections():
    """Generate 2D projections for Albici base problem."""
    from clara.engine.simplex import RevisedSimplex
    from clara.model.problem import LPProblem
    from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer

    problem = LPProblem(
        c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
        b=[1200, 1400, 2000, 800],
        var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"],
    )
    state = RevisedSimplex(problem).solve()
    if not state.is_optimal:
        return

    analyzer = SimultaneousRegionAnalyzer()
    region = analyzer.analyze(state, problem, projection_pairs=[(0, 1), (0, 2), (2, 3)])

    if region.projections:
        proj_data = {}
        for (i, j), verts in region.projections.items():
            proj_data[f"{i}_{j}"] = [(float(x), float(y)) for x, y in verts]

        out = RESULTS_DIR / "exp8_albici_projections.json"
        with open(out, "w") as fh:
            json.dump(proj_data, fh, indent=2)
        print(f"Albici projections -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []

    for f in sorted(NETLIB_DIR.glob("*.mps")):
        if f.stem != "blend":
            files.append(f)

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

    print(f"Exp 8: Sensitivity region on {len(tasks)} instances (workers={n_workers})...")

    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                results.append(r)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(tasks)}] {len(results)} successful")

    if not results:
        print("No results.")
        return

    out = RESULTS_DIR / "exp8_region.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    ratios = [float(r["simultaneity_ratio"]) for r in results]
    print(f"\nDone: {len(results)} instances -> {out}")
    print(f"Simultaneity ratio: mean={np.mean(ratios):.4f}, median={np.median(ratios):.4f}")

    # Albici 2D projections (sequential, hardcoded problem)
    _albici_projections()


if __name__ == "__main__":
    main()
