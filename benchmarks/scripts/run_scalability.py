"""Exp 6: Scalability — solve/explain/reopt time vs problem size.

Usage: python benchmarks/scripts/run_scalability.py
Output: benchmarks/results/exp6_scalability.csv
"""

import csv
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"

SIZES = [(10, 10), (20, 20), (50, 50), (100, 100), (200, 200), (500, 500)]
SEEDS = [42, 123, 456, 789, 1024]
TIMEOUT_INTERNAL = 120
TIMEOUT_HIGHS = 60


def generate_lp(n, m, seed):
    """Generate a feasible LP (same method as generate_random_lp.py)."""
    rng = np.random.RandomState(seed)
    x_feas = rng.uniform(1, 10, n)
    A = rng.randn(m, n)
    mask = rng.random((m, n)) > 0.5
    A[mask] = 0.0
    for i in range(m):
        if np.all(A[i] == 0):
            A[i, rng.randint(n)] = rng.randn()
    slack = rng.uniform(0.1, 10, m)
    b = A @ x_feas + slack
    c = rng.uniform(0.1, 5, n)
    return c, A, b


def run_size(n, m, seed):
    from clara.model.problem import LPProblem
    from clara.engine.simplex import RevisedSimplex
    from clara.engine.highs_backend import HiGHSBackend
    from clara.explain.explainer import Explainer
    from clara.explain.types import DetailLevel

    c, A, b = generate_lp(n, m, seed)
    problem = LPProblem(c=c, A=A, b=b)
    name = f"scale_n{n}_m{m}_s{seed}"

    # Internal Simplex
    t0 = time.perf_counter()
    try:
        state_i = RevisedSimplex(problem).solve()
        ti = time.perf_counter() - t0
        if ti > TIMEOUT_INTERNAL:
            i_status = "TIMEOUT"
            iters = 0
        elif state_i.is_optimal:
            i_status = "OPTIMAL"
            iters = state_i.iteration_count
        else:
            i_status = state_i.status.name
            iters = state_i.iteration_count
    except Exception:
        ti = time.perf_counter() - t0
        i_status = "ERROR"
        iters = 0
        state_i = None

    # HiGHS
    t0 = time.perf_counter()
    try:
        state_h = HiGHSBackend().solve(problem)
        th = time.perf_counter() - t0
    except Exception:
        th = 0
        state_h = None

    # Explain time
    t_explain = 0
    if state_i and state_i.is_optimal:
        t0 = time.perf_counter()
        Explainer().explain(state_i, level=DetailLevel.DETAILED, problem=problem)
        t_explain = time.perf_counter() - t0

    # Reopt pipeline time (one medium perturbation)
    t_reopt = 0
    if state_i and state_i.is_optimal and state_i.basis_inverse is not None:
        try:
            from clara.reopt.detector import ChangeDetector
            from clara.reopt.analyzer import ImpactAnalyzer
            from clara.reopt.reoptimizer import Reoptimizer

            rng = np.random.RandomState(seed + 1)
            delta_b = b * rng.uniform(-0.2, 0.2, m)
            new_b = b + delta_b
            new_problem = LPProblem(c=c, A=A, b=new_b)

            t0 = time.perf_counter()
            change = ChangeDetector().detect(problem, new_problem)
            if change:
                decision = ImpactAnalyzer().analyze(state_i, change, problem)
                Reoptimizer().reoptimize(state_i, new_problem, change, decision, old_problem=problem)
            t_reopt = time.perf_counter() - t0
        except Exception:
            pass

    memory_mb = (m * m * 8) / (1024 * 1024)  # B⁻¹ size

    return {
        "instance": name,
        "n_vars": n, "n_cons": m,
        "time_internal": f"{ti:.4f}",
        "time_highs": f"{th:.4f}",
        "iterations_internal": iters,
        "memory_binv_mb": f"{memory_mb:.3f}",
        "time_explain": f"{t_explain:.4f}",
        "time_reopt_pipeline": f"{t_reopt:.4f}",
        "internal_status": i_status,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(SIZES) * len(SEEDS)

    print(f"Exp 6: Scalability ({total} instances)...")

    for n, m in SIZES:
        for seed in SEEDS:
            print(f"  n={n}, m={m}, seed={seed}...", end=" ")
            r = run_size(n, m, seed)
            results.append(r)
            print(f"{r['internal_status']} ({r['time_internal']}s)")

    out = RESULTS_DIR / "exp6_scalability.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\nDone: {len(results)} → {out}")
    for n, m in SIZES:
        subset = [r for r in results if r["n_vars"] == n]
        times = [float(r["time_internal"]) for r in subset]
        print(f"  n={n}, m={m}: mean={np.mean(times):.3f}s")


if __name__ == "__main__":
    main()
