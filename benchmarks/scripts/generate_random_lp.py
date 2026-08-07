"""Generate random feasible+bounded LP instances for benchmarking.

Usage:
    python benchmarks/scripts/generate_random_lp.py

Output:
    benchmarks/instances/random/*.lp  (120 instances)
    benchmarks/instances/random/manifest.csv
"""

import csv
import time
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent.parent / "instances" / "random"

SIZES = [(10, 10), (10, 20), (20, 20), (20, 40),
         (50, 50), (50, 100), (100, 100), (100, 200)]
DENSITIES = [0.3, 0.5, 0.8]
SEEDS = [42, 123, 456, 789, 1024]


def generate_lp(n_vars, n_cons, density, seed):
    """Generate a feasible, bounded LP.

    Method:
        1. Generate feasible interior point x_feas ~ U(1, 10)
        2. Generate sparse A with given density, values ~ N(0, 1)
        3. Generate positive slack ~ U(0.1, 10)
        4. b = A @ x_feas + slack (guarantees feasibility)
        5. c ~ U(0.1, 5) (positive, bounded by constraints)
    """
    rng = np.random.RandomState(seed)

    x_feas = rng.uniform(1, 10, n_vars)
    A = rng.randn(n_cons, n_vars)
    mask = rng.random((n_cons, n_vars)) > density
    A[mask] = 0.0
    # Ensure each row has at least one nonzero
    for i in range(n_cons):
        if np.all(A[i] == 0):
            j = rng.randint(n_vars)
            A[i, j] = rng.randn()

    slack = rng.uniform(0.1, 10, n_cons)
    b = A @ x_feas + slack
    c = rng.uniform(0.1, 5, n_vars)
    upper_bounds = x_feas * 3  # prevents unbounded

    return c, A, b, upper_bounds


def write_lp_file(filepath, name, c, A, b, upper_bounds=None):
    """Write LP problem to .lp file format."""
    n, m = len(c), len(b)
    var_names = [f"x{j+1}" for j in range(n)]

    with open(filepath, "w") as f:
        f.write(f"\\ Random LP: {name}\n")
        f.write("Maximize\n obj:")
        for j in range(n):
            sign = " +" if j > 0 and c[j] >= 0 else " "
            f.write(f"{sign}{c[j]:.6f} {var_names[j]}")
        f.write("\n\nSubject To\n")
        for i in range(m):
            f.write(f" c{i+1}:")
            first = True
            for j in range(n):
                if abs(A[i, j]) > 1e-12:
                    sign = " +" if not first and A[i, j] >= 0 else " "
                    f.write(f"{sign}{A[i, j]:.6f} {var_names[j]}")
                    first = False
            f.write(f" <= {b[i]:.6f}\n")
        if upper_bounds is not None:
            f.write("\nBounds\n")
            for j in range(n):
                f.write(f" 0 <= {var_names[j]} <= {upper_bounds[j]:.6f}\n")
        f.write("\nEnd\n")


def solve_highs(c, A, b, upper_bounds=None):
    """Solve with HiGHS, return (optimal_value, time, status_name)."""
    import highspy
    n, m = len(c), len(b)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    for j in range(n):
        ub = float(upper_bounds[j]) if upper_bounds is not None else highspy.kHighsInf
        h.addVar(0.0, ub)
    for j in range(n):
        h.changeColCost(j, float(c[j]))
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    for i in range(m):
        idx = [int(j) for j in range(n) if abs(A[i, j]) > 1e-12]
        val = [float(A[i, j]) for j in idx]
        h.addRow(-highspy.kHighsInf, float(b[i]), len(idx), idx, val)
    start = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - start
    status = h.getModelStatus().name
    if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
        return h.getInfoValue("objective_function_value")[1], elapsed, status
    return None, elapsed, status


def solve_internal(c, A, b, upper_bounds=None):
    """Solve with Internal Simplex, return (optimal_value, time, iters, status)."""
    try:
        from clara.model.problem import LPProblem
        from clara.engine import HiGHSBackend
        ub = np.array(upper_bounds) if upper_bounds is not None else None
        p = LPProblem(c=c, A=A, b=b, upper_bounds=ub)
        start = time.perf_counter()
        state = HiGHSBackend().solve(p)
        elapsed = time.perf_counter() - start
        status = state.status.name
        if elapsed > 60:
            return None, elapsed, 0, "TIMEOUT"
        if state.is_optimal:
            return state.optimal_value, elapsed, state.iteration_count, status
        return None, elapsed, state.iteration_count, status
    except Exception as e:
        return None, 0, 0, f"ERROR:{str(e)[:30]}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(SIZES) * len(DENSITIES) * len(SEEDS)
    count = 0

    print(f"Generating {total} random LP instances...")

    for n, m in SIZES:
        for d in DENSITIES:
            for seed in SEEDS:
                count += 1
                d_str = str(int(d * 10))
                name = f"rand_n{n}_m{m}_d{d_str}_s{seed}"
                filepath = OUTPUT_DIR / f"{name}.lp"

                c, A, b, ub = generate_lp(n, m, d, seed)
                write_lp_file(filepath, name, c, A, b, ub)

                # HiGHS always runs
                h_opt, h_time, h_status = solve_highs(c, A, b, ub)

                # CLARA-backend solve (validates the wrapper against raw
                # highspy; no size guard needed with the HiGHS backend)
                i_opt, i_time, i_iters, i_status = solve_internal(c, A, b, ub)

                match = ""
                if h_opt is not None and i_opt is not None:
                    match = "✓" if abs(h_opt - i_opt) < 0.01 else "✗"
                elif i_status == "skipped":
                    match = "—"
                else:
                    match = i_status

                results.append({
                    "instance": name,
                    "n_vars": n, "n_cons": m,
                    "density": d, "seed": seed,
                    "optimal_highs": f"{h_opt:.6f}" if h_opt is not None else "",
                    "highs_status": h_status,
                    "optimal_internal": f"{i_opt:.6f}" if i_opt is not None else "",
                    "internal_status": i_status,
                    "match": match,
                    "highs_time": f"{h_time:.4f}",
                    "internal_time": f"{i_time:.4f}",
                    "internal_iters": i_iters,
                })

                if count % 20 == 0:
                    matched = sum(1 for r in results if r["match"] == "✓")
                    print(f"  [{count}/{total}] {matched} matched so far...")

    # Write manifest
    manifest = OUTPUT_DIR / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    matched = sum(1 for r in results if r["match"] == "✓")
    total_solved = sum(1 for r in results if r["optimal_internal"] != "timeout")
    print(f"\nGenerated: {len(results)} instances → {OUTPUT_DIR}")
    print(f"Cross-validated: {matched}/{total_solved} (Internal solved)")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
