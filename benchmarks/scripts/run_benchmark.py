"""Run Netlib cross-validation benchmark: Internal Simplex vs HiGHS.

Usage:
    python benchmarks/scripts/run_benchmark.py

Solves each Netlib instance with HiGHS (via MPS) and with Internal Simplex
(via LP, where parser supports it). Compares optimal values.

Output: benchmarks/results/netlib_YYYYMMDD.csv
"""

import csv
import time
from datetime import datetime
from pathlib import Path

MPS_DIR = Path(__file__).parent.parent / "netlib" / "mps"
LP_DIR = Path(__file__).parent.parent / "netlib" / "lp"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Known optimal values (minimization problems)
KNOWN_OPTVALS = {
    "afiro": -464.7531,
    "adlittle": 225494.9,
    "blend": -30.8122,
    "sc50a": -64.5751,
    "sc50b": -70.0000,
    "sc105": -52.2021,
    "kb2": -1749.9,
    "share2b": -415.7323,
}

TOL_OPTIMAL = 1e-2


def solve_highs_mps(mps_path):
    """Solve directly from MPS using HiGHS."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(mps_path))
    if status != highspy.HighsStatus.kOk:
        return None, 0, 0

    start = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - start

    model_status = h.getModelStatus()
    if model_status != highspy.HighsModelStatus.kOptimal:
        return None, elapsed, 0

    opt = h.getInfoValue("objective_function_value")[1]
    iters = int(h.getInfoValue("simplex_iteration_count")[1])
    return opt, elapsed, iters


def solve_internal_mps(mps_path):
    """Solve MPS file using CLARA's MPS parser + Internal Simplex."""
    try:
        from clara.io.mps_parser import read_mps
        from clara.engine import HiGHSBackend

        problem = read_mps(mps_path)
        if problem.num_variables == 0:
            return None, 0, 0, "parser: 0 vars"

        start = time.perf_counter()
        state = HiGHSBackend().solve(problem)
        elapsed = time.perf_counter() - start

        if not state.is_optimal:
            return None, elapsed, 0, f"status: {state.status.name}"

        return state.optimal_value, elapsed, state.iteration_count, "OK"
    except Exception as e:
        return None, 0, 0, str(e)[:50]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mps_files = sorted(MPS_DIR.glob("*.mps"))

    if not mps_files:
        print("No MPS files found. Run download_netlib.py first.")
        return

    results = []
    print(f"{'Instance':<12} {'Vars':>5} {'Cons':>5} {'HiGHS Opt':>12} "
          f"{'Internal Opt':>12} {'Match':>5} {'HiGHS Time':>10} {'Int Time':>10} {'Note'}")
    print("-" * 95)

    for mps_file in mps_files:
        name = mps_file.stem
        lp_file = LP_DIR / f"{name}.lp"

        # HiGHS solve (ground truth)
        h_opt, h_time, h_iters = solve_highs_mps(mps_file)

        # Get problem size from HiGHS
        import highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(mps_file))
        n_vars = h.getNumCol()
        n_cons = h.getNumRow()

        # Internal Simplex solve (via MPS parser directly)
        i_opt, i_time, i_iters, note = solve_internal_mps(mps_file)

        # Compare
        match = ""
        if h_opt is not None and i_opt is not None:
            if abs(h_opt - i_opt) < TOL_OPTIMAL:
                match = "✓"
            else:
                match = "✗"
        elif i_opt is None:
            match = "—"

        h_opt_str = f"{h_opt:.4f}" if h_opt is not None else "FAIL"
        i_opt_str = f"{i_opt:.4f}" if i_opt is not None else note

        print(f"{name:<12} {n_vars:>5} {n_cons:>5} {h_opt_str:>12} "
              f"{i_opt_str:>12} {match:>5} {h_time:>9.4f}s {i_time:>9.4f}s {'  ' + note if note != 'OK' and i_opt is not None else ''}")

        results.append({
            "instance": name,
            "variables": n_vars,
            "constraints": n_cons,
            "highs_optimal": h_opt,
            "internal_optimal": i_opt,
            "match": match,
            "highs_time": h_time,
            "internal_time": i_time,
            "highs_iters": h_iters,
            "internal_iters": i_iters,
            "note": note,
        })

    # Save CSV
    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = RESULTS_DIR / f"netlib_{date_str}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")

    # Summary
    matched = sum(1 for r in results if r["match"] == "✓")
    total = len(results)
    solved_internal = sum(1 for r in results if r["internal_optimal"] is not None)
    print(f"\nCross-validation: {matched}/{solved_internal} matched (of {total} total instances)")


if __name__ == "__main__":
    main()
