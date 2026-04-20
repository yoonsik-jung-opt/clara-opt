"""HiGHS warm-start comparison: CLARA vs HiGHS native advanced basis.

Compares wall-clock time and iteration counts for:
  1. HiGHS cold start
  2. HiGHS warm start (setBasis from CLARA's old solution)
  3. CLARA warm start (from_warm_start with stored basis_indices)
  4. CLARA cold start

Output: benchmarks/results/exp_highs_comparison.csv
"""

import csv
import time
from pathlib import Path

import highspy
import numpy as np

from clara.engine.highs_backend import HiGHSBackend
from clara.engine.simplex import RevisedSimplex
from clara.io.lp_parser import read_lp
from clara.model.problem import LPProblem
from clara.model.solve_state import BasisStatus
from clara.reopt.detector import ChangeDetector
from clara.reopt.reoptimizer import Reoptimizer
from clara.reopt.types import ReoptDecision

RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
PERT_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_to_highs(h, problem: LPProblem):
    """Load LPProblem into a HiGHS instance."""
    n = problem.num_variables
    m = problem.num_constraints
    for j in range(n):
        lb = float(problem.lower_bounds[j]) if problem.lower_bounds is not None else 0.0
        ub = float(problem.upper_bounds[j]) if problem.upper_bounds is not None else highspy.kHighsInf
        h.addVar(lb, ub)
    for j in range(n):
        h.changeColCost(j, float(problem.c[j]))
    if problem.sense == "minimize":
        h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    else:
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    for i in range(m):
        idx = [int(j) for j in range(n) if abs(problem.A[i, j]) > 1e-12]
        val = [float(problem.A[i, j]) for j in idx]
        h.addRow(-highspy.kHighsInf, float(problem.b[i]), len(idx), idx, val)


def extract_highs_basis(clara_state, problem):
    """Convert CLARA SolveState basis to HiGHS basis status arrays."""
    n = problem.num_variables
    m = problem.num_constraints

    col_status = [highspy.HighsBasisStatus.kLower] * n
    row_status = [highspy.HighsBasisStatus.kBasic] * m

    for j, v in enumerate(clara_state.variables[:n]):
        if v.basis_status == BasisStatus.BASIC:
            col_status[j] = highspy.HighsBasisStatus.kBasic
        elif v.basis_status == BasisStatus.NONBASIC_UPPER:
            col_status[j] = highspy.HighsBasisStatus.kUpper
        else:
            col_status[j] = highspy.HighsBasisStatus.kLower

    for i, c in enumerate(clara_state.constraints[:m]):
        if c.basis_status == BasisStatus.BASIC:
            row_status[i] = highspy.HighsBasisStatus.kBasic
        elif c.basis_status == BasisStatus.NONBASIC_UPPER:
            row_status[i] = highspy.HighsBasisStatus.kUpper
        else:
            row_status[i] = highspy.HighsBasisStatus.kLower

    return col_status, row_status


def run_comparison(base_problem, pert_problem, base_state):
    """Run 4-way comparison on one (base, perturbation) pair."""
    n = pert_problem.num_variables

    # 1. HiGHS cold start
    h1 = highspy.Highs()
    h1.setOptionValue("output_flag", False)
    h1.setOptionValue("solver", "simplex")
    load_to_highs(h1, pert_problem)
    t0 = time.perf_counter()
    h1.run()
    highs_cold_t = time.perf_counter() - t0
    highs_cold_z = h1.getInfoValue("objective_function_value")[1]
    highs_cold_it = int(h1.getInfoValue("simplex_iteration_count")[1])

    # 2. HiGHS warm start
    h2 = highspy.Highs()
    h2.setOptionValue("output_flag", False)
    h2.setOptionValue("solver", "simplex")
    load_to_highs(h2, pert_problem)
    col_st, row_st = extract_highs_basis(base_state, pert_problem)
    try:
        h2.setBasis(col_st, row_st)
    except Exception:
        pass  # setBasis may fail for dimension mismatch
    t0 = time.perf_counter()
    h2.run()
    highs_warm_t = time.perf_counter() - t0
    highs_warm_z = h2.getInfoValue("objective_function_value")[1]
    highs_warm_it = int(h2.getInfoValue("simplex_iteration_count")[1])

    # 3. CLARA warm start
    change = ChangeDetector().detect(base_problem, pert_problem)
    if change is None:
        return None
    decision = ReoptDecision(should_reoptimize=True, reason="bench", recommended_method="warm_start")
    t0 = time.perf_counter()
    result = Reoptimizer().reoptimize(base_state, pert_problem, change, decision, old_problem=base_problem)
    clara_warm_t = time.perf_counter() - t0
    clara_warm_z = result.new_state.optimal_value
    clara_warm_piv = result.pivots

    # 4. CLARA cold start
    t0 = time.perf_counter()
    clara_cold = RevisedSimplex(pert_problem).solve()
    clara_cold_t = time.perf_counter() - t0
    clara_cold_z = clara_cold.optimal_value
    clara_cold_piv = clara_cold.iteration_count

    return {
        "highs_cold_time": highs_cold_t, "highs_cold_iters": highs_cold_it,
        "highs_warm_time": highs_warm_t, "highs_warm_iters": highs_warm_it,
        "clara_cold_time": clara_cold_t, "clara_cold_pivots": clara_cold_piv,
        "clara_warm_time": clara_warm_t, "clara_warm_pivots": clara_warm_piv,
        "highs_cold_z": highs_cold_z, "clara_cold_z": clara_cold_z,
        "clara_warm_method": result.method_used,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get solvable instances where CLARA warm-start works (n≤20)
    manifest = RANDOM_DIR / "manifest.csv"
    if not manifest.exists():
        print("No manifest. Run generate_random_lp.py first.")
        return

    with open(manifest) as f:
        bases = [r["instance"] for r in csv.DictReader(f)
                 if r.get("match", "").strip() == "✓" and int(r["n_vars"]) <= 20]

    results = []
    for base_name in bases:
        base_lp = RANDOM_DIR / f"{base_name}.lp"
        if not base_lp.exists():
            continue

        base_problem = read_lp(base_lp)
        base_state = RevisedSimplex(base_problem).solve()
        if not base_state.is_optimal:
            continue

        pert_dir = PERT_DIR / base_name
        pm = pert_dir / "manifest.csv"
        if not pm.exists():
            continue

        with open(pm) as f:
            perts = [r for r in csv.DictReader(f) if r.get("type", "") in ("R", "C")][:3]

        for pi in perts:
            pf = pert_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue
            try:
                pert_problem = read_lp(pf)
                r = run_comparison(base_problem, pert_problem, base_state)
                if r:
                    r["instance"] = base_name
                    r["change_type"] = pi.get("type", "")
                    r["n"] = base_problem.num_variables
                    results.append(r)
            except Exception:
                pass

    if not results:
        print("No results.")
        return

    out = RESULTS_DIR / "exp_highs_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"HiGHS Warm-start Comparison ({len(results)} pairs)")
    print(f"{'='*60}")
    print(f"{'':>20} | {'HiGHS cold':>12} | {'HiGHS warm':>12} | {'CLARA cold':>12} | {'CLARA warm':>12}")
    print("-" * 75)

    hc_t = np.mean([r["highs_cold_time"] for r in results]) * 1000
    hw_t = np.mean([r["highs_warm_time"] for r in results]) * 1000
    cc_t = np.mean([r["clara_cold_time"] for r in results]) * 1000
    cw_t = np.mean([r["clara_warm_time"] for r in results]) * 1000
    print(f"{'Time (ms)':>20} | {hc_t:>12.2f} | {hw_t:>12.2f} | {cc_t:>12.2f} | {cw_t:>12.2f}")

    hc_i = np.mean([r["highs_cold_iters"] for r in results])
    hw_i = np.mean([r["highs_warm_iters"] for r in results])
    cc_p = np.mean([r["clara_cold_pivots"] for r in results])
    cw_p = np.mean([r["clara_warm_pivots"] for r in results])
    print(f"{'Iters/pivots':>20} | {hc_i:>12.1f} | {hw_i:>12.1f} | {cc_p:>12.1f} | {cw_p:>12.1f}")

    print(f"{'Speedup vs cold':>20} | {'1.0x':>12} | {hc_t/max(hw_t,0.01):>11.1f}x | {'1.0x':>12} | {cc_t/max(cw_t,0.01):>11.1f}x")

    # CLARA warm-start only cases
    ws_only = [r for r in results if r["clara_warm_method"] in ("warm_start", "warm_start_dual")]
    if ws_only:
        print(f"\nCLARA actual warm-start cases: {len(ws_only)}/{len(results)}")
        cw_ws = np.mean([r["clara_warm_pivots"] for r in ws_only])
        cc_ws = np.mean([r["clara_cold_pivots"] for r in ws_only])
        print(f"  CLARA warm pivots: {cw_ws:.1f}, cold pivots: {cc_ws:.1f}, ratio: {cc_ws/max(cw_ws,0.01):.1f}x")

    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
