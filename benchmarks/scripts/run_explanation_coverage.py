"""Exp 5: Explanation coverage — verify all explanation fields are populated.

Usage: python benchmarks/scripts/run_explanation_coverage.py
Output: benchmarks/results/exp5_explanation.csv
"""

import csv
import time
from pathlib import Path

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
    from clara.explain.explainer import Explainer
    from clara.explain.types import DetailLevel

    problem = load_problem(filepath)
    state = RevisedSimplex(problem).solve()
    if not state.is_optimal:
        return None

    t0 = time.perf_counter()
    report = Explainer().explain(state, level=DetailLevel.DETAILED, problem=problem)
    t_explain = time.perf_counter() - t0

    n = problem.num_variables
    m = problem.num_constraints

    # Constraint classification
    n_classified = sum(1 for c in state.constraints if c.is_binding is not None)
    n_duals = sum(1 for c in state.constraints if c.dual_value is not None)
    n_vars_status = sum(1 for v in state.variables if v.basis_status is not None)
    n_rc = sum(1 for v in state.variables if v.reduced_cost is not None)
    n_obj_ranges = len(state.sensitivity.obj_coeff_ranges)
    n_rhs_ranges = len(state.sensitivity.rhs_ranges)

    # Bottleneck
    binding = [c for c in state.constraints if c.is_binding]
    if binding:
        top = max(binding, key=lambda c: abs(c.dual_value))
        bottleneck_ok = report.binding.binding[0].name == top.name if report.binding.binding else False
    else:
        bottleneck_ok = True  # no binding → no bottleneck expected

    orig_m = min(len(state.constraints), m)

    return {
        "instance": Path(filepath).stem,
        "n_vars": n, "n_cons": m,
        "pct_constraints_classified": f"{n_classified / max(orig_m, 1):.1%}",
        "pct_duals_present": f"{n_duals / max(orig_m, 1):.1%}",
        "pct_variables_classified": f"{n_vars_status / max(n, 1):.1%}",
        "pct_reduced_costs_present": f"{n_rc / max(n, 1):.1%}",
        "pct_obj_ranges": f"{n_obj_ranges / max(n, 1):.1%}",
        "pct_rhs_ranges": f"{n_rhs_ranges / max(orig_m, 1):.1%}",
        "bottleneck_identified": bottleneck_ok,
        "explain_time_seconds": f"{t_explain:.6f}",
    }


def main():
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

    print(f"Exp 5: Explanation coverage on {len(files)} instances...")
    results = []
    for i, f in enumerate(files):
        try:
            r = run_instance(f)
            if r:
                results.append(r)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}]")

    if not results:
        print("No results.")
        return

    out = RESULTS_DIR / "exp5_explanation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    bottleneck_ok = sum(1 for r in results if r["bottleneck_identified"])
    print(f"Done: {len(results)} instances → {out}")
    print(f"Bottleneck correct: {bottleneck_ok}/{len(results)}")


if __name__ == "__main__":
    main()
