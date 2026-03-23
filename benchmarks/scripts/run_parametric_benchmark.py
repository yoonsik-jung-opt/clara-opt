"""Exp 4: Parametric LP path analysis for Type RC compound changes.

Usage: python benchmarks/scripts/run_parametric_benchmark.py
Output: benchmarks/results/exp4_parametric.csv, exp4_breakpoints.csv
"""

import csv
from pathlib import Path

import numpy as np

PERTURBATIONS_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"
NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"


def load_problem(filepath):
    p = Path(filepath)
    if p.suffix.lower() == ".mps":
        from clara.io.mps_parser import read_mps
        return read_mps(p)
    from clara.io.lp_parser import read_lp
    return read_lp(p)


def find_base_file(name):
    for ext in [".mps", ".lp"]:
        for d in [NETLIB_DIR, RANDOM_DIR]:
            p = d / f"{name}{ext}"
            if p.exists():
                return p
    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from clara.engine.simplex import RevisedSimplex
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.parametric import ParametricLPSolver

    results = []
    bp_results = []

    base_dirs = sorted(d for d in PERTURBATIONS_DIR.iterdir() if d.is_dir())[:20]
    print(f"Exp 4: Parametric LP on {len(base_dirs)} base instances (RC only)...")

    for base_dir in base_dirs:
        base_name = base_dir.name
        base_file = find_base_file(base_name)
        if not base_file:
            continue

        manifest = base_dir / "manifest.csv"
        if not manifest.exists():
            continue

        old_problem = load_problem(base_file)
        old_state = RevisedSimplex(old_problem).solve()
        if not old_state.is_optimal or old_state.basis_inverse is None:
            continue

        with open(manifest) as fh:
            perts = [r for r in csv.DictReader(fh) if r["type"] == "RC"]

        for pi in perts:
            pf = base_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue

            try:
                new_problem = load_problem(pf)
                change = ChangeDetector().detect(old_problem, new_problem)
                if change is None or change.delta_b is None or change.delta_c is None:
                    continue

                pr = ParametricLPSolver().solve(
                    old_state, old_problem, change.delta_b, change.delta_c
                )

                # Scratch verify
                scratch = RevisedSimplex(new_problem).solve()
                z_scratch = scratch.optimal_value if scratch.is_optimal else float("nan")
                opt_match = abs(pr.new_state.optimal_value - z_scratch) < 1e-4 if scratch.is_optimal else False

                results.append({
                    "base_instance": base_name,
                    "perturbation": pi["perturbation"],
                    "magnitude": pi.get("b_magnitude", ""),
                    "n_vars": old_problem.num_variables,
                    "n_cons": old_problem.num_constraints,
                    "num_breakpoints": pr.num_breakpoints,
                    "theta_first": f"{pr.breakpoints[0].theta:.4f}" if pr.breakpoints else "",
                    "num_pivots": pr.num_pivots,
                    "path_monotone": pr.path_monotone,
                    "optimal_match": opt_match,
                    "z_parametric": f"{pr.new_state.optimal_value:.6f}",
                    "z_scratch": f"{z_scratch:.6f}",
                })

                for k, bp in enumerate(pr.breakpoints):
                    bp_results.append({
                        "base_instance": base_name,
                        "perturbation": pi["perturbation"],
                        "bp_index": k,
                        "theta": f"{bp.theta:.6f}",
                        "bp_type": bp.breakpoint_type,
                        "leaving": bp.leaving_var or "",
                        "entering": bp.entering_var or "",
                        "objective": f"{bp.objective_value:.6f}",
                    })
            except Exception:
                pass

    if results:
        out = RESULTS_DIR / "exp4_parametric.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        matched = sum(1 for r in results if r["optimal_match"])
        print(f"Done: {len(results)} RC pairs → {out}")
        print(f"Optimal match: {matched}/{len(results)}")
    else:
        print("No RC perturbations found.")

    if bp_results:
        out2 = RESULTS_DIR / "exp4_breakpoints.csv"
        with open(out2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=bp_results[0].keys())
            w.writeheader()
            w.writerows(bp_results)
        print(f"Breakpoints: {len(bp_results)} → {out2}")


if __name__ == "__main__":
    main()
