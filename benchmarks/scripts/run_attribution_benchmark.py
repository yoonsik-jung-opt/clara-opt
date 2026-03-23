"""Exp 7: Attribution decomposition benchmark for Type RC changes.

Usage: python benchmarks/scripts/run_attribution_benchmark.py
Output: benchmarks/results/exp7_attribution.csv
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
    from clara.reopt.attribution import ChangeAttributor
    from clara.reopt.detector import ChangeDetector

    results = []
    base_dirs = sorted(d for d in PERTURBATIONS_DIR.iterdir() if d.is_dir())[:20]

    print(f"Exp 7: Attribution on {len(base_dirs)} base instances (RC only)...")

    for base_dir in base_dirs:
        base_name = base_dir.name
        base_file = find_base_file(base_name)
        if not base_file:
            continue

        manifest = base_dir / "manifest.csv"
        if not manifest.exists():
            continue

        try:
            old_problem = load_problem(base_file)
            old_state = RevisedSimplex(old_problem).solve()
            if not old_state.is_optimal:
                continue
        except Exception:
            continue

        with open(manifest) as fh:
            perts = [r for r in csv.DictReader(fh) if r["type"] == "RC"]

        for pi in perts:
            pf = base_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue
            try:
                new_problem = load_problem(pf)
                new_state = RevisedSimplex(new_problem).solve()
                if not new_state.is_optimal:
                    continue

                change = ChangeDetector().detect(old_problem, new_problem)
                if change is None:
                    continue

                attr = ChangeAttributor().attribute(
                    old_state, new_state, old_problem, new_problem, change
                )
                dz = attr.delta_z
                adz = max(abs(dz), 1e-10)

                top_rhs = max(attr.rhs_contributions.items(), key=lambda x: abs(x[1]), default=("", 0))
                top_obj = max(attr.obj_contributions.items(), key=lambda x: abs(x[1]), default=("", 0))

                results.append({
                    "base_instance": base_name,
                    "perturbation": pi["perturbation"],
                    "magnitude": pi.get("b_magnitude", ""),
                    "n_vars": old_problem.num_variables,
                    "n_cons": old_problem.num_constraints,
                    "delta_z": f"{dz:.6f}",
                    "rhs_effect": f"{attr.rhs_effect:.6f}",
                    "obj_effect": f"{attr.obj_effect:.6f}",
                    "interaction_effect": f"{attr.interaction_effect:.6f}",
                    "rhs_pct": f"{abs(attr.rhs_effect)/adz*100:.1f}",
                    "obj_pct": f"{abs(attr.obj_effect)/adz*100:.1f}",
                    "interaction_pct": f"{abs(attr.interaction_effect)/adz*100:.1f}",
                    "shapley_b": f"{attr.shapley_b:.6f}" if attr.shapley_b is not None else "",
                    "shapley_c": f"{attr.shapley_c:.6f}" if attr.shapley_c is not None else "",
                    "basis_preserved": attr.basis_preserved,
                    "nonlinearity": f"{attr.nonlinearity:.6f}",
                    "top_rhs_param": top_rhs[0],
                    "top_rhs_value": f"{top_rhs[1]:.6f}",
                    "top_obj_param": top_obj[0],
                    "top_obj_value": f"{top_obj[1]:.6f}",
                })
            except Exception:
                pass

        if results:
            print(f"  [{base_name}] {len(results)} total")

    if not results:
        print("No results.")
        return

    out = RESULTS_DIR / "exp7_attribution.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    preserved = sum(1 for r in results if r["basis_preserved"])
    print(f"\nDone: {len(results)} RC pairs → {out}")
    print(f"Basis preserved: {preserved}/{len(results)}")
    rhs_pcts = [float(r["rhs_pct"]) for r in results]
    obj_pcts = [float(r["obj_pct"]) for r in results]
    print(f"Mean RHS contribution: {np.mean(rhs_pcts):.1f}%")
    print(f"Mean OBJ contribution: {np.mean(obj_pcts):.1f}%")


if __name__ == "__main__":
    main()
