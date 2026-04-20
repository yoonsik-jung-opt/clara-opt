"""Collect all experimental data for CLARA paper tables.

Usage: python benchmarks/scripts/collect_paper_data.py
Output: benchmarks/results/paper_data_summary.txt + per-experiment CSVs
"""

import csv
import time
from pathlib import Path

import numpy as np

RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
PERT_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_lp(filepath):
    from clara.io.lp_parser import read_lp
    return read_lp(filepath)


def load_mps(filepath):
    from clara.io.mps_parser import read_mps
    return read_mps(filepath)


def get_solvable_bases(max_n=50):
    """Get random LP instances that are solvable by Internal Simplex."""
    m = RANDOM_DIR / "manifest.csv"
    if not m.exists():
        return []
    with open(m) as f:
        return [r["instance"] for r in csv.DictReader(f)
                if r.get("match", "").strip() == "✓" and int(r["n_vars"]) <= max_n]


# ============================================================
# Exp 1: Cross-validation + Instance Statistics
# ============================================================

def run_exp1():
    from clara.engine.simplex import RevisedSimplex
    from clara.engine.highs_backend import HiGHSBackend

    print("=== Exp 1: Cross-validation ===")
    results = []

    # Random LPs
    bases = get_solvable_bases(100)
    for name in bases:
        lp = RANDOM_DIR / f"{name}.lp"
        if not lp.exists():
            continue
        try:
            p = load_lp(lp)
            si = RevisedSimplex(p).solve()
            sh = HiGHSBackend().solve(p)
            if si.is_optimal and sh.is_optimal:
                gap = abs(si.optimal_value - sh.optimal_value) / max(abs(sh.optimal_value), 1e-10)
                results.append({
                    "instance": name, "source": "random",
                    "n": p.num_variables, "m": p.num_constraints,
                    "gap": gap,
                    "condition_number": si.condition_number or 0,
                    "degenerate_count": si.degenerate_count,
                    "basis_robustness_d0": si.basis_robustness_d0 or 0,
                })
        except Exception:
            pass

    # Netlib
    for f in sorted(NETLIB_DIR.glob("*.mps")):
        if f.stem == "blend":
            continue
        try:
            p = load_mps(f)
            si = RevisedSimplex(p).solve()
            sh = HiGHSBackend().solve(p)
            if si.is_optimal and sh.is_optimal:
                gap = abs(si.optimal_value - sh.optimal_value) / max(abs(sh.optimal_value), 1e-10)
                results.append({
                    "instance": f.stem, "source": "netlib",
                    "n": p.num_variables, "m": p.num_constraints,
                    "gap": gap,
                    "condition_number": si.condition_number or 0,
                    "degenerate_count": si.degenerate_count,
                    "basis_robustness_d0": si.basis_robustness_d0 or 0,
                })
        except Exception:
            pass

    out = RESULTS_DIR / "paper_exp1.csv"
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    print(f"  {len(results)} instances")
    return results


# ============================================================
# Exp 2: Decision Quality
# ============================================================

def run_exp2():
    from clara.engine.simplex import RevisedSimplex
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.analyzer import ImpactAnalyzer

    print("=== Exp 2: Decision Quality ===")
    results = []
    bases = get_solvable_bases(50)

    for base_name in bases:
        base_lp = RANDOM_DIR / f"{base_name}.lp"
        if not base_lp.exists():
            continue
        pert_dir = PERT_DIR / base_name
        if not pert_dir.exists():
            continue
        pm = pert_dir / "manifest.csv"
        if not pm.exists():
            continue

        try:
            old_p = load_lp(base_lp)
            old_s = RevisedSimplex(old_p).solve()
            if not old_s.is_optimal:
                continue
        except Exception:
            continue

        with open(pm) as f:
            perts = list(csv.DictReader(f))

        for pi in perts:
            pf = pert_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue
            try:
                new_p = load_lp(pf)
                change = ChangeDetector().detect(old_p, new_p)
                if change is None:
                    continue
                decision = ImpactAnalyzer().analyze(old_s, change, old_p)
                new_s = RevisedSimplex(new_p).solve()
                if not new_s.is_optimal:
                    continue

                x_old = np.array([v.value for v in old_s.variables])
                z_new_at_old = float(new_p.c @ x_old)
                actual_loss = abs(new_s.optimal_value - z_new_at_old) / max(abs(new_s.optimal_value), 1e-10)
                ground_truth = actual_loss > 0.01

                if decision.should_reoptimize and ground_truth:
                    cls = "TP"
                elif not decision.should_reoptimize and not ground_truth:
                    cls = "TN"
                elif decision.should_reoptimize and not ground_truth:
                    cls = "FP"
                else:
                    cls = "FN"

                results.append({
                    "instance": base_name, "change_type": change.change_type.name,
                    "magnitude": pi.get("b_magnitude", ""),
                    "decision": "reopt" if decision.should_reoptimize else "skip",
                    "ground_truth": "different" if ground_truth else "same",
                    "classification": cls,
                })
            except Exception:
                pass

    out = RESULTS_DIR / "paper_exp2.csv"
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    print(f"  {len(results)} pairs")
    return results


# ============================================================
# Exp 3: Warm-start
# ============================================================

def run_exp3():
    from clara.engine.simplex import RevisedSimplex
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.reoptimizer import Reoptimizer
    from clara.reopt.types import ReoptDecision

    print("=== Exp 3: Warm-start ===")
    results = []
    bases = get_solvable_bases(50)

    for base_name in bases:
        base_lp = RANDOM_DIR / f"{base_name}.lp"
        if not base_lp.exists():
            continue
        pert_dir = PERT_DIR / base_name
        pm = pert_dir / "manifest.csv"
        if not pm.exists():
            continue

        try:
            old_p = load_lp(base_lp)
            old_s = RevisedSimplex(old_p).solve()
            if not old_s.is_optimal:
                continue
        except Exception:
            continue

        with open(pm) as f:
            perts = [r for r in csv.DictReader(f) if r.get("type", "") in ("R", "C")]

        for pi in perts:
            pf = pert_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue
            try:
                new_p = load_lp(pf)
                change = ChangeDetector().detect(old_p, new_p)
                if change is None:
                    continue

                forced = ReoptDecision(should_reoptimize=True, reason="bench",
                                       recommended_method="warm_start")
                t0 = time.perf_counter()
                result = Reoptimizer().reoptimize(old_s, new_p, change, forced, old_problem=old_p)
                t_ws = time.perf_counter() - t0

                t0 = time.perf_counter()
                scratch = RevisedSimplex(new_p).solve()
                t_sc = time.perf_counter() - t0

                if not scratch.is_optimal:
                    continue

                match = abs(result.new_state.optimal_value - scratch.optimal_value) < max(abs(scratch.optimal_value) * 1e-6, 1e-4)
                speedup = t_sc / max(t_ws, 1e-9)
                ws_piv = result.pivots
                sc_piv = scratch.iteration_count
                piv_red = (sc_piv - ws_piv) / max(sc_piv, 1)

                results.append({
                    "instance": base_name, "change_type": change.change_type.name,
                    "magnitude": pi.get("perturbation", ""),
                    "warm_start_pivots": ws_piv, "scratch_pivots": sc_piv,
                    "warm_start_time": t_ws, "scratch_time": t_sc,
                    "match": match, "speedup": speedup,
                    "pivot_reduction": piv_red,
                    "method_used": result.method_used,
                })
            except Exception:
                pass

    out = RESULTS_DIR / "paper_exp3.csv"
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    print(f"  {len(results)} pairs")
    return results


# ============================================================
# Exp 7: Attribution
# ============================================================

def run_exp7():
    from clara.engine.simplex import RevisedSimplex
    from clara.engine.highs_backend import HiGHSBackend
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.attribution import ChangeAttributor

    print("=== Exp 7: Attribution ===")
    results = []
    bases = get_solvable_bases(50)

    for base_name in bases:
        base_lp = RANDOM_DIR / f"{base_name}.lp"
        if not base_lp.exists():
            continue
        pert_dir = PERT_DIR / base_name
        pm = pert_dir / "manifest.csv"
        if not pm.exists():
            continue

        try:
            old_p = load_lp(base_lp)
            old_s = RevisedSimplex(old_p).solve()
            if not old_s.is_optimal:
                continue
        except Exception:
            continue

        with open(pm) as f:
            perts = [r for r in csv.DictReader(f) if r.get("type", "") == "RC"]

        for pi in perts:
            pf = pert_dir / f"{pi['instance']}.lp"
            if not pf.exists():
                continue
            try:
                new_p = load_lp(pf)
                new_s = RevisedSimplex(new_p).solve()
                if not new_s.is_optimal:
                    new_s = HiGHSBackend().solve(new_p)
                    if not new_s.is_optimal:
                        continue

                change = ChangeDetector().detect(old_p, new_p)
                if change is None:
                    continue

                attr = ChangeAttributor().attribute(old_s, new_s, old_p, new_p, change)
                adz = max(abs(attr.delta_z), 1e-10)

                results.append({
                    "instance": base_name,
                    "perturbation_type": pi.get("perturbation", ""),
                    "rhs_pct": abs(attr.rhs_effect) / adz * 100,
                    "obj_pct": abs(attr.obj_effect) / adz * 100,
                    "interaction_pct": abs(attr.interaction_effect) / adz * 100,
                    "nonlinearity_eta": attr.nonlinearity,
                    "basis_preserved": attr.basis_preserved,
                })
            except Exception:
                pass

    out = RESULTS_DIR / "paper_exp7.csv"
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    print(f"  {len(results)} pairs")
    return results


# ============================================================
# Exp 8: Simultaneous Region
# ============================================================

def run_exp8():
    from clara.engine.simplex import RevisedSimplex
    from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer

    print("=== Exp 8: Simultaneous Region ===")
    results = []
    bases = get_solvable_bases(50)
    analyzer = SimultaneousRegionAnalyzer()

    for base_name in bases:
        base_lp = RANDOM_DIR / f"{base_name}.lp"
        if not base_lp.exists():
            continue
        try:
            p = load_lp(base_lp)
            s = RevisedSimplex(p).solve()
            if not s.is_optimal or s.basis_inverse is None:
                continue

            region = analyzer.analyze(s, p)
            center_norm = np.linalg.norm(region.chebyshev_center) if region.chebyshev_center else 0

            results.append({
                "instance": base_name,
                "n": p.num_variables, "m": p.num_constraints,
                "chebyshev_radius": region.chebyshev_radius,
                "chebyshev_center_norm": center_norm,
                "min_oat_tolerance": region.min_oat_tolerance,
                "simultaneity_ratio": region.simultaneity_ratio,
                "basis_robustness_d0": s.basis_robustness_d0 or 0,
            })
        except Exception:
            pass

    out = RESULTS_DIR / "paper_exp8.csv"
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    print(f"  {len(results)} instances")
    return results


# ============================================================
# Albici golden tests
# ============================================================

def run_albici():
    from clara.engine.simplex import RevisedSimplex
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.analyzer import ImpactAnalyzer
    from clara.reopt.reoptimizer import Reoptimizer
    from clara.model.problem import LPProblem

    print("=== Albici Golden Tests ===")
    base = LPProblem(c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
                     b=[1200, 1400, 2000, 800],
                     var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])

    scenarios = {
        "base→b1": (base.c, base.A, [1300, 1200, 1800, 1000], 33100 / 9),
        "base→b2": (base.c, base.A, [1500, 1300, 2400, 800], 4300.0),
        "base→cost": ([8, 6, 7], base.A, base.b, 24800 / 3),
        "base→columns": ([3, 4, 5, 3, 5],
                          [[1, 2, 1, 2, 3], [1, 0, 3, 0, 1], [2, 1, 2, 1, 0], [0, 2, 3, 0, 1]],
                          base.b, 33700 / 9),
        "compound(RC)": ([8, 6, 7], base.A, [1500, 1300, 2400, 800], 10000.0),
    }

    base_s = RevisedSimplex(base).solve()
    results = []

    for name, (c, A, b, expected) in scenarios.items():
        vn = ["x1", "x2", "x3"] if len(c) == 3 else ["x1", "x2", "x3", "x4", "x5"]
        new_p = LPProblem(c=c, A=A, b=b, var_names=vn, constraint_names=["S1", "S2", "S3", "S4"])
        change = ChangeDetector().detect(base, new_p)
        decision = ImpactAnalyzer().analyze(base_s, change, base)
        result = Reoptimizer().reoptimize(base_s, new_p, change, decision, old_problem=base)

        match = abs(result.new_state.optimal_value - expected) < 0.01
        print(f"  {name}: method={result.method_used}, pivots={result.pivots}, "
              f"z={result.new_state.optimal_value:.4f}, match={'✓' if match else '✗'}")
        results.append({
            "scenario": name, "method": result.method_used,
            "pivots": result.pivots, "optimal": result.new_state.optimal_value,
            "expected": expected, "match": match,
        })

    return results


# ============================================================
# Summary
# ============================================================

def generate_summary(exp1, exp2, exp3, exp7, exp8, albici):
    lines = []
    lines.append("=" * 60)
    lines.append("CLARA EXPERIMENT RESULTS — PAPER VALUES")
    lines.append("=" * 60)

    # S6.1
    lines.append("\n=== S6.1 Instance Statistics (from Exp 1) ===")
    if exp1:
        kappas = [r["condition_number"] for r in exp1 if r["condition_number"] > 0]
        d0s = [r["basis_robustness_d0"] for r in exp1 if r["basis_robustness_d0"] > 0]
        degens = [r["degenerate_count"] for r in exp1]
        degen_pct = sum(1 for d in degens if d > 0) / max(len(degens), 1) * 100
        lines.append(f"  Instances: {len(exp1)}")
        lines.append(f"  Degeneracy: {sum(1 for d in degens if d>0)}/{len(degens)} ({degen_pct:.1f}%)")
        if kappas:
            lines.append(f"  κ(B): median={np.median(kappas):.2e}, max={max(kappas):.2e}")
        if d0s:
            lines.append(f"  d₀: median={np.median(d0s):.4f}, min={min(d0s):.4f}")

    # S6.2
    lines.append("\n=== S6.2 Cross-validation (from Exp 1) ===")
    if exp1:
        gaps = [r["gap"] for r in exp1]
        lines.append(f"  Total: {len(exp1)}, max_gap={max(gaps):.2e}, mean_gap={np.mean(gaps):.2e}")

    # S6.3 Attribution
    lines.append("\n=== S6.3 Attribution (from Exp 7) ===")
    if exp7:
        by_type = {}
        for r in exp7:
            t = r["perturbation_type"]
            by_type.setdefault(t, []).append(r)
        for t in ["RC_small", "RC_medium", "RC_large", "RC_asym_bc", "RC_asym_cb"]:
            if t in by_type:
                rr = by_type[t]
                lines.append(f"  {t}: n={len(rr)}, RHS={np.mean([r['rhs_pct'] for r in rr]):.1f}%, "
                             f"OBJ={np.mean([r['obj_pct'] for r in rr]):.1f}%, "
                             f"preserved={sum(1 for r in rr if r['basis_preserved'])}/{len(rr)}")
        etas = [r["nonlinearity_eta"] for r in exp7]
        lines.append(f"  η: mean={np.mean(etas):.4f}, median={np.median(etas):.4f}, max={max(etas):.4f}")

    # S6.4 Region
    lines.append("\n=== S6.4 Simultaneous Region (from Exp 8) ===")
    if exp8:
        for lo, hi, label in [(5, 10, "n=5-10"), (11, 20, "n=11-20"), (21, 50, "n=21-50")]:
            sub = [r for r in exp8 if lo <= r["n"] <= hi]
            if sub:
                rhos = [r["simultaneity_ratio"] for r in sub]
                lines.append(f"  {label}: n={len(sub)}, mean_r*={np.mean([r['chebyshev_radius'] for r in sub]):.2f}, "
                             f"mean_ρ={np.mean(rhos):.4f}, std={np.std(rhos):.4f}")
        all_rhos = [r["simultaneity_ratio"] for r in exp8]
        lines.append(f"  ALL: n={len(exp8)}, mean_ρ={np.mean(all_rhos):.4f}")
        # d0 vs rho correlation
        d0s = [r["basis_robustness_d0"] for r in exp8]
        if len(d0s) > 2:
            corr = np.corrcoef(d0s, all_rhos)[0, 1]
            lines.append(f"  d₀ vs ρ correlation: r={corr:.4f}")

    # S6.5 Decision
    lines.append("\n=== S6.5 Decision Quality (from Exp 2) ===")
    if exp2:
        for ct in ["TYPE_R", "TYPE_C", "TYPE_RC"]:
            sub = [r for r in exp2 if r["change_type"] == ct]
            if sub:
                tp = sum(1 for r in sub if r["classification"] == "TP")
                tn = sum(1 for r in sub if r["classification"] == "TN")
                fp = sum(1 for r in sub if r["classification"] == "FP")
                fn = sum(1 for r in sub if r["classification"] == "FN")
                acc = (tp + tn) / max(len(sub), 1) * 100
                lines.append(f"  {ct}: n={len(sub)}, TP={tp}, TN={tn}, FP={fp}, FN={fn}, acc={acc:.1f}%")
        total = len(exp2)
        fn_total = sum(1 for r in exp2 if r["classification"] == "FN")
        acc_total = sum(1 for r in exp2 if r["classification"] in ("TP", "TN")) / max(total, 1) * 100
        lines.append(f"  Total: n={total}, acc={acc_total:.1f}%, FN rate={fn_total/max(total,1)*100:.1f}%")

    # S6.5 Warmstart
    lines.append("\n=== S6.5 Warm-start (from Exp 3) ===")
    if exp3:
        matched = sum(1 for r in exp3 if r["match"])
        speedups = [r["speedup"] for r in exp3 if r["match"]]
        piv_reds = [r["pivot_reduction"] for r in exp3 if r["match"]]
        lines.append(f"  Total: n={len(exp3)}, match={matched}/{len(exp3)}")
        if speedups:
            lines.append(f"  Speedup: mean={np.mean(speedups):.1f}x, median={np.median(speedups):.1f}x, max={max(speedups):.1f}x")
        if piv_reds:
            lines.append(f"  Pivot reduction: mean={np.mean(piv_reds)*100:.1f}%")

    # Albici
    lines.append("\n=== Albici Golden Tests ===")
    for r in albici:
        lines.append(f"  {r['scenario']}: method={r['method']}, pivots={r['pivots']}, "
                     f"z={r['optimal']:.4f}, match={'✓' if r['match'] else '✗'}")

    summary = "\n".join(lines)
    out = RESULTS_DIR / "paper_data_summary.txt"
    out.write_text(summary)
    print(f"\nSummary saved to: {out}")
    print(summary)


def main():
    t0 = time.perf_counter()
    exp1 = run_exp1()
    exp2 = run_exp2()
    exp3 = run_exp3()
    exp7 = run_exp7()
    exp8 = run_exp8()
    albici = run_albici()
    generate_summary(exp1, exp2, exp3, exp7, exp8, albici)
    print(f"\nTotal time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
