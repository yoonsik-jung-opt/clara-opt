"""Run all 4 final code tasks for paper revision.

Task 1: Joint (Δb, Δc) Chebyshev region
Task 2: Wendell tolerance comparison
Task 3: Runtime breakdown
Task 4: ρ vs κ(B) correlation
"""

import csv
import time
from pathlib import Path

import numpy as np

RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
PERT_DIR = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_lp(filepath):
    from clara.io.lp_parser import read_lp
    return read_lp(filepath)


def get_bases(max_n=20):
    m = RANDOM_DIR / "manifest.csv"
    if not m.exists():
        return []
    with open(m) as f:
        return [r["instance"] for r in csv.DictReader(f)
                if r.get("match", "").strip() == "✓" and int(r["n_vars"]) <= max_n]


# ============================================================
# Task 1: Joint region
# ============================================================

def run_task1():
    from clara.engine import HiGHSBackend
    from clara.model.problem import LPProblem
    from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer

    print("=== Task 1: Joint (Δb, Δc) Chebyshev Region ===")
    analyzer = SimultaneousRegionAnalyzer()

    # Albici
    albici = LPProblem(c=[3, 4, 5], A=[[1, 2, 1], [1, 0, 3], [2, 1, 2], [0, 2, 3]],
                       b=[1200, 1400, 2000, 800],
                       var_names=["x1", "x2", "x3"], constraint_names=["S1", "S2", "S3", "S4"])
    s = HiGHSBackend().solve(albici)
    r = analyzer.analyze_joint(s, albici)
    print(f"  Albici: r*_b={r['chebyshev_radius_b_only']:.4f}, r*+={r['chebyshev_radius_joint']:.4f}, "
          f"ratio={r['ratio']:.4f}")

    # Case study
    from benchmarks.scripts.supply_chain_case_study import create_base_problem
    sc, _, _ = create_base_problem()
    s_sc = HiGHSBackend().solve(sc)
    if s_sc.is_optimal:
        r_sc = analyzer.analyze_joint(s_sc, sc)
        print(f"  Case study: r*_b={r_sc['chebyshev_radius_b_only']:.4f}, r*+={r_sc['chebyshev_radius_joint']:.4f}, "
              f"ratio={r_sc['ratio']:.4f}")

    # Random instances
    bases = get_bases(20)
    ratios = []
    for name in bases[:30]:
        lp = RANDOM_DIR / f"{name}.lp"
        if not lp.exists():
            continue
        try:
            p = load_lp(lp)
            st = HiGHSBackend().solve(p)
            if st.is_optimal and st.basis_indices:
                rr = analyzer.analyze_joint(st, p)
                if rr["chebyshev_radius_b_only"] > 1e-10:
                    ratios.append(rr["ratio"])
        except Exception:
            pass

    if ratios:
        print(f"  Random (n≤20, {len(ratios)} inst): r*+/r*_b mean={np.mean(ratios):.4f}, "
              f"median={np.median(ratios):.4f}")
    print()


# ============================================================
# Task 2: Wendell tolerance
# ============================================================

def wendell_tolerance_rhs(B_inv, x_B, b_orig):
    """Wendell (1985) RHS tolerance using ORIGINAL problem coordinates.

    t = min_i x_B_i / (Σ_j |B⁻¹_{ij}| · |b_j|)

    Uses only the first m_orig columns of B_inv (original constraints),
    since upper-bound slack rows have constant RHS.
    Clamps x_B to non-negative to handle numerical degeneracy.
    """
    m_orig = len(b_orig)
    m_aug = len(x_B)
    # Use only original-problem columns of B_inv
    B_inv_orig = B_inv[:, :m_orig]
    x_B_safe = np.maximum(x_B, 0.0)  # clamp numerical negatives

    t_values = np.full(m_aug, np.inf)
    for i in range(m_aug):
        denom = np.sum(np.abs(B_inv_orig[i]) * np.abs(b_orig))
        if denom > 1e-12:
            t_values[i] = x_B_safe[i] / denom
    return float(np.min(t_values)), int(np.argmin(t_values))


def run_task2():
    from clara.engine import HiGHSBackend
    from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer

    print("=== Task 2: Wendell Tolerance Comparison ===")
    analyzer = SimultaneousRegionAnalyzer()
    bases = get_bases(50)

    wendell_ts, cheby_rs, oat_mins = [], [], []
    for name in bases:
        lp = RANDOM_DIR / f"{name}.lp"
        if not lp.exists():
            continue
        try:
            p = load_lp(lp)
            st = HiGHSBackend().solve(p)
            if not st.is_optimal or st.basis_inverse is None:
                continue

            import clara.engine.standard_form as standard_form
            _, b_aug, _ = standard_form.add_upper_bound_rows(p)
            x_B = st.basis_inverse @ b_aug
            t, _ = wendell_tolerance_rhs(st.basis_inverse, x_B, p.b)  # original b

            region = analyzer.analyze(st, p)
            wendell_ts.append(t)
            cheby_rs.append(region.chebyshev_radius)
            oat_mins.append(region.min_oat_tolerance)
        except Exception:
            pass

    if wendell_ts:
        print(f"  {'':>20} | {'Mean':>10} | {'Median':>10} | {'Min':>10} | {'Max':>10}")
        print(f"  {'Wendell t':>20} | {np.mean(wendell_ts):>10.4f} | {np.median(wendell_ts):>10.4f} | "
              f"{min(wendell_ts):>10.4f} | {max(wendell_ts):>10.4f}")
        print(f"  {'Chebyshev r*':>20} | {np.mean(cheby_rs):>10.4f} | {np.median(cheby_rs):>10.4f} | "
              f"{min(cheby_rs):>10.4f} | {max(cheby_rs):>10.4f}")
        finite_oat = [o for o in oat_mins if o < 1e10]
        if finite_oat:
            print(f"  {'OAT α_min':>20} | {np.mean(finite_oat):>10.4f} | {np.median(finite_oat):>10.4f} | "
                  f"{min(finite_oat):>10.4f} | {max(finite_oat):>10.4f}")
        if len(wendell_ts) > 2:
            corr = np.corrcoef(wendell_ts, cheby_rs)[0, 1]
            print(f"  Correlation (Wendell t vs r*): r = {corr:.4f}")
    print()


# ============================================================
# Task 3: Runtime breakdown
# ============================================================

def run_task3():
    from clara.engine import HiGHSBackend
    from clara.explain.explainer import Explainer
    from clara.explain.types import DetailLevel
    from clara.reopt.attribution import ChangeAttributor
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.reoptimizer import Reoptimizer
    from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer
    from clara.reopt.types import ReoptDecision

    print("=== Task 3: Runtime Breakdown (case study n=20) ===")
    from benchmarks.scripts.supply_chain_case_study import create_base_problem, create_scenario_labor

    base, _, _ = create_base_problem()
    pert = create_scenario_labor(base, labor_q3_delta=-40)

    timings = {}
    reps = 3
    for _ in range(reps):
        t0 = time.perf_counter()
        state = HiGHSBackend().solve(base)
        timings.setdefault("solve", []).append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        Explainer().explain(state, level=DetailLevel.DETAILED, problem=base)
        timings.setdefault("explain", []).append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        SimultaneousRegionAnalyzer().analyze(state, base)
        timings.setdefault("region", []).append(time.perf_counter() - t0)

        change = ChangeDetector().detect(base, pert)
        pert_state = HiGHSBackend().solve(pert)

        t0 = time.perf_counter()
        ChangeAttributor().attribute(state, pert_state, base, pert, change, compute_shapley=False)
        timings.setdefault("attribution", []).append(time.perf_counter() - t0)

        from clara.reopt.analyzer import ImpactAnalyzer
        t0 = time.perf_counter()
        ImpactAnalyzer().analyze(state, change, base)
        timings.setdefault("impact", []).append(time.perf_counter() - t0)

        decision = ReoptDecision(should_reoptimize=True, reason="b", recommended_method="warm_start")
        t0 = time.perf_counter()
        Reoptimizer().reoptimize(state, pert, change, decision, old_problem=base)
        timings.setdefault("reopt", []).append(time.perf_counter() - t0)

    print(f"  {'Stage':<20} | {'Time (ms)':>10}")
    print(f"  {'-'*20}-+-{'-'*10}")
    total = 0
    for stage in ["solve", "explain", "region", "attribution", "impact", "reopt"]:
        mean_ms = np.mean(timings[stage]) * 1000
        total += mean_ms
        print(f"  {stage:<20} | {mean_ms:>10.2f}")
    print(f"  {'TOTAL':<20} | {total:>10.2f}")
    print()


# ============================================================
# Task 4: Correlations
# ============================================================

def run_task4():
    print("=== Task 4: ρ vs κ(B) Correlation ===")

    exp1 = RESULTS_DIR / "paper_exp1.csv"
    exp8 = RESULTS_DIR / "paper_exp8.csv"

    if not exp1.exists() or not exp8.exists():
        print("  Missing CSV files. Run collect_paper_data.py first.")
        return

    with open(exp1) as f:
        d1 = {r["instance"]: r for r in csv.DictReader(f)}
    with open(exp8) as f:
        d8 = list(csv.DictReader(f))

    kappas, rhos, d0s, degens = [], [], [], []
    for r in d8:
        name = r["instance"]
        if name in d1:
            kappas.append(float(d1[name]["condition_number"]))
            rhos.append(float(r["simultaneity_ratio"]))
            d0s.append(float(r["basis_robustness_d0"]))
            degens.append(int(d1[name]["degenerate_count"]))

    if len(kappas) > 2:
        r_kappa = np.corrcoef(kappas, rhos)[0, 1]
        r_d0 = np.corrcoef(d0s, rhos)[0, 1]
        print(f"  κ(B) vs ρ: r = {r_kappa:.4f} (n={len(kappas)})")
        print(f"  d₀ vs ρ:   r = {r_d0:.4f}")
        degen_any = sum(1 for d in degens if d > 0)
        print(f"  Degenerate instances: {degen_any}/{len(degens)}")
    print()


def main():
    run_task1()
    run_task2()
    run_task3()
    run_task4()


if __name__ == "__main__":
    main()
