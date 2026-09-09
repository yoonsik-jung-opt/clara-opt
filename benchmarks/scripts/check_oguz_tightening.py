"""Audit of the tightened Oguz screening bound against the realized
opportunity cost on every Type C perturbation of the random suite.

For each (base, perturbation) pair:
  theta_O       componentwise max relative objective change (Oguz)
  raw           2 theta_O / (1 + theta_O)                       [Oguz]
  alpha_oat     min one-sided relative OAT objective tolerance
                (what ImpactAnalyzer currently calls the Wendell tolerance)
  tau_w         Wendell (1985) objective tolerance: all c_j may move
                simultaneously by tau_w |c_j| with the basis optimal
  code          2 t'/(1+t'),  t' = (theta_O - a)/(1 + a), a = alpha_oat
  rig(a)        2 (theta_O - a) / ((1 - a)(1 + theta_O)) -- derived bound
                with base objective c'' = clip(c', (1 -/+ a) c)
  actual        (z'* - c'^T x*) / |z'*|   (stale-solution opportunity cost)

A bound is violated when actual > bound + 1e-9.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "scripts"))
from generate_perturbations import generate_perturbation  # noqa: E402

from clara.engine import HiGHSBackend  # noqa: E402
from clara.engine import standard_form  # noqa: E402
from clara.io.lp_parser import read_lp  # noqa: E402
from clara.model.problem import LPProblem  # noqa: E402

RANDOM = ROOT / "benchmarks" / "instances" / "random"
SPECS = [("C_small", 0.05), ("C_medium", 0.20), ("C_large", 0.50),
         ("C_tiny05", 0.005), ("C_tiny1", 0.01), ("C_tiny2", 0.02)]
SEEDS = (42, 123, 456)


def wendell_obj_tolerance(problem, state):
    A_aug, _, _ = standard_form.add_upper_bound_rows(problem)
    m_aug = A_aug.shape[0]
    n = problem.num_variables
    A_full = np.hstack([A_aug, np.eye(m_aug)])
    c_full = np.zeros(n + m_aug)
    c_full[:n] = problem.c
    basis = list(state.basis_indices)
    B_inv = state.basis_inverse
    c_B = c_full[basis]
    y = c_B @ B_inv
    sign = 1.0 if problem.sense == "maximize" else -1.0
    tau = np.inf
    bset = set(basis)
    for j in range(n + m_aug):
        if j in bset:
            continue
        rc = c_full[j] - y @ A_full[:, j]
        col = B_inv @ A_full[:, j]
        denom = abs(c_full[j]) + np.sum(np.abs(c_B) * np.abs(col))
        if denom > 1e-12:
            tau = min(tau, max(-sign * rc, 0.0) / denom)
    return float(tau)


def oat_obj_tolerance(problem, state):
    alpha = np.inf
    for j, name in enumerate(problem.var_names):
        lo, hi = state.sensitivity.obj_coeff_ranges[name]
        c_j = problem.c[j]
        if abs(c_j) > 1e-10:
            dn = abs((c_j - lo) / c_j) if np.isfinite(lo) else np.inf
            up = abs((hi - c_j) / c_j) if np.isfinite(hi) else np.inf
            alpha = min(alpha, dn, up)
    return float(alpha)


def bounds(theta, a):
    raw = 2 * theta / (1 + theta)
    if not np.isfinite(a) or theta <= a:
        return raw, 0.0, 0.0
    tp = (theta - a) / (1 + a)
    code = 2 * tp / (1 + tp)
    rig = 2 * (theta - a) / ((1 - a) * (1 + theta)) if a < 1 else raw
    return raw, code, min(rig, raw)


def main():
    rows = []
    for f in sorted(RANDOM.glob("rand_*.lp")):
        old = read_lp(f)
        st = HiGHSBackend().solve(old)
        if not st.is_optimal or st.basis_inverse is None:
            continue
        a_oat = oat_obj_tolerance(old, st)
        tau = wendell_obj_tolerance(old, st)
        x_old = np.array([v.value for v in st.variables])
        for spec, mag in SPECS:
            for seed in SEEDS:
                out = generate_perturbation(old, spec, "C", 0, mag, seed)
                if out is None:
                    continue
                new_c, _, dc, _ = out
                new = LPProblem(c=new_c, A=old.A, b=old.b, sense=old.sense, var_names=old.var_names,
                                constraint_names=old.constraint_names, upper_bounds=old.upper_bounds)
                ns = HiGHSBackend().solve(new)
                if not ns.is_optimal:
                    continue
                z_new = ns.optimal_value
                z_stale = float(new.c @ x_old)
                actual = (z_new - z_stale) / max(abs(z_new), 1e-12)
                theta = max(abs(dc[j] / old.c[j]) for j in range(len(dc)) if abs(old.c[j]) > 1e-10)
                raw, code, rig_oat = bounds(theta, a_oat)
                _, _, rig_w = bounds(theta, tau)
                rows.append({"base": f.stem, "spec": spec, "seed": seed, "theta": theta,
                             "alpha_oat": a_oat, "tau_w": tau, "raw": raw, "code": code,
                             "rig_alpha_oat": rig_oat, "rig_tau_w": rig_w, "actual": actual})
    out = ROOT / "benchmarks" / "results" / "exp11_oguz_tightening_audit.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    r = {k: np.array([row[k] for row in rows]) for k in rows[0] if k not in ("base", "spec", "seed")}
    print(f"{len(rows)} Type C pairs")
    print(f"alpha_oat > tau_w (OAT tolerance exceeds Wendell): {np.sum(r['alpha_oat'] > r['tau_w'] + 1e-12)}"
          f"  median alpha_oat/tau_w = {np.median(r['alpha_oat'] / np.maximum(r['tau_w'], 1e-15)):.2f}")
    for k in ("raw", "code", "rig_alpha_oat", "rig_tau_w"):
        viol = r["actual"] > r[k] + 1e-9
        print(f"{k:14} violations {viol.sum():4d}  max excess {np.max(r['actual'] - r[k]):.5f}"
              f"  bound=0 while actual>1e-6: {np.sum((r[k] <= 0) & (r['actual'] > 1e-6))}")
    print("-> saved", out)


if __name__ == "__main__":
    main()
