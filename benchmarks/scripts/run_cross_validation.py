import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

"""Exp 1: B^-1 reconstruction validation (HiGHS backend).

Replaces the old internal-vs-HiGHS cross-validation (the internal
engine was removed in v0.3.0). For every instance, validates that the
basis inverse reconstructed from the HiGHS optimal basis is
numerically consistent with the reported solution:

    binv_identity_err = max | B^-1 B - I |
    xb_match_err      = max | x_B[pos] - x*_j |   (structural basics)
    dual_match_err    = max | (c_B^T B^-1)_i - y_i |

Also records instance dimensions and nonzeros (paper Netlib size
table) and the diagnostics kappa(B), d0, degenerate count (Online
Supp. instance statistics).

Usage: python benchmarks/scripts/run_cross_validation.py
Output: benchmarks/results/exp1_reconstruction.csv
"""

import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np

NETLIB_DIR = Path(__file__).parent.parent / "netlib" / "mps"
RANDOM_DIR = Path(__file__).parent.parent / "instances" / "random"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def process_single(task):
    import time
    from pathlib import Path

    filepath = task["filepath"]

    try:
        from clara.engine import HiGHSBackend
        import clara.engine.standard_form as standard_form

        p = Path(filepath)
        if p.suffix.lower() == ".mps":
            from clara.io.mps_parser import read_mps
            problem = read_mps(p)
        else:
            from clara.io.lp_parser import read_lp
            problem = read_lp(p)

        n = problem.num_variables
        m = problem.num_constraints
        nonzeros = int((problem.A != 0).sum())

        t0 = time.perf_counter()
        state = HiGHSBackend().solve(problem)
        elapsed = time.perf_counter() - t0

        row = {
            "instance": p.stem,
            "n_vars": n,
            "n_cons": m,
            "nonzeros": nonzeros,
            "status": state.status.name,
            "optimal_value": f"{state.optimal_value:.6f}" if state.is_optimal else "",
            "iterations": state.iteration_count,
            "time": f"{elapsed:.4f}",
            "recon_ok": "",
            "binv_identity_err": "",
            "xb_match_err": "",
            "dual_match_err": "",
            "d0": "",
            "cond_B": "",
            "degenerate_count": "",
        }

        if not state.is_optimal:
            return row

        if state.basis_inverse is None or state.basis_indices is None:
            row["recon_ok"] = "False"
            return row

        A_aug, b_aug, _ = standard_form.add_upper_bound_rows(problem)
        m_aug = A_aug.shape[0]
        A_full = np.hstack([A_aug, np.eye(m_aug)])
        B = A_full[:, list(state.basis_indices)]
        B_inv = state.basis_inverse

        identity_err = float(np.max(np.abs(B_inv @ B - np.eye(m_aug))))

        x_B = B_inv @ b_aug
        values = {v.name: v.value for v in state.variables}
        xb_errs = [abs(x_B[pos] - values[problem.var_names[j]])
                   for pos, j in enumerate(state.basis_indices) if j < n]
        xb_err = float(max(xb_errs)) if xb_errs else 0.0

        c_full = np.zeros(n + m_aug)
        c_full[:n] = problem.c
        c_B = np.array([c_full[j] for j in state.basis_indices])
        y = c_B @ B_inv
        duals = np.array([c.dual_value for c in state.constraints])
        dual_err = float(np.max(np.abs(y[:m] - duals))) if m else 0.0

        row.update({
            "recon_ok": "True",
            "binv_identity_err": f"{identity_err:.2e}",
            "xb_match_err": f"{xb_err:.2e}",
            "dual_match_err": f"{dual_err:.2e}",
            "d0": f"{state.basis_robustness_d0:.6e}",
            "cond_B": f"{state.condition_number:.4e}",
            "degenerate_count": state.degenerate_count,
        })
        return row
    except Exception as e:
        return {
            "instance": Path(filepath).stem, "n_vars": "", "n_cons": "",
            "nonzeros": "", "status": f"ERROR:{str(e)[:40]}",
            "optimal_value": "", "iterations": "", "time": "",
            "recon_ok": "", "binv_identity_err": "", "xb_match_err": "",
            "dual_match_err": "", "d0": "", "cond_B": "", "degenerate_count": "",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []

    # Netlib — all instances, including blend (solvable with HiGHS)
    for f in sorted(NETLIB_DIR.glob("*.mps")):
        files.append(f)

    # Random (solvable only)
    manifest = RANDOM_DIR / "manifest.csv"
    if manifest.exists():
        with open(manifest) as fh:
            for row in csv.DictReader(fh):
                if row["match"] == "✓":
                    lp = RANDOM_DIR / f"{row['instance']}.lp"
                    if lp.exists():
                        files.append(lp)

    tasks = [{"filepath": str(f)} for f in files]

    print(f"Exp 1: B^-1 reconstruction validation on {len(tasks)} instances "
          f"(workers={args.workers})...")
    results = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_single, tasks)):
            if r is not None:
                results.append(r)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(tasks)}]")

    results.sort(key=lambda r: r["instance"])
    out = RESULTS_DIR / "exp1_reconstruction.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if r["recon_ok"] == "True"]
    print(f"Done: {len(results)} instances -> {out}")
    print(f"Reconstruction OK: {len(ok)}/{sum(1 for r in results if r['status']=='OPTIMAL')} optimal instances")
    if ok:
        ie = [float(r["binv_identity_err"]) for r in ok]
        xe = [float(r["xb_match_err"]) for r in ok]
        de = [float(r["dual_match_err"]) for r in ok]
        print(f"max ||B^-1 B - I||: {max(ie):.2e} | max x_B err: {max(xe):.2e} | max dual err: {max(de):.2e}")


if __name__ == "__main__":
    main()
