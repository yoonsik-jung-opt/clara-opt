import os
os.environ["OMP_NUM_THREADS"] = "1"

"""Medium-scale Netlib warm-start experiment (solver-level).

Measures HiGHS cold-start vs. advanced-basis warm-start on medium
Netlib instances (m ~ 300-2,200) under RHS (Type R) and objective
(Type C) perturbations. Solves are measured at the solver level
(model already loaded; wall-clock around Highs.run() only), isolating
the reoptimization question from CLARA's prototype model-construction
layer. The basis is transferred as a raw HighsBasis object, so no
B^-1 reconstruction is required and instances with general variable
bounds are included.

Usage: python benchmarks/scripts/exp_medium_warmstart.py
Output: benchmarks/results/exp_medium_warmstart.csv
"""

import csv
import time
from pathlib import Path

import numpy as np

MPS_DIR = Path(__file__).parent.parent / "netlib" / "mps_medium"
RESULTS = Path(__file__).parent.parent / "results"

MAGNITUDES = [0.01, 0.05]
SEEDS = [42, 123, 456]
REPEATS = 3  # timing repeats; median taken


def load(path):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    return h


def timed_run(h):
    """Run and return (median wall seconds, iterations of last run)."""
    times = []
    iters = 0
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        h.run()
        times.append(time.perf_counter() - t0)
        iters = int(h.getInfoValue("simplex_iteration_count")[1])
        # re-set model state is not possible without reload; only the
        # first run does simplex work, later runs are no-ops -> time
        # only the first run.
        break
    return times[0], iters


def perturb(h, kind, mag, seed):
    """Apply a relative perturbation to the loaded model in place."""
    import highspy
    rng = np.random.RandomState(seed)
    lp = h.getLp()
    m = lp.num_row_
    n = lp.num_col_
    if kind == "R":
        lower = np.array(lp.row_lower_)
        upper = np.array(lp.row_upper_)
        for i in range(m):
            f = 1.0 + rng.uniform(-mag, mag)
            lo, up = lower[i], upper[i]
            new_lo = lo * f if np.isfinite(lo) else lo
            new_up = up * f if np.isfinite(up) else up
            if new_lo > new_up:
                new_lo, new_up = new_up, new_lo
            h.changeRowBounds(i, new_lo, new_up)
    else:  # C
        cost = np.array(lp.col_cost_)
        for j in range(n):
            if abs(cost[j]) > 1e-12:
                h.changeColCost(j, float(cost[j] * (1.0 + rng.uniform(-mag, mag))))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(MPS_DIR.glob("*.mps"))
    print(f"Medium warm-start on {len(files)} instances "
          f"({len(MAGNITUDES)} magnitudes x 2 types x {len(SEEDS)} seeds)...")

    for f in files:
        base = load(f)
        t_base0 = time.perf_counter()
        base.run()
        t_base = time.perf_counter() - t_base0
        if str(base.getModelStatus()).split(".")[-1] != "kOptimal":
            print(f"  {f.stem}: base not optimal, skipped")
            continue
        base_basis = base.getBasis()
        base_iters = int(base.getInfoValue("simplex_iteration_count")[1])
        lp = base.getLp()
        m, n = lp.num_row_, lp.num_col_

        for kind in ("R", "C"):
            for mag in MAGNITUDES:
                for seed in SEEDS:
                    # cold
                    hc = load(f)
                    perturb(hc, kind, mag, seed)
                    t_cold, it_cold = timed_run(hc)
                    zc = hc.getInfoValue("objective_function_value")[1]
                    st_c = str(hc.getModelStatus()).split(".")[-1]

                    # warm (advanced basis from base optimal)
                    hw = load(f)
                    perturb(hw, kind, mag, seed)
                    hw.setBasis(base_basis)
                    t_warm, it_warm = timed_run(hw)
                    zw = hw.getInfoValue("objective_function_value")[1]
                    st_w = str(hw.getModelStatus()).split(".")[-1]

                    if st_c != "kOptimal" or st_w != "kOptimal":
                        continue
                    match = abs(zc - zw) <= 1e-6 * max(1.0, abs(zc))
                    rows.append({
                        "instance": f.stem, "m": m, "n": n,
                        "type": kind, "magnitude": mag, "seed": seed,
                        "base_iters": base_iters,
                        "cold_iters": it_cold, "warm_iters": it_warm,
                        "cold_time": f"{t_cold:.6f}",
                        "warm_time": f"{t_warm:.6f}",
                        "iter_ratio": f"{it_cold / max(it_warm, 1):.2f}",
                        "time_ratio": f"{t_cold / max(t_warm, 1e-9):.2f}",
                        "optimal_match": match,
                    })
        print(f"  {f.stem} (m={m}, n={n}): done")

    out = RESULTS / "exp_medium_warmstart.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import statistics as st
    print(f"\nDone: {len(rows)} pairs -> {out}")
    match = sum(1 for r in rows if r["optimal_match"])
    print(f"optimal match: {match}/{len(rows)}")
    ir = [float(r["iter_ratio"]) for r in rows]
    tr = [float(r["time_ratio"]) for r in rows]
    print(f"iter ratio: mean {st.mean(ir):.1f}x median {st.median(ir):.1f}x | "
          f"solver wall-clock: mean {st.mean(tr):.1f}x median {st.median(tr):.1f}x")


if __name__ == "__main__":
    main()
