import os
os.environ["OMP_NUM_THREADS"] = "1"

"""Exp 2b: Realized opportunity cost of the skip path.

The decision experiment (Exp 2) scores a skip as a false negative
whenever the perturbed optimum differs from the old optimum by more
than the tolerance — implicitly assuming that "skip" means "keep the
stale solution". CLARA's skip path, however, performs a zero-pivot
basis recompute (x_B = B^-1 b_new; automatic fallback to warm-start
when the old basis is primal infeasible). This experiment executes
that actual skip path on every skipped pair and measures the realized
relative loss |z_skip_path - z_true| / |z_true|.

Usage: python benchmarks/scripts/run_skip_realized_loss.py
Output: benchmarks/results/exp2b_skip_realized_loss.csv
"""

import csv
from pathlib import Path

import numpy as np

RANDOM = Path(__file__).parent.parent / "instances" / "random"
PERT = Path(__file__).parent.parent / "instances" / "perturbations"
RESULTS = Path(__file__).parent.parent / "results"


def main():
    from clara.io.lp_parser import read_lp
    from clara.engine import HiGHSBackend
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.reoptimizer import Reoptimizer
    from clara.reopt.types import ReoptDecision

    rows = list(csv.DictReader(open(RESULTS / "exp2_reopt_decision.csv")))

    def b(x):
        return x.strip().lower() in ("true", "1", "yes")

    skips = [r for r in rows if not b(r["should_reoptimize"])]
    print(f"Exp 2b: realized loss of the skip path on {len(skips)} skipped pairs...")

    base_cache = {}
    out = []
    for r in skips:
        base, pert = r["base_instance"], r["perturbation"]
        if base not in base_cache:
            p = read_lp(RANDOM / f"{base}.lp")
            base_cache[base] = (p, HiGHSBackend().solve(p))
        old_p, old_s = base_cache[base]
        cand = sorted((PERT / base).glob(f"{base}_{pert}*.lp"))
        alt = [c for c in cand if c.stem == pert]
        new_p = read_lp(alt[0] if alt else cand[0])
        change = ChangeDetector().detect(old_p, new_p)
        dec = ReoptDecision(should_reoptimize=False, reason="skip path",
                            recommended_method="recompute")
        res = Reoptimizer().reoptimize(old_s, new_p, change, dec, old_problem=old_p)
        z_free = res.new_state.optimal_value
        z_true = HiGHSBackend().solve(new_p).optimal_value
        loss = abs(z_free - z_true) / max(abs(z_true), 1e-10)
        out.append({
            "base_instance": base,
            "perturbation": pert,
            "change_type": r["change_type"],
            "oracle_decision": r["oracle_decision"],
            "skip_method_used": res.method_used,
            "z_skip_path": f"{z_free:.8f}",
            "z_true": f"{z_true:.8f}",
            "realized_loss": f"{loss:.6e}",
        })

    with open(RESULTS / "exp2b_skip_realized_loss.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)

    losses = np.array([float(r["realized_loss"]) for r in out])
    fallbacks = sum(1 for r in out if r["skip_method_used"] != "recompute")
    fn = [float(r["realized_loss"]) for r in out
          if b(r["oracle_decision"])]
    print(f"Done -> exp2b_skip_realized_loss.csv")
    print(f"recompute: {len(out)-fallbacks} | auto-fallback: {fallbacks}")
    print(f"realized loss: max={losses.max():.2e} mean={losses.mean():.2e} "
          f"| zero (<1e-9): {(losses<1e-9).sum()}/{len(losses)}")
    if fn:
        fn = np.array(fn)
        print(f"on oracle-positive skips (old 'FN'): {len(fn)} pairs, "
              f"zero-loss {(fn<1e-9).sum()}, max {fn.max():.2e}")


if __name__ == "__main__":
    main()
