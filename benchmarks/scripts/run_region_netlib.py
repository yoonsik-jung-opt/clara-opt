"""Simultaneous sensitivity region on the (degenerate) Netlib instances.

Complements run_region_benchmark.py (random, non-degenerate instances):
on real LPs every optimal basis is degenerate, the basis robustness
radius d0 is zero, and the origin-centered ball therefore says nothing
(cf. Kilinc-Karzan et al., ORL 2009, on stability radii under-approximating
stability regions). The Chebyshev radius r* and its face-restricted
variant r*_F (degenerate rhs coordinates fixed) recover the region.

Output: benchmarks/results/exp8_netlib_region.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clara.engine.highs_backend import HiGHSBackend  # noqa: E402
from clara.io.mps_parser import read_mps  # noqa: E402
from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer  # noqa: E402

NETLIB_DIR = ROOT / "benchmarks" / "netlib" / "mps"
OUT = ROOT / "benchmarks" / "results" / "exp8_netlib_region.csv"


def main() -> None:
    analyzer = SimultaneousRegionAnalyzer()
    rows = []
    for path in sorted(NETLIB_DIR.glob("*.mps")):
        problem = read_mps(path)
        state = HiGHSBackend().solve(problem)
        if not state.is_optimal or state.basis_inverse is None:
            print(f"  {path.stem}: skipped (no reconstruction)")
            continue
        region = analyzer.analyze(state, problem)
        rows.append({
            "instance": path.stem,
            "m": problem.num_constraints,
            "n": problem.num_variables,
            "m_aug": state.basis_inverse.shape[0],
            "degenerate_basics": state.degenerate_count,
            "d0": max(0.0, state.basis_robustness_d0),
            "chebyshev_radius": max(0.0, region.chebyshev_radius),
            "chebyshev_radius_face": region.chebyshev_radius_face,
            "min_oat_tolerance": region.min_oat_tolerance,
            "simultaneity_ratio": region.simultaneity_ratio,
            "n_degenerate_params": region.n_degenerate_params,
            "n_two_sided_zero": region.n_two_sided_zero,
        })
        print(f"  {path.stem:10} degen={state.degenerate_count:3d} d0={state.basis_robustness_d0:.1e} "
              f"r*={region.chebyshev_radius:.4f} r*_F={region.chebyshev_radius_face:.4f} "
              f"alpha_min={region.min_oat_tolerance:.4f} rho={region.simultaneity_ratio:.2f} "
              f"degen_params={region.n_degenerate_params}/{problem.num_constraints} "
              f"two_sided_zero={region.n_two_sided_zero}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone: {len(rows)} instances -> {OUT}")


if __name__ == "__main__":
    main()
