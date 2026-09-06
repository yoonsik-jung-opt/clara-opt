import os
os.environ["OMP_NUM_THREADS"] = "1"

"""Generate the paper's figure PDFs (fig4, fig5-left/right, fig7)
from the current benchmark result CSVs.

Usage: python benchmarks/scripts/generate_paper_figures.py [outdir]
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "paper_figures"


def fig5_left():
    """2D projection of S for the Albici base instance (S1 vs S3)."""
    from clara.io.lp_parser import read_lp
    from clara.engine import HiGHSBackend
    from clara.reopt.sensitivity_region import SimultaneousRegionAnalyzer

    p = read_lp(ROOT.parent / "tests" / "fixtures" / "albici_base.lp")
    s = HiGHSBackend().solve(p)
    an = SimultaneousRegionAnalyzer()
    reg = an.analyze(s, p)
    names = list(p.constraint_names)
    proj = json.load(open(RESULTS / "exp8_albici_projections.json"))
    key, i, j = "0_2", 0, 2
    verts = proj[key]
    # Asymmetric OAT box [-alpha^-, alpha^+] per Definition 1 of the paper
    lo_i, hi_i = s.sensitivity.rhs_ranges[names[i]]
    lo_j, hi_j = s.sensitivity.rhs_ranges[names[j]]
    bi, bj = float(p.b[i]), float(p.b[j])
    x0, x1 = lo_i - bi, hi_i - bi
    y0, y1 = lo_j - bj, hi_j - bj

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.add_patch(Polygon(verts, closed=True, alpha=0.35, color="steelblue"))
    xs, ys = zip(*verts)
    ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], "b-", lw=0.6,
            label=r"$\mathcal{S}$ (simultaneous)")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           ls="--", ec="crimson", lw=1.2, label="OAT box"))
    ax.set_xlim(x0 - 0.08 * (x1 - x0), x1 + 0.08 * (x1 - x0))
    ax.set_ylim(y0 - 0.08 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlabel(f"$\\Delta b$ ({names[i]})")
    ax.set_ylabel(f"$\\Delta b$ ({names[j]})")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_left_region.pdf", dpi=150)
    plt.close()


def fig5_right():
    rows = list(csv.DictReader(open(RESULTS / "exp8_region.csv")))
    rho = [float(r["simultaneity_ratio"]) for r in rows
           if float(r["chebyshev_radius"]) > 1e-10]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.hist(np.clip(rho, 0, 25), bins=40, color="steelblue",
            edgecolor="white", lw=0.3)
    ax.axvline(np.mean(rho), color="darkorange", lw=1.3,
               label=f"mean {np.mean(rho):.1f}")
    ax.axvline(np.median(rho), color="crimson", lw=1.3, ls="--",
               label=f"median {np.median(rho):.1f}")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"Simultaneity ratio $\rho = r^*/\alpha_{\min}$ (clipped at 25)")
    ax.set_ylabel("Instances")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_right_histogram.pdf", dpi=150)
    plt.close()


def fig7():
    rows = list(csv.DictReader(open(RESULTS / "exp3_warmstart.csv")))
    it = [float(r["speedup"]) for r in rows]
    wc = [float(r["time_scratch"]) / max(float(r["time_warmstart"]), 1e-12)
          for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    axes[0].hist(np.clip(it, 0, 80), bins=40, color="steelblue",
                 edgecolor="white", lw=0.3)
    axes[0].axvline(np.median(it), color="darkorange", lw=1.2,
                    label=f"median {np.median(it):.1f}$\\times$")
    axes[0].set_xlabel("Iteration reduction (cold / warm)")
    axes[0].set_ylabel("Pairs")
    axes[0].legend(fontsize=7)
    axes[1].hist(wc, bins=40, color="seagreen", edgecolor="white", lw=0.3)
    axes[1].axvline(np.median(wc), color="darkorange", lw=1.2,
                    label=f"median {np.median(wc):.2f}$\\times$")
    axes[1].set_xlabel("Wall-clock speedup")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_warmstart.pdf", dpi=150)
    plt.close()


def fig4():
    """Attribution decomposition figure: reuse generate_all_figures output."""
    import shutil
    src = ROOT.parent / "paper" / "figures" / "exp7_attribution.pdf"
    if src.exists():
        shutil.copy(src, OUT / "fig4_attribution.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig5_left()
    fig5_right()
    fig7()
    fig4()
    print(f"Paper figures written to {OUT}")


if __name__ == "__main__":
    main()
