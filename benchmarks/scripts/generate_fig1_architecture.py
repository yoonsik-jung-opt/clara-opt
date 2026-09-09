"""Generate the paper's architecture diagram (fig1_architecture.pdf).

Reflects the v0.3.0 architecture: single HiGHS backend, B^-1
reconstruction from the reported optimal basis, and the
advanced-basis warm-start feedback loop.

Usage: python benchmarks/scripts/generate_fig1_architecture.py [outdir]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).parent.parent / "results" / "paper_figures")

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})

BLUE = "#dce7f5"
CREAM = "#fdf3d6"
GRAY = "#e9e9e9"
EDGE = "#444444"
ARROW = "#3d3d3d"


def box(ax, cx, cy, w, h, lines, fc, lw=1.0, fontsizes=None, bold_first=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        fc=fc, ec="black" if lw > 1.5 else EDGE, lw=lw, zorder=2))
    n = len(lines)
    if fontsizes is None:
        fontsizes = [9.5] * n
    top = cy + (n - 1) * 0.5 * 0.046
    for k, (line, fs) in enumerate(zip(lines, fontsizes)):
        weight = "bold" if (bold_first and k == 0) else "normal"
        ax.text(cx, top - k * 0.046, line, ha="center", va="center",
                fontsize=fs, weight=weight, zorder=3)


def arrow(ax, p, q, style="-|>", ls="-", lw=1.4, shrinkA=2, shrinkB=2,
          connectionstyle=None):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, linestyle=ls, mutation_scale=13,
        color=ARROW, lw=lw, shrinkA=shrinkA, shrinkB=shrinkB, zorder=1,
        connectionstyle=connectionstyle or "arc3,rad=0.0"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.set_xlim(0, 1.38)
    ax.set_ylim(0, 0.62)
    ax.axis("off")

    ymid = 0.36

    # --- Input ---
    box(ax, 0.105, ymid, 0.155, 0.105,
        ["LP problem", r"$(A,\,b,\,c)$"], BLUE)

    # --- HiGHS backend ---
    box(ax, 0.305, ymid, 0.175, 0.105,
        ["HiGHS", "(cold / advanced basis)"], BLUE,
        fontsizes=[9.5, 7.6])

    # --- B^-1 reconstruction ---
    box(ax, 0.505, ymid, 0.155, 0.105,
        [r"$B^{-1}$", "reconstruction"], BLUE,
        fontsizes=[9.5, 8.5])

    # --- SolveState (emphasized) ---
    box(ax, 0.735, ymid, 0.22, 0.23,
        ["Retained basis",
         r"$B^{-1},\ x_B = B^{-1} b$",
         r"$y = c_B^{\top} B^{-1},\ \bar{c}$",
         r"$\kappa(B),\ d_0$"],
        CREAM, lw=2.2,
        fontsizes=[10.0, 9.0, 9.0, 9.0], bold_first=True)

    # --- Right column modules ---
    rx, rw = 1.185, 0.33
    box(ax, rx, 0.535, rw, 0.1,
        ["Sensitivity geometry (Sec. 4):",
         r"$\mathcal{S},\ r^*,\ r^*_F,\ d_0,\ \delta^*$"],
        GRAY, fontsizes=[9.5, 9.0])

    box(ax, rx, 0.36, rw, 0.115,
        ["Reoptimization decisions (Sec. 5):",
         r"screen $\rightarrow$ bound $\rightarrow$ certify /",
         "primal / dual / parametric"],
        GRAY, fontsizes=[9.5, 8.3, 8.3])

    box(ax, rx, 0.175, rw, 0.1,
        ["Comparison report:",
         r"binding, attribution, $\eta$"],
        GRAY, fontsizes=[9.5, 9.0])

    # --- Main flow arrows ---
    arrow(ax, (0.185, ymid), (0.222, ymid))
    arrow(ax, (0.388, ymid), (0.425, ymid))
    arrow(ax, (0.585, ymid), (0.627, ymid))
    # SolveState -> three modules
    arrow(ax, (0.845, 0.43), (1.015, 0.535))
    arrow(ax, (0.845, ymid), (1.015, ymid))
    arrow(ax, (0.845, 0.29), (1.015, 0.185))

    # --- Perturbation input ---
    ax.text(0.875, 0.066, r"perturbation $(\Delta b,\, \Delta c)$",
            fontsize=9.5, style="italic", ha="center", va="center")
    arrow(ax, (0.93, 0.10), (1.013, 0.332))

    # --- Advanced-basis feedback loop ---
    arrow(ax, (1.043, 0.301), (0.305, 0.305),
          ls=(0, (4, 3)), lw=1.1, shrinkA=4, shrinkB=4,
          connectionstyle="arc3,rad=-0.36")
    ax.text(0.52, 0.058, "old basis (advanced-basis warm start)",
            fontsize=8.0, color=ARROW, ha="center", va="center",
            style="italic")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig1_architecture.pdf", dpi=200)
    fig.savefig(OUT / "fig1_architecture.png", dpi=160)
    plt.close()
    print(f"fig1 written to {OUT}")


if __name__ == "__main__":
    main()
