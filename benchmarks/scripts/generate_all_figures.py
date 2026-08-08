"""Generate all figures from experiment results.

Usage: python benchmarks/scripts/generate_all_figures.py
Output: paper/figures/exp*.pdf
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent.parent / "paper" / "figures"


def _read_csv(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def gen_scalability_fig():
    """Solve time vs problem size (log-log)."""
    rows = _read_csv("exp6_scalability.csv")
    if not rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed, skipping figures")
        return

    by_size = {}
    for r in rows:
        n = int(r["n_vars"])
        by_size.setdefault(n, {"internal": [], "highs": []})
        by_size[n]["internal"].append(float(r["time_internal"]))
        by_size[n]["highs"].append(float(r["time_highs"]))

    sizes = sorted(by_size.keys())
    ti = [np.mean(by_size[n]["internal"]) for n in sizes]
    th = [np.mean(by_size[n]["highs"]) for n in sizes]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.loglog(sizes, ti, "o-", label="Internal Simplex", markersize=4)
    ax.loglog(sizes, th, "s--", label="HiGHS", markersize=4)
    ax.set_xlabel("Number of variables ($n$)")
    ax.set_ylabel("Solve time (s)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "exp6_scalability.pdf"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


def gen_warmstart_fig():
    """Speedup distribution by method."""
    rows = _read_csv("exp3_warmstart.csv")
    if not rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    methods = {}
    for r in rows:
        m = r["method_used"]
        if m != "scratch":
            methods.setdefault(m, []).append(float(r["speedup"]))

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    labels = sorted(methods.keys())
    data = [methods[l] for l in labels]
    ax.boxplot(data)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels])
    ax.set_ylabel("Speedup (× scratch)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    out = FIGURES_DIR / "exp3_warmstart_speedup.pdf"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


def gen_decision_fig():
    """Decision accuracy by change type."""
    rows = _read_csv("exp2_reopt_decision.csv")
    if not rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    by_type = {}
    for r in rows:
        t = r["change_type"]
        by_type.setdefault(t, {"total": 0, "correct": 0})
        by_type[t]["total"] += 1
        if r["correct"] == "True":
            by_type[t]["correct"] += 1

    types = sorted(by_type.keys())
    accs = [by_type[t]["correct"] / max(by_type[t]["total"], 1) for t in types]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar(range(len(types)), accs)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()

    out = FIGURES_DIR / "exp2_decision_accuracy.pdf"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


def gen_attribution_fig():
    """Stacked bar chart: attribution decomposition."""
    rows = _read_csv("exp7_attribution.csv")
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Pick up to 5 representative instances
    sample = rows[:5]
    names = [r["base_instance"][:12] for r in sample]
    rhs = [float(r["rhs_effect"]) for r in sample]
    obj = [float(r["obj_effect"]) for r in sample]
    inter = [float(r["interaction_effect"]) for r in sample]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(names))
    ax.bar(x, rhs, label="RHS", color="steelblue")
    ax.bar(x, obj, bottom=rhs, label="Obj", color="indianred")
    bottoms = [r + o for r, o in zip(rhs, obj)]
    ax.bar(x, inter, bottom=bottoms, label="Interaction", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel(r"$\Delta z$ contribution")
    ax.legend(fontsize=6)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "exp7_attribution.pdf"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


def gen_region_projection_fig():
    """2D projection of Albici sensitivity region."""
    import json
    proj_file = RESULTS_DIR / "exp8_albici_projections.json"
    if not proj_file.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        return

    with open(proj_file) as f:
        proj = json.load(f)

    for key, verts in proj.items():
        if not verts:
            continue
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        xs, ys = zip(*verts)
        poly = Polygon(verts, closed=True, alpha=0.3, color="steelblue", label="Simultaneous")
        ax.add_patch(poly)
        ax.plot(xs + (xs[0],), ys + (ys[0],), "b-", linewidth=0.5)
        ax.plot(0, 0, "ko", markersize=4, label="Current")
        ax.set_xlabel(f"Δb[{key.split('_')[0]}]")
        ax.set_ylabel(f"Δb[{key.split('_')[1]}]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / f"exp8_region_{key}.pdf"
        fig.savefig(out, dpi=150)
        plt.close()
        print(f"  {out.name}")
        break  # just one projection


def gen_region_histogram_fig():
    """Histogram of simultaneity ratios."""
    rows = _read_csv("exp8_region.csv")
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    ratios = [float(r["simultaneity_ratio"]) for r in rows if float(r["simultaneity_ratio"]) < 1e6]
    if not ratios:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.hist(ratios, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(np.mean(ratios), color="red", linestyle="--", label=f"Mean={np.mean(ratios):.2f}")
    ax.set_xlabel("Simultaneity ratio")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "exp8_ratio_histogram.pdf"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  {out.name}")


def main():
    print("Generating figures:")
    gen_scalability_fig()
    gen_warmstart_fig()
    gen_decision_fig()
    gen_attribution_fig()
    gen_region_projection_fig()
    gen_region_histogram_fig()
    print("Done.")


if __name__ == "__main__":
    main()
