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
    ax.boxplot(data, labels=[l.replace("_", "\n") for l in labels])
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


def main():
    print("Generating figures:")
    gen_scalability_fig()
    gen_warmstart_fig()
    gen_decision_fig()
    print("Done.")


if __name__ == "__main__":
    main()
