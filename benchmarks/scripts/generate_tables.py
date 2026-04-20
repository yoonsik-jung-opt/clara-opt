"""Generate LaTeX tables for the MPC paper from benchmark results.

Usage:
    python benchmarks/scripts/generate_tables.py

Output:
    paper/tables/netlib_cross_validation.tex
    paper/tables/reopt_comparison.tex
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FINAL_DIR = RESULTS_DIR / "final"
TABLES_DIR = Path(__file__).parent.parent.parent / "paper" / "tables"


def generate_netlib_table():
    """Generate Netlib cross-validation LaTeX table."""
    # Find most recent netlib CSV
    csvs = sorted(RESULTS_DIR.glob("netlib_*.csv"))
    if not csvs:
        print("No netlib results found. Run run_benchmark.py first.")
        return

    csv_path = csvs[-1]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-validation of Internal Simplex vs.\ HiGHS on Netlib LP instances.}",
        r"\label{tab:netlib-cross}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Instance & Vars & Cons & HiGHS Opt.\ & Internal Opt.\ & Match & Note \\",
        r"\midrule",
    ]

    for r in rows:
        h_opt = f"{float(r['highs_optimal']):.4f}" if r["highs_optimal"] else "---"
        i_opt = f"{float(r['internal_optimal']):.4f}" if r["internal_optimal"] else "---"
        match = r["match"] if r["match"] else "---"
        match_tex = r"\checkmark" if match == "✓" else ("$\\times$" if match == "✗" else "---")
        note = r["note"].replace("_", r"\_") if r["note"] != "OK" and r["internal_optimal"] == "" else ""
        lines.append(
            f"  {r['instance']} & {r['variables']} & {r['constraints']} "
            f"& {h_opt} & {i_opt} & {match_tex} & {note} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = TABLES_DIR / "netlib_cross_validation.tex"
    out.write_text("\n".join(lines))
    print(f"  Generated: {out}")


def generate_reopt_table():
    """Generate reoptimization comparison LaTeX table."""
    csv_path = FINAL_DIR / "reopt_comparison.csv"
    if not csv_path.exists():
        print("No reopt results found. Run run_reopt_benchmark.py first.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reoptimization comparison for Albici et al.\ (2010) scenarios. "
        r"Warm-start methods use $B^{-1}$ from the original solution.}",
        r"\label{tab:reopt-comparison}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Method & Pivots & Scratch & Speedup & Optimal \\",
        r"\midrule",
    ]

    for r in rows:
        method = r["method"].replace("_", r"\_")
        speedup = r["speedup"] if r["speedup"] else "---"
        speedup_tex = f"${speedup}$" if speedup != "---" else "---"
        if "∞" in speedup:
            speedup_tex = r"$\infty$"
        scenario = r["scenario"].replace("→", r"$\to$")
        lines.append(
            f"  {scenario} & {method} & {r['pivots']} & {r['scratch_pivots']} "
            f"& {speedup_tex} & {float(r['optimal']):.4f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = TABLES_DIR / "reopt_comparison.tex"
    out.write_text("\n".join(lines))
    print(f"  Generated: {out}")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating LaTeX tables:")
    generate_netlib_table()
    generate_reopt_table()


if __name__ == "__main__":
    main()
