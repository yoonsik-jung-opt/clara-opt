"""Generate all LaTeX tables from experiment results.

Usage: python benchmarks/scripts/generate_all_tables.py
Output: paper/tables/exp*.tex
"""

import csv
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR = Path(__file__).parent.parent.parent / "paper" / "tables"


def _read_csv(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _write_tex(filename, lines):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename
    out.write_text("\n".join(lines))
    print(f"  {out.name}")


def gen_exp1():
    """Cross-validation summary table."""
    rows = _read_csv("exp1_cross_validation.csv")
    if not rows:
        return
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cross-validation: Internal Simplex vs.\ HiGHS.}",
        r"\label{tab:cross-val}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Instance & $n$ & $m$ & Opt.\ Gap & Sol.\ Diff \\", r"\midrule",
    ]
    for r in rows[:15]:  # top 15
        lines.append(f"  {r['instance'][:20]} & {r['n_vars']} & {r['n_cons']} "
                     f"& {r['opt_gap']} & {r['sol_diff']} \\\\")
    if len(rows) > 15:
        lines.append(f"  \\multicolumn{{5}}{{c}}{{\\ldots ({len(rows)} total instances)}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex("exp1_cross_validation.tex", lines)


def gen_exp2():
    """Decision quality summary."""
    rows = _read_csv("exp2_reopt_decision.csv")
    if not rows:
        return
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"] == "True")
    by_type = {}
    for r in rows:
        t = r["change_type"]
        by_type.setdefault(t, {"total": 0, "correct": 0})
        by_type[t]["total"] += 1
        if r["correct"] == "True":
            by_type[t]["correct"] += 1

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Reoptimization decision accuracy by change type.}",
        r"\label{tab:decision-quality}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Change Type & Total & Correct & Accuracy \\", r"\midrule",
    ]
    for t, v in sorted(by_type.items()):
        acc = v["correct"] / max(v["total"], 1)
        lines.append(f"  {t} & {v['total']} & {v['correct']} & {acc:.1%} \\\\")
    lines.append(r"\midrule")
    lines.append(f"  Total & {total} & {correct} & {correct/max(total,1):.1%} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex("exp2_decision_quality.tex", lines)


def gen_exp3():
    """Warm-start pivot reduction by method."""
    rows = _read_csv("exp3_warmstart.csv")
    if not rows:
        return
    by_method = {}
    for r in rows:
        m = r["method_used"]
        by_method.setdefault(m, [])
        by_method[m].append(float(r["speedup"]))

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Warm-start speedup by reoptimization method.}",
        r"\label{tab:warmstart}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & Count & Mean Speedup & Max & Median \\", r"\midrule",
    ]
    import numpy as np
    for m, speeds in sorted(by_method.items()):
        a = np.array(speeds)
        lines.append(f"  {m.replace('_', r' ')} & {len(a)} & {a.mean():.1f}$\\times$ "
                     f"& {a.max():.1f}$\\times$ & {np.median(a):.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex("exp3_warmstart.tex", lines)


def gen_exp4():
    """Parametric LP breakpoint statistics."""
    rows = _read_csv("exp4_parametric.csv")
    if not rows:
        return
    import numpy as np
    bps = [int(r["num_breakpoints"]) for r in rows]
    matched = sum(1 for r in rows if r["optimal_match"] == "True")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Parametric LP results for Type RC compound changes.}",
        r"\label{tab:parametric}",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Metric & Value \\", r"\midrule",
        f"  Total instances & {len(rows)} \\\\",
        f"  Mean breakpoints & {np.mean(bps):.1f} \\\\",
        f"  Max breakpoints & {max(bps)} \\\\",
        f"  Zero breakpoints & {sum(1 for b in bps if b == 0)} \\\\",
        f"  Optimal match & {matched}/{len(rows)} \\\\",
    ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex("exp4_parametric.tex", lines)


def gen_exp6():
    """Scalability table."""
    rows = _read_csv("exp6_scalability.csv")
    if not rows:
        return
    import numpy as np
    by_size = {}
    for r in rows:
        k = (int(r["n_vars"]), int(r["n_cons"]))
        by_size.setdefault(k, []).append(r)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Scalability: solve time vs.\ problem size.}",
        r"\label{tab:scalability}",
        r"\begin{tabular}{rrrrrr}", r"\toprule",
        r"$n$ & $m$ & Internal (s) & HiGHS (s) & Iters & $B^{-1}$ (MB) \\", r"\midrule",
    ]
    for (n, m), rr in sorted(by_size.items()):
        ti = np.mean([float(r["time_internal"]) for r in rr])
        th = np.mean([float(r["time_highs"]) for r in rr])
        it = int(np.mean([int(r["iterations_internal"]) for r in rr]))
        mem = float(rr[0]["memory_binv_mb"])
        lines.append(f"  {n} & {m} & {ti:.3f} & {th:.3f} & {it} & {mem:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex("exp6_scalability.tex", lines)


def main():
    print("Generating LaTeX tables:")
    gen_exp1()
    gen_exp2()
    gen_exp3()
    gen_exp4()
    gen_exp6()
    print("Done.")


if __name__ == "__main__":
    main()
