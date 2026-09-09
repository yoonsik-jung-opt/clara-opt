"""Print the numbers that fill the machine-dependent spots of the v7
manuscript (marked %% TODO(Mac) in clara_ejor_trackB.tex), formatted as
LaTeX table rows where applicable.

Reads (from benchmarks/results/ unless --final is given, then from
benchmarks/results/final/):
  exp10_chebyshev_timing.csv   -> Table 'chebyshev_cost'
  exp9_method_ablation.csv     -> Table 'ablation' (speedup column) and
                                  the Type C run-time saving sentence
  exp12_skip_cost.csv          -> skip-cost sentence in Section 6.3
  exp3_warmstart.csv           -> headline warm-start numbers (check)

Usage: python benchmarks/scripts/summarize_v7.py [--final]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def fmt_time(sec):
    ms = sec * 1e3
    if ms < 100:
        return f"{ms:.1f}\\,ms"
    if ms < 1000:
        return f"{ms:.0f}\\,ms"
    return f"{sec:.1f}\\,s"


def chebyshev(d):
    p = d / "exp10_chebyshev_timing.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    rows = list(csv.DictReader(open(p)))
    print("\n=== Table chebyshev_cost (rows: n & m_aug & simplex & solve & region & ratio) ===")
    rand = [r for r in rows if r["family"] == "random"]
    sizes = sorted({(int(r["n"]), int(r["m"])) for r in rand})
    for n, m in sizes:
        rs = [r for r in rand if int(r["n"]) == n and int(r["m"]) == m]
        s = np.median([float(r["simplex_time"]) for r in rs])
        so = np.median([float(r["solve_time"]) for r in rs])
        ch = np.median([float(r["chebyshev_time"]) for r in rs])
        print(f"{n:,} & {int(rs[0]['m_aug']):,} & {fmt_time(s)} & {fmt_time(so)} & {fmt_time(ch)} & {ch/s:.1f} \\\\")
    net = [r for r in rows if r["family"] == "netlib"]
    if net:
        s = [float(r["simplex_time"]) for r in net]; so = [float(r["solve_time"]) for r in net]
        ch = [float(r["chebyshev_time"]) for r in net]
        ratio = [c / x for c, x in zip(ch, s)]
        ma = [int(r["m_aug"]) for r in net]
        print(f"Netlib ({len(net)}) & {min(ma)}--{max(ma)} & {min(s)*1e3:.1f}--{max(s)*1e3:.1f}\\,ms & "
              f"{min(so)*1e3:.1f}--{max(so)*1e3:.1f}\\,ms & {min(ch)*1e3:.1f}--{max(ch)*1e3:.1f}\\,ms & "
              f"{min(ratio):.1f}--{max(ratio):.1f} \\\\")


def ablation(d):
    p = d / "exp9_method_ablation.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    rows = [r for r in csv.DictReader(open(p)) if r.get("rule_match") == "True"]
    arms = [("rule", "Rule (feasibility-based)"), ("always_dual", "Always dual (HiGHS default)"),
            ("always_primal", "Always primal"), ("choose", "HiGHS choose"), ("cold", "Cold start")]
    print(f"\n=== Table ablation ({len(rows)} pairs; rows: arm & iters (median) & R & C & zero-iter & speedup) ===")
    cold = np.array([float(r["cold_time"]) for r in rows])
    for key, label in arms:
        it = np.array([int(r[f"{key}_iters"]) for r in rows])
        itR = np.mean([int(r[f"{key}_iters"]) for r in rows if r["change_type"] == "TYPE_R"])
        itC = np.mean([int(r[f"{key}_iters"]) for r in rows if r["change_type"] == "TYPE_C"])
        t = np.array([float(r[f"{key}_time"]) for r in rows])
        ok = (t > 0) & (cold > 0)
        sp = np.median(cold[ok] / t[ok])
        print(f"{label} & {it.mean():.2f} ({int(np.median(it))}) & {itR:.2f} & {itC:.2f} & "
              f"{np.mean(it == 0)*100:.1f}\\% & ${sp:.1f}\\times$ \\\\")
    rc = [r for r in rows if r["change_type"] == "TYPE_C"]
    t_rule = np.array([float(r["rule_time"]) for r in rc]); t_dual = np.array([float(r["always_dual_time"]) for r in rc])
    ok = (t_rule > 0) & (t_dual > 0)
    print(f"Type C run time: rule vs always-dual median ratio {np.median(t_dual[ok]/t_rule[ok]):.2f} "
          f"(saving {100*(1-1/np.median(t_dual[ok]/t_rule[ok])):.0f}%)")
    for a in ("choose", "always_dual", "always_primal"):
        dlt = np.array([int(r["rule_iters"]) - int(r[f"{a}_iters"]) for r in rows])
        print(f"  rule vs {a}: fewer iters on {np.sum(dlt<0)}, equal {np.sum(dlt==0)}, more {np.sum(dlt>0)}")


def skip_cost(d):
    p = d / "exp12_skip_cost.csv"
    if not p.exists():
        print(f"[skip] {p} not found"); return
    rows = list(csv.DictReader(open(p)))
    rec = np.array([float(r["recompute_time"]) for r in rows]) * 1e3
    warm = np.array([float(r["warm_time"]) for r in rows]) * 1e3
    cold = np.array([float(r["cold_time"]) for r in rows]) * 1e3
    sizes = np.array([int(r["n_cons"]) for r in rows])
    print(f"\n=== Skip cost sentence ({len(rows)} certified recomputes) ===")
    print(f"median: recompute {np.median(rec):.2f} ms | warm-started call {np.median(warm):.2f} ms | "
          f"cold {np.median(cold):.2f} ms | ratio warm/recompute {np.median(warm/rec):.1f}x")
    k = sizes == sizes.max()
    print(f"at m={sizes.max()}: ratio {np.median(warm[k]/rec[k]):.1f}x")


def warmstart(d):
    p = d / "exp3_warmstart.csv"
    if not p.exists():
        return
    rows = [r for r in csv.DictReader(open(p)) if r.get("method_used") != "scratch"]
    tw = np.array([float(r["run_time_warmstart"]) for r in rows]); tc = np.array([float(r["run_time_scratch"]) for r in rows])
    ok = (tw > 0) & (tc > 0)
    print(f"\n=== exp3 check: {len(rows)} pairs, run-time speedup mean {np.mean(tc[ok]/tw[ok]):.2f} median {np.median(tc[ok]/tw[ok]):.2f} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    d = ROOT / "benchmarks" / "results" / ("final" if args.final else "")
    chebyshev(d); ablation(d); skip_cost(d); warmstart(d)


if __name__ == "__main__":
    main()
