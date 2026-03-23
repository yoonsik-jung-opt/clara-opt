"""Run all MPC paper experiments end-to-end.

Usage: python benchmarks/scripts/run_all.py
Reproduces all experimental results from the paper.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = [
    ("Download Netlib instances", "download_netlib.py"),
    ("Convert MPS → LP", "convert_mps_to_lp.py"),
    ("Generate random LPs", "generate_random_lp.py"),
    ("Generate perturbations (Netlib)", "generate_perturbations.py --all-netlib"),
    ("Exp 1: Cross-validation", "run_cross_validation.py"),
    ("Exp 2: Decision quality", "run_reopt_decision.py"),
    ("Exp 3: Warm-start", "run_warmstart_benchmark.py"),
    ("Exp 4: Parametric LP", "run_parametric_benchmark.py"),
    ("Exp 5: Explanation coverage", "run_explanation_coverage.py"),
    ("Exp 6: Scalability", "run_scalability.py"),
    ("Reopt comparison (Albici)", "run_reopt_benchmark.py"),
    ("Netlib benchmark", "run_benchmark.py"),
    ("Generate tables", "generate_all_tables.py"),
    ("Generate figures", "generate_all_figures.py"),
]


def main():
    total_start = time.perf_counter()
    passed = 0
    failed = []

    print("=" * 60)
    print("CLARA MPC Paper — Full Experiment Suite")
    print("=" * 60)

    for i, (desc, script_args) in enumerate(STEPS, 1):
        parts = script_args.split()
        script = SCRIPTS_DIR / parts[0]
        args = parts[1:]

        print(f"\n[{i}/{len(STEPS)}] {desc}")
        print("-" * 40)

        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(script)] + args,
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=False,
            timeout=1800,  # 30 min per step
        )
        elapsed = time.perf_counter() - start

        if result.returncode == 0:
            print(f"  ✓ Done ({elapsed:.1f}s)")
            passed += 1
        else:
            print(f"  ✗ Failed (exit {result.returncode}, {elapsed:.1f}s)")
            failed.append(desc)

    total = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(f"Complete: {passed}/{len(STEPS)} steps ({total:.0f}s total)")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
