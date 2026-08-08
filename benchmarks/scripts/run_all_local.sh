#!/usr/bin/env bash
# Track B full local benchmark re-run.
# Usage (from repo root):  bash benchmarks/scripts/run_all_local.sh
set -e

python - <<'PYCHECK'
import sys
v = sys.version_info
assert v >= (3, 10), (
    f"Python >=3.10 required, got {sys.version.split()[0]}. "
    "Activate the clara env first (see runbook).")
import clara
import highspy
print("environment ok:", sys.version.split()[0])
PYCHECK

python -m pytest tests/ -q

P=benchmarks/scripts
python "$P/download_netlib.py" --medium
python "$P/generate_random_lp.py"
python "$P/generate_perturbations.py" --all-random
python "$P/generate_perturbations.py" --all-netlib
python "$P/run_cross_validation.py"
python "$P/run_warmstart_benchmark.py"
python "$P/run_reopt_decision.py"
python "$P/run_skip_realized_loss.py"
python "$P/run_parametric_benchmark.py"
python "$P/run_attribution_benchmark.py"
python "$P/run_region_benchmark.py"
python "$P/collect_paper_data.py"
python "$P/run_final_tasks.py"
python "$P/exp_highs_warmstart.py"
python "$P/exp_medium_warmstart.py"
python "$P/run_oguz_threshold_validation.py"
python "$P/supply_chain_case_study.py"
python "$P/generate_all_figures.py"
python "$P/generate_paper_figures.py"
python "$P/generate_fig1_architecture.py"

echo ""
echo "ALL DONE — results in benchmarks/results/"
