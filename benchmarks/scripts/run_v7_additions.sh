#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "== 0. tests"
python -m pytest -q

echo "== 1. perturbation instances (regenerated only if missing)"
if [ ! -f benchmarks/instances/perturbations/rand_n10_m10_d3_s42/rand_n10_m10_d3_s42_R_small_s42.lp ]; then
  python benchmarks/scripts/generate_perturbations.py --all-random
fi

echo "== 2. decision experiments with the certified bound (machine-independent counts)"
python benchmarks/scripts/run_reopt_decision.py
python benchmarks/scripts/run_skip_realized_loss.py
python benchmarks/scripts/run_oguz_threshold_validation.py
python benchmarks/scripts/check_oguz_tightening.py

echo "== 3. routing-rule ablation (exp9)"
python benchmarks/scripts/run_method_ablation.py

echo "== 4. region cost versus simplex (exp10); m=2000 takes a few minutes"
python benchmarks/scripts/run_chebyshev_timing.py --max-m 2000 --reps 3

echo "== 5. certified-skip cost versus a warm-started solver call (exp12)"
python benchmarks/scripts/run_skip_cost.py --reps 3

echo "== 6. copy to results/final"
mkdir -p benchmarks/results/final
for f in exp2_reopt_decision exp2b_skip_realized_loss exp_oguz_threshold exp11_oguz_tightening_audit \
         exp9_method_ablation exp10_chebyshev_timing exp12_skip_cost; do
  cp "benchmarks/results/$f.csv" benchmarks/results/final/
done

echo "== 7. numbers for the %% TODO(Mac) spots"
python benchmarks/scripts/summarize_v7.py --final
