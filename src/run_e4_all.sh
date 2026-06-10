#!/usr/bin/env bash
# All 18 E4 runs (6 arms x 3 seeds), sequential, resumable. Run on the local
# GPU from the repo root:
#   bash src/run_e4_all.sh
# A killed run loses at most one epoch; rerunning this script resumes.
# Each run prints its wall-clock estimate at startup. Commit results/e4/*.json
# afterwards; figures regenerate via: python src/fig_e4.py
set -e
for seed in 0 1 2; do
  for arm in baseline weight_decay label_smoothing logitnorm focal temperature; do
    python src/e4_fixes.py --config "configs/e4/${arm}.yaml" --seed "$seed" --out results/e4/
  done
done
