#!/usr/bin/env bash
# All 18 E4 runs (6 arms x 3 seeds), resumable. Run on the local GPU from the
# repo root:
#   bash src/run_e4_all.sh            # sequential
#   PARALLEL=2 bash src/run_e4_all.sh # two runs share the GPU (~1.5-1.8x
#                                     # throughput at ResNet-18/CIFAR scale)
# Data is cached on-device with GPU augmentation -- no dataloader workers,
# safe on Windows. A killed run loses at most one epoch; rerunning this
# script resumes (completed runs are skipped by their checkpoints in
# seconds). Each run prints its wall-clock estimate at startup. Commit
# results/e4/*.json afterwards; figures regenerate via: python src/fig_e4.py
set -e
PARALLEL="${PARALLEL:-1}"
for seed in 0 1 2; do
  for arm in baseline weight_decay label_smoothing logitnorm focal temperature; do
    echo "$arm $seed"
  done
done | xargs -n2 -P"$PARALLEL" sh -c \
  'python src/e4_fixes.py --config configs/e4/$0.yaml --seed $1 --out results/e4/'
