#!/bin/sh
# Recalibrated E-R thresholds for normalised sums (research log §19): the
# activity experiments with normalize=true, resting_threshold 0.2 or auto,
# at three learning rates, and without normalisation at the same rates.
# Usage: sh results/normalize/run_calibration.sh OUT_DIR   (repository root, after ./build.sh)
set -e
OUT=${1:-results/normalize/raw}
mkdir -p "$OUT"
NN=./build/NNtesting/nntest
for e in er_economy er_paths er_fatigue er_history er_silence er_habituation; do
  for lr in 0.0003 0.001 0.003; do
    $NN run $e --threads 1 --set habituation=false --set lr=$lr \
      --out "$OUT/cal_${e}_raw_$lr.jsonl" > /dev/null 2>&1 || true
    for r in 0.2 auto; do
      $NN run $e --threads 1 --set habituation=false --set normalize=true --set resting_threshold=$r --set lr=$lr \
        --out "$OUT/cal_${e}_norm_${r}_$lr.jsonl" > /dev/null 2>&1 || true
    done
  done
done
echo done > "$OUT/CAL_DONE"
