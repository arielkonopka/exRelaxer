#!/bin/sh
# Normalised weighted sum (LayerSpec::normalize) in the hidden layers, habituation
# off, current defaults otherwise. Compared with results/rerun (variant "off"),
# which ran the same grids without normalisation.
# Usage: sh results/normalize/run.sh OUT_DIR   (from the repository root, after ./build.sh)
set -e
OUT=${1:-results/normalize/raw}
mkdir -p "$OUT"
NN=./build/NNtesting/nntest
N="--set normalize=true --set habituation=false"
(
  for e in er_economy er_paths er_fatigue er_history er_silence er_habituation; do
    $NN run $e --threads 1 $N --out "$OUT/${e}.jsonl" > "$OUT/${e}.log" 2>&1 || true
  done
) &
(
  $NN run nl_temporal --threads 1 --trials 5 $N --set task=t1,t2,t4 \
    --set model=relu,er,er_memoryless,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
    --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_temporal.jsonl" > "$OUT/nl_temporal.log" 2>&1 || true
  $NN run nl_temporal --threads 1 --trials 5 $N --set task=t3 --set n=4,8 \
    --set model=relu,er,er_memoryless,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
    --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_temporal.jsonl" >> "$OUT/nl_temporal.log" 2>&1 || true
) &
(
  $NN run nl_static --threads 1 --trials 5 $N --set task=l0,l1,l2,l3 \
    --set model=relu,er,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
    --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_static.jsonl" > "$OUT/nl_static.log" 2>&1 || true
) &
(
  $NN run nl_static --threads 1 --trials 5 $N --set task=l4 --set k=1,4,16 \
    --set model=relu,er,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
    --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_static_l4.jsonl" > "$OUT/nl_static_l4.log" 2>&1 || true
) &
wait
echo done > "$OUT/DONE"
