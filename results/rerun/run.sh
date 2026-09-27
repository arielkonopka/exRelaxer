#!/bin/sh
# Rerun of the E-R experiments with the current defaults (linear threshold
# growth) and three habituation variants:
#   off    no habituation
#   cut5   habituation cuts a repeated input after 5 identical ticks
#   fade2  habituation fades a repeated input by 0.9 per tick from its 2nd repeat
# Usage: sh results/rerun/run.sh OUT_DIR   (from the repository root, after ./build.sh)
set -e
OUT=${1:-results/rerun/raw}
mkdir -p "$OUT"
NN=./build/NNtesting/nntest
variant() {
  case $1 in
    off)   echo "--set habituation=false" ;;
    cut5)  echo "--set habituation=true --set habituation_steps=5 --set habituation_decay=0" ;;
    fade2) echo "--set habituation=true --set habituation_decay=0.9 --set habituation_fade_after=2" ;;
  esac
}
for v in off cut5 fade2; do
  H=$(variant $v)
  (
    for e in er_economy er_paths er_fatigue er_history er_silence er_habituation; do
      $NN run $e --threads 1 $H --out "$OUT/${e}_$v.jsonl" > "$OUT/${e}_$v.log" 2>&1 || true
    done
    $NN run nl_temporal --threads 1 --trials 5 $H --set task=t1,t2,t4 \
      --set model=relu,er,er_memoryless,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
      --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_temporal_$v.jsonl" > "$OUT/nl_temporal_$v.log" 2>&1 || true
    $NN run nl_temporal --threads 1 --trials 5 $H --set task=t3 --set n=4,8 \
      --set model=relu,er,er_memoryless,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
      --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_temporal_$v.jsonl" >> "$OUT/nl_temporal_$v.log" 2>&1 || true
  ) &
  (
    $NN run nl_static --threads 1 --trials 5 $H --set task=l0,l1,l2,l3 \
      --set model=relu,er,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
      --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_static_$v.jsonl" > "$OUT/nl_static_$v.log" 2>&1 || true
    $NN run nl_static --threads 1 --trials 5 $H --set task=l4 --set k=1,4,16 \
      --set model=relu,er,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
      --set depth=1,2,3 --set width=4,8,16,32,64 --out "$OUT/nl_static_$v.jsonl" >> "$OUT/nl_static_$v.log" 2>&1 || true
  ) &
done
wait
# The library's other experiments, current defaults (their own habituation settings).
for e in bar_orientation chirp_direction snake stereo_depth gapped_pattern audiovisual snake_rules; do
  $NN run $e --out "$OUT/$e.jsonl" > "$OUT/$e.log" 2>&1 || true
done
echo done > "$OUT/DONE"
