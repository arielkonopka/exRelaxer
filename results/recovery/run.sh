#!/bin/sh
# Do the earlier E-R findings hold at the new default recovery (0.5, PR #48)?
# Every C++ experiment with E-R neurons, current defaults, same seeds, at
# recovery 0.9 (the default when they were run; reproduces the recorded runs
# exactly) and 0.5 (the default since 2026-10-08). Only the E-R models are
# run where an experiment compares several (recovery does not touch the others).
# Usage: sh results/recovery/run.sh OUT_DIR   (repository root, after ./build.sh)
set -e
OUT=${1:-results/recovery/raw}
mkdir -p "$OUT"
NT=${NTEST:-./build/NNtesting/nntest}
jobs="$OUT/jobs.txt"; : > "$jobs"
job() { echo "$*" >> "$jobs"; }   # job NAME ARGS...
for r in 0.9 0.5; do
  for e in er_economy er_paths er_fatigue er_history er_silence er_habituation; do
    job "${e}_rec$r" run $e --trials 20 --set recovery=$r
  done
  M="--set model=er --set lr=0.001,0.003,0.01,0.03 --set depth=1,2,3 --set width=4,8,16,32,64"
  for g in l0 l1 l2 l3 l4; do
    job "nl_static_${g}_rec$r" run nl_static --trials 5 --set task=$g $M --set recovery=$r
  done
  for g in t1 t2 t4 t3; do
    G="--trials 5 --set task=$g $M --set recovery=$r"; [ $g = t3 ] && G="$G --set n=4,8"
    job "nl_temporal_${g}_rec$r" run nl_temporal $G
  done
  job "dyn_ladder_rec$r" run dyn_ladder --trials 5 --set task=dir,change,vel,catch --set model=er --set recovery=$r
  job "snake_rules_rec$r" run snake_rules --trials 10 --set mix_er=true --set readout=sign,fa --set mix=frozen,fa --set recovery=$r
  job "bar_orientation_rec$r" run bar_orientation --trials 20 --set readout_er=true --set recovery=$r
done
while read -r name args; do
  echo "$NT $args --threads 1 --out $OUT/$name.jsonl > $OUT/$name.log 2>&1 || true"
done < "$jobs" | xargs -P "${JOBS:-4}" -I{} sh -c '{}'
echo done > "$OUT/DONE"
