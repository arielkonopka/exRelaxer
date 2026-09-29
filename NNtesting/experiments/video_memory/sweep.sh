#!/usr/bin/env bash
# The runs behind results/video-memory (research log §21; gzipped there). Four jobs in
# parallel, each appending to its own file (nntest --out appends: start
# from an empty directory).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # four jobs on four cores
out=${1:-results/video-memory}
mkdir -p "$out"
run() { NNtesting/nntest.py run video_memory --trials 20 --threads 1 "$@"; }
gaps=--set\ gap=1,2,4,8,16
{
  # Primary: side task, the six required conditions plus the matched window.
  run --set task=side --set model=C0,C1,E0,R1,R4,E1,Rg $gaps --out "$out/side.jsonl"
} > "$out/side.log" 2>&1 &
{
  # Recurrence comparison (after the primary), and the recovery sensitivity study.
  run --set task=side --set model=D2,D2R $gaps --out "$out/side_recurrence.jsonl"
  run --set task=side --set model=E0 --set recovery=0.95,0.97,0.99 $gaps --out "$out/side_recovery.jsonl"
} > "$out/side_extra.log" 2>&1 &
{
  # Order task: every single frame is the same for both classes.
  run --set task=order --set model=C0,C1,E0,R1,R4,E1,Rg,D2,D2R $gaps --out "$out/order.jsonl"
} > "$out/order.log" 2>&1 &
{
  # Separate experiment: the hidden layer trained by feedback alignment first.
  NNtesting/nntest.py run video_memory --trials 10 --threads 1 --set task=side --set model=E0 \
      --set hidden_rule=fa --set fa_lr=0.003,0.03 --set fa_epochs=10 $gaps --out "$out/side_fa.jsonl"
} > "$out/side_fa.log" 2>&1 &
wait
