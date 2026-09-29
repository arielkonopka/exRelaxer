#!/usr/bin/env bash
# The runs behind results/video-memory-learned (research log §22; gzipped there).
# Four jobs in parallel, each appending to its own files (nntest --out
# appends: start from an empty directory). Lists inside one parameter
# (traces, lrs) use ';', since ',' makes a grid.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # four jobs on four cores
out=${1:-results/video-memory-learned}
mkdir -p "$out"
run() { NNtesting/nntest.py run video_memory_learned --threads 1 "$@"; }
gaps=--set\ gap=1,2,4,8,16
{
  # Primary: the video_memory baseline (recovery 0.9), E-R and the two non-E-R controls.
  run --trials 20 --set model=E0 $gaps --out "$out/primary_E0.jsonl"
  run --trials 20 --set model=C1 $gaps --out "$out/primary_C1.jsonl"
  # The readout on raw hidden outputs (no input gain).
  run --trials 20 --set model=E0 --set recovery=0.9,0.99 --set 'traces=0;0.8;0.99' --set input_gain=none $gaps \
      --out "$out/raw.jsonl"
} > "$out/job1.log" 2>&1 &
{
  run --trials 20 --set model=C0 $gaps --out "$out/primary_C0.jsonl"
  # Separate study: recovery (memory), with the same learning comparison.
  run --trials 20 --set model=E0 --set recovery=0.95 $gaps --out "$out/recovery.jsonl"
  # Immediate learning with an error at every readout tick (Python loop per tick: fewer seeds and rates).
  run --trials 10 --set model=E0 --set recovery=0.9 --set traces=0 --set error_at=tick --set 'lrs=0.0003;0.001;0.003' \
      $gaps --out "$out/tick.jsonl"
} > "$out/job2.log" 2>&1 &
{
  run --trials 20 --set model=E0 --set recovery=0.97,0.99 $gaps --out "$out/recovery.jsonl.2"
  run --trials 10 --set model=E0 --set recovery=0.99 --set traces=0 --set error_at=tick --set 'lrs=0.0003;0.001;0.003' \
      $gaps --out "$out/tick.jsonl.2"
} > "$out/job3.log" 2>&1 &
{
  # Separate experiment (condition D): the hidden layer learns too, with its own trace.
  run --trials 10 --set model=E0 --set mode=hidden --set recovery=0.9,0.99 --set hidden_trace=0,0.9,0.99 \
      --set hidden_lr=0.003,0.01 --set 'traces=0;0.8' --set gap=1,4,16 --set credit=false --out "$out/hidden.jsonl"
  # Per-neuron credit diagnostics of one configuration (npz per seed).
  run --trials 5 --set model=E0 --set recovery=0.99 --set gap=4 --set dump="$out/credit" --out "$out/credit.jsonl"
} > "$out/job4.log" 2>&1 &
wait
for f in recovery tick; do cat "$out/$f.jsonl.2" >> "$out/$f.jsonl" && rm "$out/$f.jsonl.2"; done
