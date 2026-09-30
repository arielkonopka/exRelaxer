#!/usr/bin/env bash
# The runs behind results/delayed-credit (research log §25). nntest --out
# appends: start from an empty directory.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=${1:-results/delayed-credit}
mkdir -p "$out"
run() { NNtesting/nntest.py run delayed_credit --trials 10 --threads 1 "$@"; }
delays=--set\ delay=0,1,2,4,8,16,32,64
{
  # Trace rule: trace decay x E-R recovery, and a plain linear neuron.
  run --set rule=trace --set neuron=er --set trace=0,0.5,0.8,0.9,0.95,0.99 --set recovery=0.9,0.97,0.99 $delays \
      --out "$out/trace_er.jsonl"
} > "$out/trace_er.log" 2>&1 &
{
  run --set rule=trace --set neuron=linear --set trace=0,0.5,0.8,0.9,0.95,0.99 $delays --out "$out/trace_linear.jsonl"
  # Sign rule: its eligibility is the E-R threshold, set by recovery.
  run --set rule=sign --set neuron=er --set recovery=0.9,0.97,0.99 $delays --out "$out/sign_er.jsonl"
  run --set rule=sign --set neuron=linear $delays --out "$out/sign_linear.jsonl"
  # Twice the training episodes: does the horizon move with more data?
  run --set rule=trace --set neuron=er --set trace=0.8,0.9,0.95 --set episodes=1600 $delays \
      --out "$out/trace_er_1600.jsonl"
} > "$out/other.log" 2>&1 &
wait
