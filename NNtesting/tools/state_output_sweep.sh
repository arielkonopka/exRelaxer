#!/usr/bin/env bash
# State as output (research log §26): the controlled benchmarks with and
# without a State layer (the thresholds and habituation streaks of a hidden
# layer as outputs a downstream layer reads; doc/model.md#state-as-output).
# nntest --out appends: start from an empty directory.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=${1:-results/state-output}
mkdir -p "$out"
py() { NNtesting/nntest.py run "$@" --threads 1; }
cpp() { build/NNtesting/nntest run "$@"; }
{
  # memory_readability: readouts on outputs, thresholds, streaks and all three (read through the State layer).
  py memory_readability --trials 20 --set model=er --set recovery=0.9,0.99 --set habituation=0,1 --set hab_steps=4 \
     --set hab_tolerance=0.5 --set blank=0,1,2,4,8,16,32,64,128 --out "$out/memory_readability.jsonl"
} > "$out/memory_readability.log" 2>&1 &
{
  # delayed_credit: the learner reads the inputs, or the inputs and E-R relay thresholds.
  py delayed_credit --trials 10 --set rule=sign,trace --set trace=0.9 --set neuron=er --set state=none,tap \
     --set relay_recovery=0.9,0.99 --set delay=0,1,2,4,8,16,32,64 --out "$out/delayed_credit.jsonl"
} > "$out/delayed_credit.log" 2>&1 &
{
  # video_memory_learned: online readouts on the top layer's outputs, or its outputs + thresholds + streaks.
  py video_memory_learned --trials 20 --set model=E0 --set recovery=0.9,0.99 --set gap=1,2,4,8,16 \
     --set readout_inputs=output,output_state --set 'traces=0;0.8;0.99' --set credit=false \
     --out "$out/video_memory_learned.jsonl"
} > "$out/video_memory_learned.log" 2>&1 &
{
  # nl_temporal and dyn_ladder: a readout trained end to end (feedback alignment) with or without the State layer.
  cpp nl_temporal --trials 10 --set task=t1,t2,t3,t4 --set model=er --set habituation=false,true \
      --set habituation_tolerance=0.1 --set state_readout=false,true --set lr=0.001,0.003,0.01 \
      --out "$out/nl_temporal.jsonl"
  cpp dyn_ladder --trials 10 --set task=dir,change,vel,catch --set model=er --set state_readout=false,true \
      --set lr=0.001,0.003,0.01 --out "$out/dyn_ladder.jsonl"
} > "$out/cpp.log" 2>&1 &
wait
