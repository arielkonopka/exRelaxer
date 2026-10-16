#!/bin/sh
# Neuron models (doc/neuron_models.md) on the nonlinearity tasks: the same
# grid as results/rerun (habituation off), with the new models next to
# relu, er, er_memoryless and clamp.
# Usage: sh results/neuron-models/run.sh OUT_DIR   (from the repository root, after ./build.sh)
set -e
OUT=${1:-results/neuron-models/raw}
mkdir -p "$OUT"
NN=./build/NNtesting/nntest
GRID="--threads 1 --trials 5 --set lr=0.001,0.003,0.01,0.03 --set depth=1,2,3 --set width=4,8,16,32,64"
NEW=sigmoid,tanh,softplus,lif,alif,izhikevich,resonator
for t in t1 t2 t4; do
  $NN run nl_temporal $GRID --set task=$t --set model=relu,er,er_memoryless,clamp,$NEW \
    --out "$OUT/nl_temporal_$t.jsonl" > "$OUT/nl_temporal_$t.log" 2>&1 &
done
$NN run nl_temporal $GRID --set task=t3 --set n=4,8 --set model=relu,er,er_memoryless,clamp,$NEW \
  --out "$OUT/nl_temporal_t3.jsonl" > "$OUT/nl_temporal_t3.log" 2>&1 &
wait
# Leaky rate units (smoothing 0.5): memory from the output's own decay.
$NN run nl_temporal $GRID --set task=t1,t2,t4 --set model=sigmoid,tanh,softplus --set model_smoothing=0.5 \
  --out "$OUT/nl_temporal_smooth.jsonl" > "$OUT/nl_temporal_smooth.log" 2>&1 &
for t in l0 l1 l2 l3; do
  $NN run nl_static $GRID --set task=$t --set model=relu,er,clamp,$NEW \
    --out "$OUT/nl_static_$t.jsonl" > "$OUT/nl_static_$t.log" 2>&1 &
done
wait
$NN run nl_static $GRID --set task=l4 --set k=1,4,16 --set model=relu,er,clamp,$NEW \
  --out "$OUT/nl_static_l4.jsonl" > "$OUT/nl_static_l4.log" 2>&1
echo done > "$OUT/DONE"
