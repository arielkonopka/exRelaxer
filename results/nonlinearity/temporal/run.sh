#!/bin/sh
# nl_temporal exploratory grid: 5 seeds, same lr grid for every model.
R=results/nonlinearity/temporal
job() { name=$1; shift
  ./build/NNtesting/nntest run nl_temporal --threads 1 --trials 5 "$@" \
    --set model=relu,er,er_memoryless,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
    --set depth=1,2,3,4,6,8 --set width=4,8,16,32,64,128 --out $R/$name.jsonl > $R/$name.log 2>&1; }
job t1 --set task=t1 &
job t2 --set task=t2 &
job t3n4 --set task=t3 --set n=4 &
job t4 --set task=t4 &
wait
job t3n8 --set task=t3 --set n=8 &
job t3n16 --set task=t3 --set n=16 &
wait
echo done > $R/DONE
