#!/bin/sh
# E-R threshold growth rules x learning on every tick, small architectures, 5 seeds, lr grid.
R=results/nonlinearity/growth
common="--threads 1 --trials 5 --set model=er --set growth=log,linear,fixed,multiplicative --set learn_ticks=last,all --set lr=0.001,0.003,0.01,0.03 --set depth=1,2 --set width=8,16,32,64"
./build-dev/NNtesting/nntest run nl_temporal $common --set task=t1,t2,t4 --out $R/temporal.jsonl > $R/temporal.log 2>&1
./build-dev/NNtesting/nntest run nl_temporal $common --set task=t3 --set n=4 --out $R/temporal.jsonl >> $R/temporal.log 2>&1
./build-dev/NNtesting/nntest run nl_static $common --set task=l1,l2 --set settle=7 --out $R/static.jsonl > $R/static.log 2>&1
echo done > $R/DONE
