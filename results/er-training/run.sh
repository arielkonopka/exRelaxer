#!/bin/sh
# E-R training: pretrain without E-R and switch, vs E-R trained longer. Linear growth (default).
R=results/er-training
C=results/er-cycles
common="--threads 1 --trials 5 --set model=er --set pretrain_model=,clamp,relu,gate --set pretrain=20000 --set train=0,20000,80000 --set early_stop=false --set eval_every=5000 --set lr=0.001,0.003,0.01,0.03"
./build/NNtesting/nntest run nl_static $common --set task=l1,l2 --set settle=7 --set depth=1,2 --set width=16,32 --out $R/static.jsonl > $R/static.log 2>&1 &
./build/NNtesting/nntest run nl_temporal $common --set task=t1,t2 --set depth=1 --set width=16,64 --out $R/temporal.jsonl > $R/temporal.log 2>&1 &
# Activation cycles: spontaneous-firing settings, with and without recurrence, E-R only.
./build/NNtesting/nntest run er_silence --threads 1 --set models=er --set spontaneous_below=1e-10,0.001,0.05 --set spontaneous_amplitude=0.01,0.1,0.5,1 --set spontaneous_rate=0,0.01,0.05 --set recurrent=false,true --out $C/er_silence_spont.jsonl > $C/er_silence_spont.log 2>&1 &
# Habituation fading from the 2nd repeat vs from the 100th.
./build/NNtesting/nntest run er_habituation --threads 1 --set habituation=true --set habituation_decay=0.5,0.9,0.99 --set habituation_fade_after=2,5,100 --set hold_noise=0,0.001 --set habituation_tolerance=0,0.01 --out results/er-habituation/er_habituation_fade.jsonl > results/er-habituation/fade.log 2>&1 &
wait
echo done > $R/DONE
