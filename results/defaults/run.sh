#!/usr/bin/env bash
# Research log §21: spontaneous amplitude vs recovery after silence.
set -euo pipefail
nntest=${NNTEST:-build/NNtesting/nntest}
$nntest run er_silence --set models=er --set spontaneous_amplitude=0.01,0.03,0.1,0.3,1.0 \
    --set silence=300,3000 --set normalize=false,true --out silence_amplitude.jsonl
$nntest run er_silence --set models=er --set spontaneous_amplitude=0.01,0.1,1.0 --set silence=3000 \
    --set habituation=true --set recurrent=false,true --set growth=linear,log --out silence_amplitude_log.jsonl
