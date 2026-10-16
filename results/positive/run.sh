#!/usr/bin/env bash
# The positive-only grid behind doc/research.md §33 (about 4 minutes).
set -euo pipefail
cd "$(dirname "$0")/../.."
out=results/positive/snake_rules.jsonl
rm -f "$out"
build/NNtesting/nntest run snake_rules --trials 10 \
    --set readout=sign,trace,fa,perturbation --set mix=frozen,sign,trace,fa,perturbation,oja,bcm \
    --set mix_er=false,true --set positive=none,all,mix,readouts --set positive_kind=both,values,weights \
    --out "$out"
python3 NNtesting/tools/positive_summary.py "$out" > results/positive/summary.md
gzip -f "$out"
