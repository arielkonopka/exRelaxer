#!/usr/bin/env bash
# 8-bit number formats (doc/precision.md, research log §32). From the repository root, after
# ./build.sh --python.
set -euo pipefail
export PYTHONPATH=build/EXrelaxer.py/package

# nl_static: E-R and ReLU hidden layers, four formats, four learning rates, 5 seeds.
build/NNtesting/nntest run nl_static --threads 1 --trials 5 --set task=l0,l1,l2 --set model=er,relu \
    --set precision=float,int8w,int8v,int8 --set lr=0.001,0.003,0.01,0.03 --set width=16,32 \
    --set early_stop=false --out nl_static.jsonl
python3 results/precision/nl_summary.py nl_static.jsonl > nl_static_summary.txt

# Snake, fixed ER(16) at recovery 0.5, 1500 games, 10 seeds; all8 = 8-bit readouts too.
for prec in float int8w int8v int8 all8; do
    for seed in $(seq 1 10); do
        if [ "$prec" = all8 ]; then a="--precision int8 --readout-precision int8"; else a="--precision $prec"; fi
        echo "$prec $seed $(python3 NNtesting/experiments/snake_growth/experiment.py --start 16 --grow False \
            --recovery 0.5 --games 1500 --seed "$seed" $a | grep '^{')"
    done
done > snake_er16.txt
