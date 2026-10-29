#!/bin/bash
# G7 / research §40: G3's growing run resumed from its snapshot before the first growth
# (G3 grow/before_stage1, generation 280: copy its state.npz and frozen.npz into OUT first),
# older columns evolving and grown columns young, to 1200 generations as G3.
# Played on Gardens of Eris 8a3ea88 (G3's game): put that build's goe package on PYTHONPATH.
OUT=${1:-results/goe-grow/open_young}
python3 NNtesting/experiments/goe_rl/grow.py --generations 1200 --workers 4 --out "$OUT" \
    --config '{"directions": 4, "train_old": true, "young_resting": 0.5, "young_gain": 3.0}'
