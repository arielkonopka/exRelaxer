#!/bin/bash
# Doom width-growth arms (§40): defend_the_center, 1-minute games, evolve all, recovery 0.5, 300 generations.
cd /home/claude/exRelaxer
P=/home/claude/venv/bin/python
ES="NNtesting/experiments/doom_rl/es.py --sigma 0.05 --lr 0.02 --episodes 3 --validation 10 --generations ${GENS:-300} --workers 1"
C='"depth": 1, "feedback": "none", "evolve": "all", "recovery": 0.5'
O=/home/claude/runs/doom/es
$P $ES --out $O/fixed128 --config "{$C, \"width\": 128}" >> $O/fixed128.out 2>&1 &
$P $ES --out $O/fixed8 --config "{$C, \"width\": 8}" >> $O/fixed8.out 2>&1 &
$P $ES --out $O/grow1 --config "{$C, \"width\": 1}" --grow-width 128 >> $O/grow1.out 2>&1 &
$P $ES --out $O/grow1_young --config "{$C, \"width\": 1}" --grow-width 128 --young-resting 0.5 --young-gain 3 >> $O/grow1_young.out 2>&1 &
wait
