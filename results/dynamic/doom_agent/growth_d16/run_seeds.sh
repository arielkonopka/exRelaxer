#!/bin/bash
# D16 replication: seeds 2 and 3 (ES seed and network seed) of every arm.
cd /home/claude/exRelaxer
P=/home/claude/venv/bin/python
C='"depth": 1, "feedback": "none", "evolve": "all", "recovery": 0.5'
O=/home/claude/runs/doom/es
for s in 2 3; do
  ES="NNtesting/experiments/doom_rl/es.py --sigma 0.05 --lr 0.02 --episodes 3 --validation 10 --generations 300 --workers 1 --seed $s --net-seed $s"
  $P $ES --out $O/fixed128_s$s --config "{$C, \"width\": 128}" >> $O/fixed128_s$s.out 2>&1 &
  $P $ES --out $O/fixed8_s$s --config "{$C, \"width\": 8}" >> $O/fixed8_s$s.out 2>&1 &
  $P $ES --out $O/grow1_s$s --config "{$C, \"width\": 1}" --grow-width 128 >> $O/grow1_s$s.out 2>&1 &
  $P $ES --out $O/grow1_young_s$s --config "{$C, \"width\": 1}" --grow-width 128 --young-resting 0.5 --young-gain 3 >> $O/grow1_young_s$s.out 2>&1 &
done
wait
