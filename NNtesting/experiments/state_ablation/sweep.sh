#!/usr/bin/env bash
# The runs behind results/state-ablation (research log §27). nntest --out
# appends: start from an empty directory.
#
# 1. Tune the readout learning rate once, on tuning seeds 1000-1002 (never
#    used for testing), at gap 4, on validation clips: the rate with the best
#    validation balanced accuracy averaged over all six configurations and
#    both recoveries. (The output-only baseline alone is at chance, so it
#    cannot pick a rate; the average favours no configuration.)
# 2. Every configuration, gap and recovery with that one rate, seeds 0-9.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=${1:-results/state-ablation}
mkdir -p "$out"
run() { NNtesting/nntest.py run state_ablation --threads 1 "$@"; }

run --seed 1000 --trials 3 --set gap=4 --set recovery=0.9,0.99 --set lr=0.0001,0.0003,0.001,0.003,0.01 \
    --out "$out/tune.jsonl" > "$out/tune.log" 2>&1
lr=$(python3 - "$out/tune.jsonl" <<'PY'
import json, sys
from collections import defaultdict
score = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("type") == "trial" and not r.get("error"):
        score[r["params"]["lr"]] += [r["metrics"][f"{c}_val_balanced"] for c in "ABCDEF"]
print(max(score, key=lambda k: sum(score[k]) / len(score[k])))
PY
)
echo "tuned learning rate: $lr" | tee "$out/tuned_lr.txt"

for rec in 0.9 0.99; do
  run --seed 0 --trials 10 --set recovery=$rec --set gap=0,1,2,4,8,16,32,64 --set lr="$lr" \
      --out "$out/ablation_$rec.jsonl" > "$out/ablation_$rec.log" 2>&1 &
done
wait
