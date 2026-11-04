#!/bin/bash
# Research §69: every arm of results/td-growth, 20 seeds each (4 runs in parallel).
# Each run appends one JSON line to OUT/<arm>.jsonl; summary.md was made from those lines.
set -e
cd "$(dirname "$0")/../.."
S=NNtesting/experiments/snake_growth/td_growth.py
OUT=${OUT:-td-growth-runs}
mkdir -p "$OUT"
C="--set games=3000 --set max_width=64"
arms=(
  "fixed1:$C --set start=1 --set grow=false"
  "fixed4:$C --set start=4 --set grow=false"
  "fixed16:$C --set start=16 --set grow=false"
  "fixed32:$C --set start=32 --set grow=false"
  "fixed64:$C --set start=64 --set grow=false"
  "tired:$C --set start=1 --set trigger=saturation"
  "td_rand:$C --set start=1"
  "timer:$C --set start=1 --set trigger=timer"
  "td_cand:$C --set start=1 --set candidates=8"
  "td_cand0:$C --set start=1 --set candidates=8 --set candidate_rate=0"
  "td_cand_sig:$C --set start=1 --set candidates=8 --set candidate_rate=0 --set significance=true --set memory=80"
  "timer_sig:$C --set start=1 --set trigger=timer --set significance=true"
  "over64_sig:$C --set start=64 --set trigger=none --set significance=true --set minimum=1"
  "over64_hard:$C --set start=64 --set trigger=none --set significance=true --set minimum=1 --set sig_fraction=0.7 --set sig_every=1000"
  "over64_harder:$C --set start=64 --set trigger=none --set significance=true --set minimum=1 --set sig_fraction=1.0 --set sig_every=500 --set sig_tolerance=0.2"
  "sat1_l3:$C --set start=1 --set trigger=none --set saturation=true --set weight_limit=3"
  "sat16_l3:$C --set start=16 --set trigger=none --set saturation=true --set weight_limit=3"
  "td_sat_l3:$C --set start=1 --set saturation=true --set weight_limit=3"
  "sat1_l15:$C --set start=1 --set trigger=none --set saturation=true --set weight_limit=1.5"
  "sat4_l15:$C --set start=4 --set trigger=none --set saturation=true --set weight_limit=1.5"
)
for a in "${arms[@]}"; do n=${a%%:*}; o=${a#*:}
  for s in $(seq 1 20); do echo "python3 $S --quiet --seeds $s --arm $n --out $OUT/$n.jsonl $o"; done
done | xargs -P 4 -I{} bash -c "{}"
