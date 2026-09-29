#!/bin/sh
# Did the 2026-09-28 defaults (normalised sums in the Dense / Conv2D /
# LocallyConnected2D builders, spontaneous amplitude 0.1, alpha 2.0) change
# the earlier results? Every C++ experiment except Doom, same seeds, with
# three builds of nntest:
#   OLD  commit 99164c0, the last one before the new defaults
#   NEW  this commit (the runs recorded in raw/ used its parent's experiment defaults;
#        the learning rates that changed are pinned below)
#   RAW  this commit with the two `s.normalize = true;` lines in
#        core/layers/layer_factory.hpp removed (builders give raw sums again)
# Usage: OLD=... RAW=... sh results/new-defaults/run.sh OUT_DIR   (repository root, after ./build.sh)
# Variants of the experiments that have the settings (activity, nl_*):
#   old        OLD build, its defaults (raw sums, amplitude 0.01)
#   norm_rn    NEW, hidden and readout normalised (main after PR #20)
#   norm       NEW, hidden normalised, readout raw (this commit's default)
#   raw        NEW, raw hidden and readout, amplitude 0.1
#   raw_a01    NEW, raw, amplitude 0.01 (differs from old only in alpha, which the linear rule ignores)
#   norm_lr10 / norm_rn_lr10 / raw_lr10   the same at 10x the learning rate (activity experiments)
set -e
OUT=${1:-results/new-defaults/raw}
mkdir -p "$OUT"
NEW=${NEW:-./build/NNtesting/nntest}
: "${OLD:?set OLD to the nntest of commit 99164c0}" "${RAW:?set RAW to the raw-builder nntest}"
jobs="$OUT/jobs.txt"; : > "$jobs"
job() { echo "$*" >> "$jobs"; }   # job NAME BINARY ARGS...
for e in er_economy er_paths er_fatigue er_history er_silence er_habituation; do
  T="--trials 20 --set lr=0.0003"   # the activity experiments' default when this was run (now 0.003)
  job "${e}_old" "$OLD" run $e $T
  job "${e}_norm_rn" "$NEW" run $e $T --set readout_normalize=true
  job "${e}_norm" "$NEW" run $e $T
  job "${e}_raw" "$NEW" run $e $T --set normalize=false
  job "${e}_raw_a01" "$NEW" run $e $T --set normalize=false --set spontaneous_amplitude=0.01
  T="--trials 20 --set lr=0.003"
  job "${e}_norm_lr10" "$NEW" run $e $T
  job "${e}_norm_rn_lr10" "$NEW" run $e $T --set readout_normalize=true
  job "${e}_raw_lr10" "$NEW" run $e $T --set normalize=false
done
S4="--trials 5 --set task=l4 --set k=1,4,16 --set model=relu,er,gate,clamp --set lr=0.001,0.003,0.01,0.03 --set depth=1,2,3 --set width=4,8,16,32,64"
M="--set model=relu,er,er_memoryless,gate,clamp --set lr=0.001,0.003,0.01,0.03 --set depth=1,2,3 --set width=4,8,16,32,64"
for g in l0 l1 l2 l3; do
  G="--trials 5 --set task=$g --set model=relu,er,gate,clamp --set lr=0.001,0.003,0.01,0.03 --set depth=1,2,3 --set width=4,8,16,32,64"
  job "nl_static_${g}_old" "$OLD" run nl_static $G
  job "nl_static_${g}_norm_rn" "$NEW" run nl_static $G --set readout_normalize=true
  job "nl_static_${g}_norm" "$NEW" run nl_static $G
  job "nl_static_${g}_raw" "$NEW" run nl_static $G --set normalize=false
done
job nl_static_l4_old "$OLD" run nl_static $S4
job nl_static_l4_norm_rn "$NEW" run nl_static $S4 --set readout_normalize=true
job nl_static_l4_norm "$NEW" run nl_static $S4
job nl_static_l4_raw "$NEW" run nl_static $S4 --set normalize=false
for g in t1 t2 t4 t3; do
  G="--trials 5 --set task=$g $M"; [ $g = t3 ] && G="$G --set n=4,8"
  job "nl_temporal_${g}_old" "$OLD" run nl_temporal $G
  job "nl_temporal_${g}_norm_rn" "$NEW" run nl_temporal $G --set readout_normalize=true
  job "nl_temporal_${g}_norm" "$NEW" run nl_temporal $G
  job "nl_temporal_${g}_raw" "$NEW" run nl_temporal $G --set normalize=false
done
# The other experiments have no settings for these: compare the builds, and
# the new build at 3x and 10x the learning rate.
for e in bar_orientation chirp_direction snake snake_rules stereo_depth gapped_pattern audiovisual dyn_ladder; do
  T="--trials 20"; [ $e = chirp_direction ] && T="--trials 10"
  [ $e = audiovisual ] && T="$T --set lr=0.01"   # its default when this was run (now 0.1)
  job "${e}_old" "$OLD" run $e $T
  job "${e}_new" "$NEW" run $e $T
  job "${e}_rawbuild" "$RAW" run $e $T
done
for e in bar_orientation:0.01 chirp_direction:0.01 snake:0.03 gapped_pattern:0.005 audiovisual:0.01; do
  n=${e%%:*}; lr=${e#*:}; T="--trials 20"; [ $n = chirp_direction ] && T="--trials 10"
  for f in 3 10; do
    job "${n}_new_lr$f" "$NEW" run $n $T --set lr=$(python3 -c "print($lr*$f)")
  done
done
# 4 at a time, one thread each
while read -r name bin args; do
  echo "$bin $args --threads 1 --out $OUT/$name.jsonl > $OUT/$name.log 2>&1 || true"
done < "$jobs" | xargs -P "${JOBS:-4}" -I{} sh -c '{}'
echo done > "$OUT/DONE"
