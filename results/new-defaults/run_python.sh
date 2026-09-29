#!/bin/sh
# The Python experiments (except Doom and video_memory*, which set normalize
# explicitly) with the Python package of commit 99164c0 (OLD_PY) and of this
# commit (NEW_PY), and the new one at 3x and 10x the learning rate.
# Usage: OLD_PY=<old build>/EXrelaxer.py/package NEW_PY=<new build>/EXrelaxer.py/package \
#        sh results/new-defaults/run_python.sh OUT_DIR    (needs: NNtesting/datasets/fetch.py get mnist)
set -e
OUT=${1:-results/new-defaults/raw}
mkdir -p "$OUT"
: "${OLD_PY:?}" "${NEW_PY:?}"
run() { name=$1; pkg=$2; shift 2; PYTHONPATH=$pkg python3 NNtesting/nntest.py run "$@" --out "$OUT/$name.jsonl" > "$OUT/$name.log" 2>&1 || true; }
for e in bar_orientation_py:0.01:10 snake_py:0.03:10 mnist_gabor:0.01:3; do
  n=${e%%:*}; r=${e#*:}; lr=${r%%:*}; t=${r#*:}
  run ${n}_old "$OLD_PY" $n --trials $t
  run ${n}_new "$NEW_PY" $n --trials $t
  for f in 3 10; do
    run ${n}_new_lr$f "$NEW_PY" $n --trials $t --set lr=$(python3 -c "print($lr*$f)")
  done
done
echo done > "$OUT/PY_DONE"
