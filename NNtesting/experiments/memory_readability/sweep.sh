#!/usr/bin/env bash
# The runs behind results/memory-readability (research log §25). nntest --out
# appends: start from an empty directory. The files in results/ are these, gzipped.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=${1:-results/memory-readability}
mkdir -p "$out"
run() { NNtesting/nntest.py run memory_readability --trials 20 --threads 1 "$@"; }
blanks=--set\ blank=0,1,2,4,8,16,32,64,128
{
  run --set model=er --set recovery=0.9,0.99 $blanks --out "$out/er.jsonl"
} > "$out/er.log" 2>&1 &
{
  run --set model=reset,relu $blanks --out "$out/controls.jsonl"
  # Without input noise nothing can overwrite the stored state.
  run --set model=er --set noise=0 --set recovery=0.9,0.99 $blanks --out "$out/er_noiseless.jsonl"
} > "$out/other.log" 2>&1 &
wait
