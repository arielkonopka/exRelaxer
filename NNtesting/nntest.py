#!/usr/bin/env python3
"""Runs the Python experiments in NNtesting/experiments (see exrelaxer.harness).

    NNtesting/nntest.py list
    NNtesting/nntest.py run mnist_gabor --trials 3 --out results.jsonl

Needs the exrelaxer package: pip install ./EXrelaxer.py. The C++ experiments
run with build/NNtesting/nntest; both write the same result format.
"""
import sys
from pathlib import Path

try:
    from exrelaxer.harness import main
except ImportError as e:
    sys.exit(f"nntest.py: {e}; install the Python package first: pip install ./EXrelaxer.py")

if __name__ == "__main__":
    sys.exit(main(experiments_dir=Path(__file__).resolve().parent / "experiments"))
