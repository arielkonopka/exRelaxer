#!/usr/bin/env python3
"""Normalised weighted sum vs the same grids without it (results/rerun, variant off).

usage: analyze.py NORMALIZE_DIR [RERUN_RAW_DIR] > summary.md
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rerun"))
import analyze as rerun  # noqa: E402

ER = rerun.ER_METRICS


def main():
    new = sys.argv[1]
    old = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rerun", "raw")
    print("## Normalised weighted sum (hidden layers), habituation off\n")
    print("Medians over trials; \"raw\" is the same run without normalisation (results/rerun, variant off).")
    for exp, metrics in ER.items():
        a = rerun.trials(os.path.join(old, f"{exp}_off.jsonl.gz"))
        b = rerun.trials(os.path.join(new, f"{exp}.jsonl"))
        print(f"\n### {exp}\n")
        print("| model | metric | raw | normalised |")
        print("|---|---|---|---|")
        for model in ["er", "gate", "linear"]:
            for m in metrics:
                k = f"{model}_{m}"
                print(f"| {model} | {m} | {rerun.fmt(rerun.median(a, k))} | {rerun.fmt(rerun.median(b, k))} |")
    for name, files, binary, models in [
        ("nl_static", ["nl_static.jsonl", "nl_static_l4.jsonl"], lambda t: False, ["relu", "er", "gate", "clamp"]),
        ("nl_temporal", ["nl_temporal.jsonl"], lambda t: not t.startswith("t4"),
         ["relu", "er", "er_memoryless", "gate", "clamp"]),
    ]:
        runs = {"off": rerun.trials(os.path.join(old, f"{name}_off.jsonl.gz")), "normalised": []}
        for f in files:
            runs["normalised"] += rerun.trials(os.path.join(new, f))
        rerun.VARIANTS = ["off", "normalised"]
        rerun.nl_table(name, runs, [], binary, models, None)


if __name__ == "__main__":
    main()
