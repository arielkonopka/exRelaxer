#!/usr/bin/env python3
"""Tables for run_calibration.sh: medians per experiment, model, variant and learning rate.

usage: analyze_calibration.py DIR > calibration.md
"""
import json
import os
import statistics as st
import sys

METRICS = {
    "er_economy": ["accuracy", "active_fraction"],
    "er_paths": ["accuracy", "internal_switch_rate"],
    "er_fatigue": ["A_first_accuracy", "all_first_accuracy"],
    "er_history": ["all_accuracy", "all_pattern_change"],
    "er_silence": ["rest_accuracy", "after_accuracy"],
    "er_habituation": ["h4_accuracy_sum", "h500_accuracy_end"],
}
LRS = ["0.0003", "0.001", "0.003"]
VARIANTS = [("raw", "raw"), ("norm_0.2", "normalised, rest 0.2"), ("norm_auto", "normalised, rest auto")]


def trials(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in (json.loads(l) for l in f if l.strip()) if r.get("type") == "trial"]


def med(rs, k):
    v = [r["metrics"][k] for r in rs if k in r["metrics"]]
    return f"{st.median(v):.3g}" if v else "–"


d = sys.argv[1]
print("## Recalibrated thresholds: activity experiments by learning rate\n")
print("Medians over trials; columns are learning rates " + ", ".join(LRS) + ".")
for e, ms in METRICS.items():
    print(f"\n### {e}\n")
    print("| model | metric | variant | " + " | ".join(f"lr {lr}" for lr in LRS) + " |")
    print("|---|---|---|" + "---|" * len(LRS))
    for model in ["er", "gate", "linear"]:
        for m in ms:
            for v, label in VARIANTS:
                row = [med(trials(os.path.join(d, f"cal_{e}_{v}_{lr}.jsonl")), f"{model}_{m}") for lr in LRS]
                print(f"| {model} | {m} | {label} | " + " | ".join(row) + " |")
