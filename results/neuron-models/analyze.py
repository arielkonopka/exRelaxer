#!/usr/bin/env python3
"""Summarise results/neuron-models/run.sh as Markdown tables.

usage: analyze.py RAW_DIR > summary.md

Per task and model: the best median test result over depth {1, 2, 3} x
width {4 ... 64}, the learning rate chosen per architecture on median
validation; in brackets the smallest network (neurons) that solves the task
in >= 80 % of seeds; and the active fraction of hidden neurons of the best
architecture (a neuron counts as active on a tick when its output is not 0).
"""
import collections
import glob
import gzip
import json
import os
import statistics as st
import sys


def trials(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") == "trial":
                yield r


def median(rs, key):
    vals = [r["metrics"][key] for r in rs if key in r["metrics"]]
    return st.median(vals) if vals else None


def fmt(v):
    if v is None:
        return "–"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if v == 0:
        return "0"
    return f"{v:.3g}"


def label(p):
    m = p["model"]
    return m + ("~" + p["model_smoothing"] if p.get("model_smoothing", "0") != "0" else "")


def task_name(p):
    t = p["task"]
    return t + (f" k{p['k']}" if t == "l4" else "") + (f" n{p['n']}" if t == "t3" else "")


def table(title, rs, binary, note):
    groups = collections.defaultdict(list)
    for r in rs:
        p = r["params"]
        groups[(task_name(p), label(p), int(p["depth"]), int(p["width"]), p["lr"])].append(r)
    best = {}
    for (task, model, d, w, lr), g in groups.items():
        vkey = "best_validation" if binary(task) else "validation_mse"
        v = median(g, vkey)
        if v is None:
            vkey = "test_accuracy" if binary(task) else "test_mse"
            v = median(g, vkey)
        if v is None:
            continue
        k = (task, model, d, w)
        if k not in best or (v > best[k][0] if binary(task) else v < best[k][0]):
            best[k] = (v, g)
    tasks = sorted({k[0] for k in best})
    models = []
    for k in best:
        if k[1] not in models:
            models.append(k[1])
    order = ["relu", "er", "er_memoryless", "clamp", "sigmoid", "tanh", "softplus", "lif", "alif", "izhikevich",
             "resonator"]
    models.sort(key=lambda m: (order.index(m.split("~")[0]) if m.split("~")[0] in order else 99, m))
    print(f"\n### {title}\n\n{note}\n")
    print("| model | " + " | ".join(tasks) + " |")
    print("|---|" + "---|" * len(tasks))
    for model in models:
        cells = []
        for task in tasks:
            key = "test_accuracy" if binary(task) else "test_mse"
            rows = [(median(g, key), median(g, "success"), d * w, median(g, "active_fraction"))
                    for (tk, m, d, w), (v, g) in best.items() if tk == task and m == model]
            rows = [r for r in rows if r[0] is not None]
            if not rows:
                cells.append("–")
                continue
            rows.sort(key=lambda x: -x[0] if binary(task) else x[0])
            solved = [n for (_, s, n, _) in rows if s is not None and s >= 0.8]
            c = fmt(rows[0][0]) + (f" ({min(solved)})" if solved else "")
            if rows[0][3] is not None:
                c += f", act {rows[0][3]:.2f}"
            cells.append(c)
        print(f"| {model} | " + " | ".join(cells) + " |")


def main():
    raw = sys.argv[1]
    temporal = [r for f in sorted(glob.glob(os.path.join(raw, "nl_temporal_*.jsonl*"))) for r in trials(f)]
    static = [r for f in sorted(glob.glob(os.path.join(raw, "nl_static_*.jsonl*"))) for r in trials(f)]
    print("## Neuron models on the nonlinearity tasks\n")
    print("Medians over 5 seeds; lr in {0.001, 0.003, 0.01, 0.03} chosen per architecture on validation; "
          "feedback alignment with a learned bias, habituation off, default model parameters "
          "(LIF/ALIF leak 0.9, threshold 1; Izhikevich regular spiking, gain 10; resonator f 0.1, damping 0.9).")
    table("nl_temporal (test accuracy; t4: test MSE)", temporal, lambda t: not t.startswith("t4"),
          "Cell: best test result (smallest solving network, neurons; success = accuracy ≥ 0.95, MSE ≤ 0.001), "
          "active fraction of the best architecture. `~0.5`: rate units with smoothing 0.5.")
    table("nl_static (test MSE)", static, lambda t: False,
          "Cell: best test MSE (smallest network with MSE ≤ 1e-3 in ≥ 80 % of seeds, neurons), active fraction.")


if __name__ == "__main__":
    main()
