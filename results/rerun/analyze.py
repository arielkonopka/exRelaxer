#!/usr/bin/env python3
"""Summarise the rerun (results/rerun/run.sh) as Markdown tables.

usage: analyze.py RERUN_DIR [--before RESULTS_DIR] > summary.md

Each table shows the median over trials of selected metrics per model and
habituation variant (off, cut5, fade2). With --before, a "log" column adds
the earlier runs (log threshold growth, no habituation) from RESULTS_DIR
(the repository's results/ folder), restricted to the same architectures.
"""
import argparse
import collections
import gzip
import json
import os
import statistics as st

VARIANTS = ["off", "cut5", "fade2"]


def trials(path):
    if not os.path.exists(path):
        return []
    opener = gzip.open if path.endswith(".gz") else open
    out = []
    with opener(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") == "trial":
                out.append(r)
    return out


def median(rs, key):
    vals = [r["metrics"][key] for r in rs if key in r["metrics"]]
    return st.median(vals) if vals else None


def fmt(v):
    if v is None:
        return "–"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1 or v == 0:
        return f"{v:.2f}".rstrip("0").rstrip(".") if v != 0 else "0"
    return f"{v:.3g}"


ER_METRICS = {
    "er_economy": ["accuracy", "active_fraction", "spikes_per_inference", "accuracy_per_100_spikes"],
    "er_paths": ["accuracy", "active_fraction", "switch_rate", "internal_switch_rate"],
    "er_fatigue": ["rest_first_accuracy", "A_first_accuracy", "A_first_share", "A_recovery_ticks", "all_first_accuracy"],
    "er_history": ["rest_accuracy", "busy_pattern_change", "all_pattern_change", "all_decision_change", "all_accuracy"],
    "er_silence": ["rest_accuracy", "after_accuracy", "silence_active_w4", "self_active_ticks", "end_threshold_max"],
    "er_habituation": ["h4_accuracy_sum", "h500_spikes_per_sample", "h500_accuracy_sum", "h500_accuracy_end",
                       "h500_accuracy_per_100_spikes"],
}
BEFORE = {
    "er_economy": "er-activity/er_economy.jsonl.gz",
    "er_paths": "er-activity/er_paths.jsonl.gz",
    "er_fatigue": "er-activity/er_fatigue.jsonl.gz",
    "er_history": "er-activity/er_history.jsonl.gz",
    "er_silence": "er-silence/er_silence.jsonl.gz",
    "er_habituation": "er-habituation/er_habituation.jsonl.gz",
}


def default_params(r):
    """Earlier runs swept some parameters; keep the default configuration."""
    p = r["params"]
    return (p.get("habituation", "false") == "false" and p.get("recovery", "0.9") == "0.9"
            and p.get("recurrent", "false") == "false" and p.get("learning", "fa") == "fa"
            and p.get("online", "false") == "false")


def er_tables(rerun, before):
    for exp, metrics in ER_METRICS.items():
        runs = {v: trials(os.path.join(rerun, f"{exp}_{v}.jsonl")) for v in VARIANTS}
        old = [r for r in trials(os.path.join(before, BEFORE[exp]))] if before else []
        old = [r for r in old if default_params(r)]
        print(f"\n### {exp}\n")
        cols = (["log, off"] if old else []) + VARIANTS
        print("| model | metric | " + " | ".join(cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for model in ["er", "gate", "linear"]:
            for m in metrics:
                key = f"{model}_{m}"
                row = ([fmt(median(old, key))] if old else []) + [fmt(median(runs[v], key)) for v in VARIANTS]
                print(f"| {model} | {m} | " + " | ".join(row) + " |")


def best_per_arch(rs, binary):
    """lr chosen per (task, model, depth, width) on median validation."""
    groups = collections.defaultdict(list)
    for r in rs:
        p = r["params"]
        task = p["task"] + (f" k{p['k']}" if p["task"] == "l4" else "") + (f" n{p['n']}" if p["task"] == "t3" else "")
        groups[(task, p["model"], int(p["depth"]), int(p["width"]), p["lr"])].append(r)
    best = {}
    for (task, model, d, w, lr), g in groups.items():
        m = g[0]["metrics"]
        vkey = next((k for k in ("best_validation", "best_validation_mse", "validation_mse", "test_accuracy", "test_mse")
                     if k in m), None)
        v = median(g, vkey) if vkey else None
        if v is None:
            continue
        k = (task, model, d, w)
        if k not in best or (v > best[k][0] if binary(task) else v < best[k][0]):
            best[k] = (v, g)
    return best


def nl_table(name, runs, old, binary, models, archs):
    tkey = lambda task: "test_accuracy" if binary(task) else "test_mse"
    print(f"\n### {name}\n")
    print("Best median test result over depth {1, 2, 3} × width {4 … 64} (accuracy for binary tasks, MSE otherwise); "
          "in brackets the smallest network solving it in ≥ 80 % of seeds (neurons).\n")
    tables = {v: best_per_arch(runs[v], binary) for v in VARIANTS}
    oldb = best_per_arch([r for r in old if int(r["params"]["depth"]) <= 3 and int(r["params"]["width"]) <= 64],
                         binary) if old else {}
    cols = (["log, off"] if old else []) + VARIANTS
    print("| task | model | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    tasks = sorted({k[0] for t in tables.values() for k in t})

    def cell(b, task, model):
        rows = [(median(g, tkey(task)), median(g, "success"), d * w)
                for (tk, m, d, w), (v, g) in b.items() if tk == task and m == model]
        rows = [r for r in rows if r[0] is not None]
        if not rows:
            return "–"
        rows.sort(key=lambda x: -x[0] if binary(task) else x[0])
        solved = [n for (_, s, n) in rows if s is not None and s >= 0.8]
        return fmt(rows[0][0]) + (f" ({min(solved)})" if solved else "")

    for task in tasks:
        for model in models:
            row = ([cell(oldb, task, model)] if old else []) + [cell(tables[v], task, model) for v in VARIANTS]
            print(f"| {task} | {model} | " + " | ".join(row) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rerun")
    ap.add_argument("--before", default="")
    a = ap.parse_args()
    print("## Rerun: linear threshold growth, habituation off / cut after 5 / fade 0.9 after 2\n")
    print("Medians over trials. \"log, off\": the earlier runs (log growth, no habituation), where available.")
    er_tables(a.rerun, a.before)
    runs = {v: trials(os.path.join(a.rerun, f"nl_static_{v}.jsonl")) for v in VARIANTS}
    old = []
    if a.before:
        d = os.path.join(a.before, "nonlinearity/explore")
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".jsonl.gz"):
                old += [r for r in trials(os.path.join(d, f))
                        if r["params"]["task"] != "l4" or r["params"]["k"] in ("1", "4", "16")]
    nl_table("nl_static", runs, old, lambda t: False, ["relu", "er", "gate", "clamp"], None)
    runs = {v: trials(os.path.join(a.rerun, f"nl_temporal_{v}.jsonl")) for v in VARIANTS}
    old = []
    if a.before:
        d = os.path.join(a.before, "nonlinearity/temporal")
        for f in ["t1", "t2", "t4", "t3n4", "t3n8"]:
            old += trials(os.path.join(d, f + ".jsonl.gz"))
    nl_table("nl_temporal", runs, old, lambda t: not t.startswith("t4"),
             ["relu", "er", "er_memoryless", "gate", "clamp"], None)
    others = ["bar_orientation", "chirp_direction", "snake", "stereo_depth", "gapped_pattern", "audiovisual",
              "snake_rules"]
    print("\n### Other experiments (current defaults, their own habituation settings)\n")
    print("| experiment | metric | median |")
    print("|---|---|---|")
    for e in others:
        rs = trials(os.path.join(a.rerun, f"{e}.jsonl"))
        if not rs:
            continue
        keys = sorted(k for k in rs[0]["metrics"] if any(s in k for s in ("accuracy", "apples")))[:6]
        for k in keys:
            print(f"| {e} | {k} | {fmt(median(rs, k))} |")


if __name__ == "__main__":
    main()
