#!/usr/bin/env python3
"""Summarise results/recovery/run.sh: recovery 0.9 against 0.5, as Markdown.

usage: analyze.py RAW_DIR > summary.md

Activity experiments: means over 20 trials of the metrics the research log
cites. nl_static / nl_temporal: per task the best architecture (depth x
width) by median test result, the learning rate per architecture chosen on
median validation. dyn_ladder, snake_rules, bar_orientation: means.
"""
import collections
import gzip
import json
import os
import statistics as st
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "raw")
RECS = ("0.9", "0.5")


def trials(name, r):
    path = os.path.join(RAW, f"{name}_rec{r}.jsonl")
    if not os.path.exists(path):
        path += ".gz"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if '"trial"' in line:
                d = json.loads(line)
                if d.get("type") == "trial" and not d.get("error"):
                    yield d


def mean(name, r, metric, where=None):
    v = [t["metrics"][metric] for t in trials(name, r)
         if metric in t["metrics"] and (where is None or where(t["params"]))]
    return st.mean(v) if v else float("nan")


ACTIVITY = [
    ("er_economy", "er_accuracy", "E-R accuracy"),
    ("er_economy", "er_active_fraction", "E-R active fraction"),
    ("er_economy", "gate_active_fraction", "fixed threshold (matched) active fraction"),
    ("er_economy", "er_spikes_per_inference", "E-R spikes per inference"),
    ("er_history", "er_A_pattern_change", "history: pattern change after path A's use"),
    ("er_history", "er_all_pattern_change", "history: pattern change after all paths' use"),
    ("er_history", "er_all_decision_change", "history: decision change after all paths' use"),
    ("er_fatigue", "er_all_first_active_fraction", "fatigue (all paths): first active fraction"),
    ("er_fatigue", "er_all_first_accuracy", "fatigue (all paths): first accuracy"),
    ("er_fatigue", "er_B_first_active_fraction", "fatigue (path B): first active fraction"),
    ("er_silence", "er_self_active_ticks", "silence: self-active ticks"),
    ("er_silence", "er_after_spikes_ratio", "silence: spikes after / before"),
    ("er_habituation", "er_h50_spikes_per_sample", "habituation off, hold 50: spikes per sample"),
    ("er_habituation", "er_h50_accuracy_end", "hold 50: accuracy at the end of the hold"),
    ("er_paths", "er_dwell_mean_active_run", "paths: mean active run during a dwell"),
]


def best(kind, task, r, temporal):
    groups = collections.defaultdict(list)
    for t in trials(f"{kind}_{task}", r):
        p, m = t["params"], t["metrics"]
        if temporal:
            val, test = m["best_validation"], m.get("test_accuracy")
            if test is None:
                test = -m.get("test_mse", float("nan"))
        else:
            val, test = -m["validation_mse"], -m["test_mse"]
        groups[(p.get("n", ""), p["depth"], p["width"], p["lr"])].append((val, test, m["active_fraction"]))
    by_n = collections.defaultdict(lambda: collections.defaultdict(list))
    for (n, d, w, lr), v in groups.items():
        by_n[n][(d, w)].append((st.median(x[0] for x in v), st.median(x[1] for x in v), st.mean(x[2] for x in v)))
    out = {}
    for n, archs in by_n.items():
        chosen = {a: max(v, key=lambda x: x[0]) for a, v in archs.items()}
        a, (val, test, act) = max(chosen.items(), key=lambda kv: kv[1][1])
        out[n] = (a, test if temporal else -test, act)
    return out


def main():
    print("# E-R experiments at recovery 0.9 and 0.5\n")
    print("Runs of `results/recovery/run.sh` (current defaults, same seeds, only the recovery differs).\n")
    print("## Activity experiments (20 trials, means)\n")
    print("| experiment | metric | 0.9 | 0.5 |\n|---|---|---|---|")
    for e, m, label in ACTIVITY:
        print(f"| {e} | {label} | {mean(e, '0.9', m):.3g} | {mean(e, '0.5', m):.3g} |")
    print("\n## nl_static (test MSE) and nl_temporal (test accuracy; t4: -MSE), E-R only, 5 seeds\n")
    print("| task | 0.9: best (depth, width) | 0.9 active | 0.5: best (depth, width) | 0.5 active |\n|---|---|---|---|---|")
    for kind, tasks, temporal in (("nl_static", ("l0", "l1", "l2", "l3", "l4"), False),
                                  ("nl_temporal", ("t1", "t2", "t3", "t4"), True)):
        for task in tasks:
            a, b = best(kind, task, "0.9", temporal), best(kind, task, "0.5", temporal)
            for n in sorted(a):
                label = task + (f" n={n}" if n else "")
                (aa, at, aact), (ba, bt, bact) = a[n], b[n]
                print(f"| {label} | {at:.4g} ({aa[0]}, {aa[1]}) | {aact:.2f} | {bt:.4g} ({ba[0]}, {ba[1]}) | {bact:.2f} |")
    print("\n## Other experiments (means)\n")
    print("| experiment | metric | 0.9 | 0.5 |\n|---|---|---|---|")
    for task in ("dir", "change", "vel", "catch"):
        f = lambda p, task=task: p["task"] == task
        print(f"| dyn_ladder {task} | test score | {mean('dyn_ladder', '0.9', 'test_score', f):.3f} "
              f"| {mean('dyn_ladder', '0.5', 'test_score', f):.3f} |")
    for readout in ("sign", "fa"):
        for mix in ("frozen", "fa"):
            f = lambda p, readout=readout, mix=mix: p["readout"] == readout and p["mix"] == mix
            print(f"| snake_rules E-R mix, {readout} readouts, {mix} mix | test apples "
                  f"| {mean('snake_rules', '0.9', 'apples', f):.2f} | {mean('snake_rules', '0.5', 'apples', f):.2f} |")
    print(f"| bar_orientation, E-R readout | accuracy | {mean('bar_orientation', '0.9', 'accuracy'):.3f} "
          f"| {mean('bar_orientation', '0.5', 'accuracy'):.3f} |")


if __name__ == "__main__":
    main()
