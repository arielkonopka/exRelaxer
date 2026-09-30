#!/usr/bin/env python3
"""Tables for the state-as-output runs (research log §26; NNtesting/tools/state_output_sweep.sh).

    python3 NNtesting/tools/state_output_summary.py results/state-output --md results/state-output/summary.md

Reads every *.jsonl or *.jsonl.gz in the directory (trial lines only) and
prints Markdown: each benchmark with and without the State layer, from the
same seeds.
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def trials(d):
    rows = defaultdict(list)
    for f in sorted(Path(d).glob("*.jsonl*")):
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("type") == "trial" and not r.get("error"):
                    rows[r["experiment"]].append(r)
    return rows


def ms(values, digits=2):
    v = np.array(values, float)
    if len(v) == 0:
        return ""
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.{digits}f}"


def group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(str(r["params"][k]) for k in keys)].append(r)
    return g


def memory_readability(rows, out):
    blanks = sorted({int(r["params"]["blank"]) for r in rows})
    readouts = [("output", "outputs (A)"), ("state", "thresholds via State (B)"),
                ("habituation", "habituation streaks via State"), ("output_tap", "outputs + State (D)")]
    out.append("### memory_readability: balanced accuracy by blank length (mean ± sd over seeds; chance 0.50)\n")
    out.append("The State layer's values are what a downstream neuron reads; ridge = best linear readout, "
               "online = the library's delta rule.\n")
    for (rec, hab), rs in sorted(group(rows, "recovery", "habituation").items()):
        by_b = defaultdict(list)
        for r in rs:
            by_b[int(r["params"]["blank"])].append(r["metrics"])
        n = len(next(iter(by_b.values())))
        out.append(f"**recovery={rec}, habituation={'on (steps 4, tolerance 0.5)' if hab == '1' else 'off'}** "
                   f"({n} seeds)\n")
        out.append("| readout | " + " | ".join(str(b) for b in blanks) + " |")
        out.append("|---|" + "---|" * len(blanks))
        for how in ("ridge", "online"):
            for key, label in readouts:
                if hab == "0" and key == "habituation":
                    continue
                cells = [ms([m[f"{how}_{key}_balanced"] for m in by_b[b]]) if b in by_b else "" for b in blanks]
                out.append(f"| {how}: {label} | " + " | ".join(cells) + " |")
        if hab == "1":
            cells = [f"{np.mean([m['query_habituated'] for m in by_b[b]]):.2f}" for b in blanks]
            out.append("| neurons habituated at the query | " + " | ".join(cells) + " |")
        out.append("")


def delayed_credit(rows, out):
    delays = sorted({int(r["params"]["delay"]) for r in rows})
    groups = group(rows, "rule", "lr", "state", "relay_recovery")
    out.append("### delayed_credit: success rate over seeds by delay\n")
    out.append("Success: the rewarded cue's weight ends above every irrelevant weight and the punished cue's "
               "below every one; for state=tap separately for the direct input weights (inputs) and the "
               "relay-threshold weights (State).\n")
    out.append("| rule | lr | state | relay recovery | weights | " + " | ".join(f"d={d}" for d in delays) + " |")
    out.append("|---|---|---|---|---|" + "---|" * len(delays))
    for (rule, lr, state, rr), rs in sorted(groups.items()):
        if state == "none" and rr != "0.99":
            continue  # relay_recovery is unused without relays: one row
        by_d = defaultdict(list)
        for r in rs:
            by_d[int(r["params"]["delay"])].append(r["metrics"])
        halves = [("", "inputs")] + ([("_state", "State")] if state == "tap" else [])
        for suffix, label in halves:
            cells = [f"{np.mean([m['success' + suffix] for m in by_d[d]]):.1f}" if d in by_d else "" for d in delays]
            out.append(f"| {rule} | {lr} | {state} | {rr if state == 'tap' else '-'} | {label} | "
                       + " | ".join(cells) + " |")
    out.append("")
    out.append("Learned response difference, rewarded cue minus punished cue, at the cue tick (test, no "
               "learning; mean ± sd over seeds; 0 = the cues are not told apart):\n")
    out.append("| rule | lr | state | relay recovery | " + " | ".join(f"d={d}" for d in delays) + " |")
    out.append("|---|---|---|---|" + "---|" * len(delays))
    for (rule, lr, state, rr), rs in sorted(groups.items()):
        if state == "none" and rr != "0.99":
            continue
        by_d = defaultdict(list)
        for r in rs:
            by_d[int(r["params"]["delay"])].append(r["metrics"])
        cells = [ms([m["response_cue0"] - m["response_cue1"] for m in by_d[d]], 1) if d in by_d else ""
                 for d in delays]
        out.append(f"| {rule} | {lr} | {state} | {rr if state == 'tap' else '-'} | " + " | ".join(cells) + " |")
    out.append("")


def video_memory_learned(rows, out):
    gaps = sorted({int(r["params"]["gap"]) for r in rows})
    keys = [("ridge_accuracy", "ridge"), ("t0_accuracy", "online, trace 0"), ("t0.8_accuracy", "online, trace 0.8"),
            ("t0.99_accuracy", "online, trace 0.99")]
    out.append("### video_memory_learned: test accuracy by blank frames (mean ± sd over seeds; chance 0.50)\n")
    for (rec,), rs in sorted(group(rows, "recovery").items()):
        by = defaultdict(list)
        for r in rs:
            by[(r["params"]["readout_inputs"], int(r["params"]["gap"]))].append(r["metrics"])
        out.append(f"**recovery={rec}** ({len(by[next(iter(by))])} seeds)\n")
        out.append("| readout | inputs | " + " | ".join(f"gap {g}" for g in gaps) + " |")
        out.append("|---|---|" + "---|" * len(gaps))
        for key, label in keys:
            for inputs in ("output", "output_state"):
                cells = [ms([m[key] for m in by[(inputs, g)] if key in m]) for g in gaps]
                if not any(cells):
                    continue  # a trace that was not run
                out.append(f"| {label} | {'outputs' if inputs == 'output' else 'outputs + State'} | "
                           + " | ".join(cells) + " |")
        out.append("")


def trained(rows, out, name, metric, tasks_key="task"):
    """nl_temporal / dyn_ladder: per task and setting, the learning rate with the
    best mean validation score, and the test metric there."""
    extra = ["habituation"] if name == "nl_temporal" else []
    g = group(rows, tasks_key, *extra, "state_readout", "lr")
    best = {}
    for key, rs in g.items():
        cfg, lr = key[:-1], key[-1]
        val = np.mean([r["metrics"]["best_validation"] for r in rs])
        if cfg not in best or val > best[cfg][0]:
            best[cfg] = (val, lr, rs)
    out.append(f"### {name}: test score at the learning rate with the best validation "
               "(mean ± sd over seeds)\n")
    head = "| task | " + ("habituation | " if extra else "") + "readout reads | lr | test score | spikes |"
    out.append(head)
    out.append("|" + "---|" * (head.count("|") - 1))
    spikes = "spikes_per_sample" if name == "nl_temporal" else "spikes_per_step"
    for cfg in sorted(best):
        _, lr, rs = best[cfg]
        task, sr = cfg[0], cfg[-1]
        m = [r["metrics"] for r in rs]
        score = [x[metric] if metric in x else x.get("test_accuracy") for x in m]
        cols = [task] + ([cfg[1]] if extra else []) + ["outputs + State" if sr == "true" else "outputs", lr,
                                                       ms(score, 3), ms([x.get(spikes, np.nan) for x in m], 1)]
        out.append("| " + " | ".join(cols) + " |")
    out.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--md", help="also write the Markdown here")
    args = ap.parse_args()
    rows = trials(args.dir)
    out = []
    if rows.get("memory_readability"):
        memory_readability(rows["memory_readability"], out)
    if rows.get("delayed_credit"):
        delayed_credit(rows["delayed_credit"], out)
    if rows.get("video_memory_learned"):
        video_memory_learned(rows["video_memory_learned"], out)
    if rows.get("nl_temporal"):
        out.append("Score: test accuracy (t1-t3), 1 - NMSE (t4).\n")
        for r in rows["nl_temporal"]:
            m = r["metrics"]
            m["score"] = 1.0 - m["test_nmse"] if r["params"]["task"] == "t4" else m["test_accuracy"]
        trained(rows["nl_temporal"], out, "nl_temporal", "score")
    if rows.get("dyn_ladder"):
        out.append("Score: accuracy (dir, change), R² (vel), catch rate (catch).\n")
        trained(rows["dyn_ladder"], out, "dyn_ladder", "test_score")
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
