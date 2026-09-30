#!/usr/bin/env python3
"""Tables for the delayed_credit and memory_readability runs (research log §25).

    python3 NNtesting/tools/temporal_summary.py results/delayed-credit results/memory-readability --md summary.md

Reads every *.jsonl or *.jsonl.gz in the given directories (trial lines
only) and prints Markdown.
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def trials(paths):
    for d in paths:
        for f in sorted(Path(d).glob("*.jsonl*")):
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("type") == "trial" and not r.get("error"):
                        r["file"] = f.name.split(".")[0]
                        yield r


def ms(values, digits=2):
    v = np.array(values, float)
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.{digits}f}"


def credit(rows, out):
    groups = defaultdict(list)
    for r in rows:
        p = r["params"]
        rule = p["rule"]
        key = (r["file"], rule, p["neuron"], p["recovery"] if p["neuron"] == "er" else "-",
               p["trace"] if rule == "trace" else "-", p["episodes"])
        groups[key].append(r)
    delays = sorted({int(r["params"]["delay"]) for r in rows})
    out.append("### delayed_credit: success rate (of seeds) and selectivity SNR by delay\n")
    out.append("Success: the rewarded cue's weight ends above every irrelevant weight and the punished cue's "
               "below every one. SNR (median over seeds): (Δw₊ − Δw₋)/2 over the RMS change of the irrelevant weights.\n")
    head = "| rule | neuron | recovery | λ | episodes | " + " | ".join(f"d={d}" for d in delays) + " | horizon |"
    out.append(head)
    out.append("|" + "---|" * (6 + len(delays)))
    for key in sorted(groups, key=lambda k: (k[1], k[2], str(k[3]), str(k[4]), k[5])):
        _, rule, neuron, rec, lam, eps = key
        by_d = defaultdict(list)
        for r in groups[key]:
            by_d[int(r["params"]["delay"])].append(r["metrics"])
        cells, horizon, unbroken = [], None, True
        for d in delays:
            m = by_d.get(d)
            if not m:
                cells.append("")
                continue
            s = np.mean([x["success"] for x in m])
            # median: an SNR is infinite (null in the JSON) when no irrelevant weight moved
            snr = np.median([np.inf if x["selectivity_snr"] is None else x["selectivity_snr"] for x in m])
            cells.append(f"{s:.1f} ({snr:.1f})")
            unbroken = unbroken and s >= 0.8
            if unbroken:
                horizon = d
        out.append(f"| {rule} | {neuron} | {rec} | {lam} | {eps} | " + " | ".join(cells) +
                   f" | {horizon if horizon is not None else '<0'} |")
    out.append("\nHorizon: the largest delay up to which every tested delay has success ≥ 0.8.\n")

    # Weight changes for a few conditions.
    out.append("### delayed_credit: weight changes (mean ± sd over seeds)\n")
    out.append("| rule | neuron | recovery | λ | episodes | delay | Δw rewarded cue | Δw punished cue | |Δw| irrelevant cues | "
               "|Δw| distractors | reward corr. (relevant cues) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    # nntest records parameters as text
    pick = [("trace", "er", "0.9", "0.9"), ("trace", "er", "0.9", "0.5"), ("trace", "er", "0.9", "0.99"),
            ("trace", "linear", "-", "0.9"), ("sign", "er", "0.9", "-"), ("sign", "er", "0.99", "-"),
            ("sign", "linear", "-", "-")]
    for rule, neuron, rec, lam in pick:
        for eps in ("400", "1600"):
            for d in delays:
                m = [r["metrics"] for key, rs in groups.items() for r in rs
                     if key[1:5] == (rule, neuron, rec, lam) and key[5] == eps and int(r["params"]["delay"]) == d]
                if not m or d not in (0, 1, 4, 16, 64):
                    continue
                out.append(f"| {rule} | {neuron} | {rec} | {lam} | {eps} | {d} | "
                           f"{ms([x['dw_relevant_pos'] for x in m])} | {ms([x['dw_relevant_neg'] for x in m])} | "
                           f"{ms([x['dw_irrelevant_cue_abs'] for x in m])} | {ms([x['dw_distractor_abs'] for x in m])} | "
                           f"{ms([x['reward_corr_relevant'] for x in m])} |")
    out.append("")


def readability(rows, out):
    groups = defaultdict(list)
    for r in rows:
        p = r["params"]
        key = (p["model"], p["recovery"] if p["model"] != "relu" else "-", p["noise"])
        groups[key].append(r)
    blanks = sorted({int(r["params"]["blank"]) for r in rows})
    names = [("ridge_state_before", "stored: ridge on thresholds before the query"),
             ("ridge_state", "stored: ridge on thresholds at the query"),
             ("ridge_eligibility", "ridge on Sign eligibility"),
             ("ridge_output", "readable: ridge on outputs (A)"),
             ("ridge_output_state", "ridge on outputs + thresholds (C)"),
             ("online_output", "usable: online readout on outputs (A)"),
             ("online_state", "online readout on thresholds (B)"),
             ("online_output_state", "online readout on outputs + thresholds (C)")]
    out.append("### memory_readability: balanced accuracy by blank length (mean ± sd over seeds; chance 0.50)\n")
    for key in sorted(groups, key=lambda k: (k[0] != "er", k[0], str(k[1]), -float(k[2]))):
        model, rec, noise = key
        by_b = defaultdict(list)
        for r in groups[key]:
            by_b[int(r["params"]["blank"])].append(r["metrics"])
        n = len(next(iter(by_b.values())))
        out.append(f"**model={model}, recovery={rec}, noise={noise}** ({n} seeds)\n")
        out.append("| readout | " + " | ".join(f"{b}" for b in blanks) + " |")
        out.append("|---|" + "---|" * len(blanks))
        for metric, label in names:
            cells = []
            for b in blanks:
                m = by_b.get(b)
                cells.append(ms([x[f"{metric}_balanced"] for x in m]) if m else "")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        cells = [f"{np.mean([x['query_output_active'] for x in by_b[b]]):.2f}" if b in by_b else "" for b in blanks]
        out.append("| active hidden neurons at the query | " + " | ".join(cells) + " |")
        out.append("")
    # One confusion matrix example.
    for key in groups:
        if key[0] == "er" and key[1] == 0.9 and key[2] == 0.05:
            for b in (4, 32):
                m = [r["metrics"] for r in groups[key] if int(r["params"]["blank"]) == b]
                if m:
                    c = {k: int(sum(x[f"online_output_{k}"] for x in m)) for k in ("tp", "fn", "fp", "tn")}
                    out.append(f"Confusion, online readout on outputs, er, recovery 0.9, noise 0.05, blank {b}, "
                               f"summed over seeds (rows true LEFT / RIGHT, columns predicted LEFT / RIGHT): "
                               f"[[{c['tp']}, {c['fn']}], [{c['fp']}, {c['tn']}]]\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", help="also write the Markdown here")
    args = ap.parse_args()
    rows = list(trials(args.dirs))
    out = []
    dc = [r for r in rows if r["experiment"] == "delayed_credit"]
    mr = [r for r in rows if r["experiment"] == "memory_readability"]
    if dc:
        credit(dc, out)
    if mr:
        readability(mr, out)
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
