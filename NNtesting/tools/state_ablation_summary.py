#!/usr/bin/env python3
"""Tables for the state component ablation (research log §27;
NNtesting/experiments/state_ablation/sweep.sh).

    python3 NNtesting/tools/state_ablation_summary.py results/state-ablation --md results/state-ablation/summary.md

Reads the ablation_*.jsonl(.gz) files (trial lines only) and prints Markdown:
the online readout per configuration and gap, the same columns read by the
best linear readout, paired differences between configurations (same
seed), confusion matrices, and the state analysis independent of readouts.
"""
import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONFIGS = {"A": "output", "B": "threshold", "C": "habituation", "D": "output + threshold",
           "E": "threshold + habituation", "F": "output + threshold + habituation"}
COMPARISONS = [("B", "A", "primary"), ("D", "A", "primary"), ("D", "B", "primary"),
               ("C", None, "habituation vs chance"), ("E", "B", "habituation added to threshold"),
               ("F", "D", "habituation added to output + threshold")]
T975 = {9: 2.262, 19: 2.093}  # two-sided 95% t quantiles by degrees of freedom


def load(d):
    rows = []
    for f in sorted(Path(d).glob("ablation_*.jsonl*")):
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("type") == "trial" and not r.get("error"):
                    rows.append(r)
    return rows


def by(rows, rec):
    g = defaultdict(dict)  # gap -> seed -> metrics
    for r in rows:
        if r["params"]["recovery"] == rec:
            g[int(r["params"]["gap"])][r["seed"]] = r["metrics"]
    return g


def ms(v, digits=2):
    v = np.asarray(v, float)
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.{digits}f}"


def ci(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2:
        return d.mean(), d.mean(), d.mean()
    half = T975.get(n - 1, 1.96) * d.std(ddof=1) / math.sqrt(n)
    return d.mean(), d.mean() - half, d.mean() + half


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--md")
    args = ap.parse_args()
    rows = load(args.dir)
    out = []
    lr_file = Path(args.dir) / "tuned_lr.txt"
    if lr_file.exists():
        out.append(f"Readout learning rate (tuned once, the same for every configuration): "
                   f"{lr_file.read_text().split(':')[-1].strip()}\n")
    for rec in sorted({r["params"]["recovery"] for r in rows}):
        g = by(rows, rec)
        gaps = sorted(g)
        seeds = sorted(next(iter(g.values())))
        head = "| configuration | " + " | ".join(f"gap {x}" for x in gaps) + " |"
        sep = "|---|" + "---|" * len(gaps)
        out.append(f"## Recovery {rec} ({len(seeds)} paired seeds: {', '.join(map(str, seeds))})\n")

        out.append("### Online readout (library delta rule): balanced accuracy, mean ± sd over seeds (chance 0.50)\n")
        out += [head, sep]
        for k, name in CONFIGS.items():
            out.append(f"| {k} {name} | " + " | ".join(ms([m[f"{k}_balanced"] for m in g[x].values()]) for x in gaps) + " |")
        out.append("")
        out.append("Minimum and maximum over seeds:\n")
        out += [head, sep]
        for k, name in CONFIGS.items():
            cells = []
            for x in gaps:
                v = [m[f"{k}_balanced"] for m in g[x].values()]
                cells.append(f"{min(v):.2f}–{max(v):.2f}")
            out.append(f"| {k} {name} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Accuracy (classes are exactly balanced, so it equals balanced accuracy up to rounding):\n")
        out += [head, sep]
        for k, name in CONFIGS.items():
            out.append(f"| {k} {name} | " + " | ".join(ms([m[f"{k}_accuracy"] for m in g[x].values()]) for x in gaps) + " |")
        out.append("")

        out.append("### The same columns, best linear readout (ridge, offline): is the information there?\n")
        out += [head, sep]
        for k, name in CONFIGS.items():
            out.append(f"| {k} {name} | " + " | ".join(ms([m[f"{k}_ridge"] for m in g[x].values()]) for x in gaps) + " |")
        out.append("")

        out.append("### Paired differences of the online readout (same seed): mean [95% CI]; "
                   "wins = seeds where the first is higher\n")
        out += ["| comparison | " + " | ".join(f"gap {x}" for x in gaps) + " |", sep]
        for a, b, what in COMPARISONS:
            cells = []
            for x in gaps:
                ms_ = g[x].values()
                d = [m[f"{a}_balanced"] - (m[f"{b}_balanced"] if b else 0.5) for m in ms_]
                mean, lo, hi = ci(d)
                wins = sum(v > 0 for v in d)
                mark = "**" if lo > 0 or hi < 0 else ""
                cells.append(f"{mark}{mean:+.2f}{mark} [{lo:+.2f}, {hi:+.2f}] {wins}/{len(d)}")
            label = f"{a} − {b}" if b else f"{a} − 0.5"
            out.append(f"| {label} ({what}) | " + " | ".join(cells) + " |")
        out.append("\nBold: the 95% interval excludes 0.\n")

        out.append("### Confusion matrices, summed over seeds (rows true LEFT, RIGHT; columns predicted LEFT, RIGHT)\n")
        pick = [x for x in (1, 16) if x in g]
        out.append("| configuration | " + " | ".join(f"gap {x}" for x in pick) + " |")
        out.append("|---|" + "---|" * len(pick))
        for k, name in CONFIGS.items():
            cells = []
            for x in pick:
                c = {n: sum(int(m[f"{k}_{n}"]) for m in g[x].values()) for n in ("tp", "fn", "fp", "tn")}
                # RIGHT is the positive class: LEFT row = (tn, fp), RIGHT row = (fn, tp)
                cells.append(f"[[{c['tn']}, {c['fp']}], [{c['fn']}, {c['tp']}]]")
            out.append(f"| {k} {name} | " + " | ".join(cells) + " |")
        out.append("")

        out.append("### The state itself, independent of the readouts (test clips)\n")
        out.append("Per component: population mean for LEFT / RIGHT clips, median over neurons of the class "
                   "separation |d| (Cohen's d), neurons with |d| > 0.5 (of 128), and a ridge probe on that "
                   "component alone. *before*: end of the blank; *final*: last readout tick, immediately "
                   "before the prediction.\n")
        for moment in ("before", "final"):
            out.append(f"**{moment}**\n")
            out.append("| component | measure | " + " | ".join(f"gap {x}" for x in gaps) + " |")
            out.append("|---|---|" + "---|" * len(gaps))
            for c in ("output", "threshold", "habituation"):
                pre = f"state_{moment}_{c}"
                out.append(f"| {c} | mean LEFT / RIGHT | " + " | ".join(
                    f"{np.mean([m[pre + '_mean_left'] for m in g[x].values()]):.3f} / "
                    f"{np.mean([m[pre + '_mean_right'] for m in g[x].values()]):.3f}" for x in gaps) + " |")
                out.append(f"| | median \\|d\\| | " + " | ".join(
                    f"{np.mean([m[pre + '_d_median'] for m in g[x].values()]):.2f}" for x in gaps) + " |")
                out.append(f"| | neurons \\|d\\| > 0.5 | " + " | ".join(
                    f"{np.mean([m[pre + '_neurons_d_over_0.5'] for m in g[x].values()]):.0f}" for x in gaps) + " |")
                out.append(f"| | ridge probe | " + " | ".join(
                    ms([m[f"probe_{moment}_{c}"] for m in g[x].values()]) for x in gaps) + " |")
            out.append("")
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
