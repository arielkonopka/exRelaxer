#!/usr/bin/env python3
"""Summarises video_memory runs (nntest JSON Lines): accuracy per model and
gap over seeds, the memory horizon, probes, activity, and a figure.

    python3 NNtesting/tools/video_memory_summary.py results/video-memory/*.jsonl.gz \
        --md results/video-memory/summary.md --png doc/diagrams/video_memory.png

Significantly above chance: the one-sided 99% lower confidence bound of the
mean over seeds (Student t) is above 0.5. The memory horizon of a model is
the largest gap at which that holds.
"""
import argparse
import collections
import gzip
import json
import math
import sys

import numpy as np

# One-sided 99% Student t critical values by degrees of freedom.
T99 = {1: 31.82, 2: 6.965, 3: 4.541, 4: 3.747, 5: 3.365, 6: 3.143, 7: 2.998, 8: 2.896, 9: 2.821, 10: 2.764,
       11: 2.718, 12: 2.681, 13: 2.650, 14: 2.624, 15: 2.602, 16: 2.583, 17: 2.567, 18: 2.552, 19: 2.539,
       20: 2.528, 24: 2.492, 29: 2.462, 39: 2.426}
GAPS = (1, 2, 4, 8, 16)
ORDER = ("C0", "C1", "E0", "R1", "R4", "E1", "Rg", "D2", "D2R")


def t99(df):
    return T99.get(df) or T99[max(k for k in T99 if k <= df)]


def load(paths):
    groups = collections.defaultdict(list)
    for path in paths:
        with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") != "trial" or r.get("experiment") != "video_memory":
                    continue
                p = r["params"]
                variant = "frozen"
                if p.get("hidden_rule", "frozen") == "fa":
                    variant = f"fa lr={p['fa_lr']}"
                elif float(p.get("recovery", 0.9)) != 0.9:
                    variant = f"recovery={p['recovery']}"
                groups[(p["task"], variant, p["model"], int(p["gap"]))].append(r["metrics"])
    return groups


def stat(values):
    v = np.asarray(values, float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else 0.0
    lower = v.mean() - t99(n - 1) * sd / math.sqrt(n) if n > 1 else v.mean()
    return dict(n=n, mean=v.mean(), sd=sd, median=float(np.median(v)), min=v.min(), max=v.max(),
                lower99=lower, above=lower > 0.5)


def horizon(groups, key3, metric="accuracy"):
    best = 0
    for g in GAPS:
        ms = groups.get(key3 + (g,))
        if ms and stat([m[metric] for m in ms])["above"]:
            best = g
    return best


def table(groups, task, variant, metric, models, fmt="{:.3f}"):
    rows = [f"| model | " + " | ".join(f"gap {g}" for g in GAPS) + " | horizon |",
            "|---|" + "---|" * (len(GAPS) + 1)]
    for m in models:
        if not any((task, variant, m, g) in groups for g in GAPS):
            continue
        cells = []
        for g in GAPS:
            ms = groups.get((task, variant, m, g))
            if not ms or metric not in ms[0]:
                cells.append("")
                continue
            s = stat([x[metric] for x in ms])
            cell = fmt.format(s["mean"]) + f" ± {s['sd']:.3f}"
            cells.append(f"**{cell}**" if s["above"] and metric.startswith(("accuracy", "probe")) else cell)
        h = horizon(groups, (task, variant, m), metric) if metric.startswith(("accuracy", "probe")) else ""
        rows.append(f"| {m} | " + " | ".join(cells) + f" | {h} |")
    return "\n".join(rows)


def per_seed(groups, task, variant, m, metric="accuracy"):
    out = []
    for g in GAPS:
        ms = groups.get((task, variant, m, g))
        if ms:
            s = stat([x[metric] for x in ms])
            vals = " ".join(f"{x[metric]:.3f}" for x in ms)
            out.append(f"- gap {g}: median {s['median']:.3f}, range {s['min']:.3f}-{s['max']:.3f}; {vals}")
    return "\n".join(out)


def paired(groups, task, a, b, variant="frozen"):
    out = []
    for g in GAPS:
        A, B = groups.get((task, variant, a, g)), groups.get((task, variant, b, g))
        if A and B and len(A) == len(B):
            d = [x["accuracy"] - y["accuracy"] for x, y in zip(A, B)]
            s = stat(np.asarray(d) + 0.5)  # "above 0.5" == difference above 0
            out.append(f"gap {g}: {s['mean'] - 0.5:+.3f} (lower 99% bound {s['lower99'] - 0.5:+.3f}, "
                       f"{sum(x > 0 for x in d)}/{len(d)} seeds)")
    return "; ".join(out)


def figure(groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"E0": "#2a78d6", "C1": "#eb6834", "C0": "#8a8984", "R1": "#1baf7a", "R4": "#eda100",
              "E1": "#e87ba4", "Rg": "#4a3aa7", "D2": "#008300", "D2R": "#4a3aa7"}
    ink, muted = "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    x = np.arange(len(GAPS))

    def line(ax, key3, metric, label, color, style="-", marker="o"):
        means = []
        for i, g in enumerate(GAPS):
            ms = groups.get(key3 + (g,))
            if not ms:
                means.append(np.nan)
                continue
            v = [m[metric] for m in ms]
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)), v,
                       s=9, color=color, alpha=0.25, linewidths=0)
            means.append(np.mean(v))
        ax.plot(x, means, style, color=color, lw=2, marker=marker, ms=6, label=label)

    panels = [
        ("Readout accuracy (side task)", [(("side", "frozen", m), "accuracy", m, colors[m], "-")
                                          for m in ("C0", "C1", "E0", "R1", "R4", "E1", "Rg")]),
        ("E0: state holds it, readout loses it", [
            (("side", "frozen", "E0"), "probe_before_reappearance", "E0 probe: state before reappearance", colors["E0"], "--"),
            (("side", "frozen", "E0"), "accuracy", "E0 readout", colors["E0"], "-"),
            (("side", "frozen", "C1"), "probe_before_reappearance", "C1 probe (reset every frame)", colors["C1"], "--"),
            (("side", "frozen", "D2R"), "accuracy", "D2R readout (recurrent)", colors["D2"], "-")]),
        ("E0 readout by E-R recovery (sensitivity)", [
            (("side", "frozen", "E0"), "accuracy", "0.9 (library default)", colors["E0"], "-"),
            (("side", "recovery=0.95", "E0"), "accuracy", "0.95", "#1baf7a", "-"),
            (("side", "recovery=0.97", "E0"), "accuracy", "0.97", "#eda100", "-"),
            (("side", "recovery=0.99", "E0"), "accuracy", "0.99", "#e87ba4", "-")]),
    ]
    for ax, (title, series) in zip(axes, panels):
        ax.axhline(0.5, color=muted, lw=1, ls=":")
        for key3, metric, label, color, style in series:
            if any(key3 + (g,) in groups for g in GAPS):
                line(ax, key3, metric, label, color, style)
        ax.set_xticks(x, [str(g) for g in GAPS])
        ax.set_xlabel("blank frames (gap)", color=muted)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=8, labelcolor=muted)
    axes[0].set_ylabel("test accuracy (dots: seeds)", color=muted)
    axes[0].set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--png")
    args = ap.parse_args()
    groups = load(args.files)
    tasks_variants = sorted({(k[0], k[1]) for k in groups})
    out = []
    for task, variant in tasks_variants:
        models = [m for m in ORDER if any((task, variant, m, g) in groups for g in GAPS)]
        out.append(f"## {task}, {variant}\n")
        for metric in ("accuracy", "probe_pre_occlusion", "probe_before_reappearance", "accuracy_tick_fit",
                       "fa_online_accuracy", "train_accuracy"):
            if any(metric in groups[k][0] for k in groups if k[:2] == (task, variant)):
                out.append(f"### {metric} (mean ± sd over seeds; bold: 99% lower bound above 0.5)\n")
                out.append(table(groups, task, variant, metric, models) + "\n")
        for metric in ("spikes_per_frame", "spikes_visible", "spikes_blank", "spikes_readout",
                       "spikes_first_readout_tick", "threshold_pre_occlusion_mean", "threshold_blank_mean",
                       "threshold_reappearance_p10", "threshold_reappearance_p50", "threshold_reappearance_p90"):
            if any(metric in groups[k][0] for k in groups if k[:2] == (task, variant)):
                out.append(f"### {metric}\n")
                out.append(table(groups, task, variant, metric, models, "{:.4g}") + "\n")
        if variant == "frozen":
            out.append("### Per-seed accuracy\n")
            for m in models:
                out.append(f"**{m}**\n\n" + per_seed(groups, task, variant, m) + "\n")
            out.append("### Paired differences (same clips, same seeds)\n")
            for a, b in (("E0", "C1"), ("E0", "C0"), ("E1", "R1"), ("E0", "R1"), ("D2R", "D2"), ("D2", "E0")):
                s = paired(groups, task, a, b)
                if s:
                    out.append(f"- {a} − {b}: {s}")
            out.append("")
    text = "\n".join(out)
    if args.md:
        with open(args.md, "w") as f:
            f.write("# video_memory summary\n\nGenerated by NNtesting/tools/video_memory_summary.py.\n\n" + text)
    else:
        sys.stdout.write(text)
    if args.png:
        figure(groups, args.png)


if __name__ == "__main__":
    main()
