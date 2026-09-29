#!/usr/bin/env python3
"""Summarises video_memory_learned runs (nntest JSON Lines): per readout
(ridge, online with each eligibility trace), gap and configuration, accuracy
over seeds with per-class accuracy and confusion matrices, probes, training
error, credit-assignment diagnostics, and a figure.

    python3 NNtesting/tools/video_memory_learned_summary.py results/video-memory-learned/*.jsonl.gz \
        --md results/video-memory-learned/summary.md --png doc/diagrams/video_memory_learned.png

Significantly above chance: the one-sided 99% lower confidence bound of the
mean over seeds (Student t) is above 0.5 (as video_memory_summary.py).
"""
import argparse
import collections
import gzip
import json
import math

import numpy as np

T99 = {1: 31.82, 2: 6.965, 3: 4.541, 4: 3.747, 5: 3.365, 6: 3.143, 7: 2.998, 8: 2.896, 9: 2.821, 10: 2.764,
       11: 2.718, 12: 2.681, 13: 2.650, 14: 2.624, 15: 2.602, 16: 2.583, 17: 2.567, 18: 2.552, 19: 2.539,
       20: 2.528, 24: 2.492, 29: 2.462, 39: 2.426}
GAPS = (1, 2, 4, 8, 16)
TRACES = ("0", "0.5", "0.8", "0.9", "0.95", "0.99")


def t99(df):
    return T99.get(df) or T99[max(k for k in T99 if k <= df)]


def config(p):
    """A configuration label (everything but the gap and the seed)."""
    parts = [p["model"], f"recovery {float(p['recovery']):g}"]
    if p.get("mode", "readout") == "hidden":
        parts.append(f"hidden learns (trace {float(p['hidden_trace']):g}, lr {float(p['hidden_lr']):g})")
    if p.get("error_at", "end") != "end":
        parts.append(f"error every {p['error_at']}")
    if p.get("input_gain", "sd") != "sd":
        parts.append(f"input gain {p['input_gain']}")
    return ", ".join(parts)


def load(paths):
    groups = collections.defaultdict(list)
    for path in paths:
        with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") != "trial" or r.get("experiment") != "video_memory_learned" or r.get("error"):
                    continue
                if r["params"].get("dump"):
                    continue   # the credit dump runs repeat a primary configuration
                groups[(config(r["params"]), int(r["params"]["gap"]))].append(r["metrics"])
    return groups


def stat(values):
    v = np.asarray(values, float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else 0.0
    lower = v.mean() - t99(n - 1) * sd / math.sqrt(n) if n > 1 else v.mean()
    return dict(n=n, mean=v.mean(), sd=sd, var=sd ** 2, min=v.min(), max=v.max(), lower99=lower,
                above=lower > 0.5)


def cell(ms, metric, bold=True, fmt="{:.3f}"):
    if not ms or metric not in ms[0]:
        return ""
    s = stat([m[metric] for m in ms])
    text = fmt.format(s["mean"]) + f" ± {s['sd']:.3f}"
    return f"**{text}**" if bold and s["above"] else text


def readouts(ms):
    keys = [k[:-len("_accuracy")] for k in ms[0] if k.endswith("_accuracy") and k.startswith("t")
            and k.count("_") == 1]
    keys.sort(key=lambda k: float(k[1:]) if k[1:].replace(".", "").isdigit() else -1)
    return ["ridge"] + keys


def horizon(groups, cfg, metric):
    best = 0
    for g in GAPS:
        ms = groups.get((cfg, g))
        if ms and metric in ms[0] and stat([m[metric] for m in ms])["above"]:
            best = g
    return best


def label(key):
    if key == "ridge":
        return "ridge readout (offline, video_memory)"
    if key == "t0":
        return "online, immediate (no trace)"
    return f"online, eligibility trace {key[1:]}"


def accuracy_table(groups, cfg):
    first = next(groups[(cfg, g)] for g in GAPS if (cfg, g) in groups)
    rows = ["| readout | " + " | ".join(f"gap {g}" for g in GAPS) + " | horizon |",
            "|---|" + "---|" * (len(GAPS) + 1)]
    rows.append("| constant (trivial 50/50) | " + " | ".join(
        cell(groups.get((cfg, g)), "constant_accuracy", bold=False) for g in GAPS) + " | 0 |")
    for key in readouts(first):
        rows.append(f"| {label(key)} | " + " | ".join(cell(groups.get((cfg, g)), f"{key}_accuracy") for g in GAPS)
                    + f" | {horizon(groups, cfg, key + '_accuracy')} |")
    return "\n".join(rows)


def probe_table(groups, cfg):
    first = next(groups[(cfg, g)] for g in GAPS if (cfg, g) in groups)
    metrics = [("probe_before_reappearance", "state before the reappearance (thresholds + outputs)"),
               ("probe_prediction_state", "state at the prediction (thresholds after it + mean outputs)"),
               ("probe_prediction_outputs", "outputs at the prediction (what the readout sees)")]
    metrics += [(k, f"eligibility trace {k.split('_t')[-1]} at the error") for k in first[0]
                if k.startswith("probe_eligibility_t")]
    rows = ["| probe on | " + " | ".join(f"gap {g}" for g in GAPS) + " |", "|---|" + "---|" * len(GAPS)]
    for metric, name in metrics:
        rows.append(f"| {name} | " + " | ".join(cell(groups.get((cfg, g)), metric) for g in GAPS) + " |")
    return "\n".join(rows)


def class_table(groups, cfg):
    first = next(groups[(cfg, g)] for g in GAPS if (cfg, g) in groups)
    rows = ["| readout | gap | accuracy | LEFT acc | RIGHT acc | balanced | confusion, summed over seeds "
            "(true L: pred L / pred R; true R: pred L / pred R) | seeds |",
            "|---|---|---|---|---|---|---|---|"]
    for key in readouts(first):
        for g in GAPS:
            ms = groups.get((cfg, g))
            if not ms or f"{key}_accuracy" not in ms[0]:
                continue
            cm = {c: int(sum(m[f"{key}_cm_{c}"] for m in ms)) for c in ("LL", "LR", "RL", "RR")}
            rows.append(f"| {label(key)} | {g} | {cell(ms, key + '_accuracy', False)} | "
                        f"{cell(ms, key + '_left_accuracy', False)} | {cell(ms, key + '_right_accuracy', False)} | "
                        f"{cell(ms, key + '_balanced_accuracy', False)} | "
                        f"{cm['LL']} / {cm['LR']}; {cm['RL']} / {cm['RR']} | {len(ms)} |")
    return "\n".join(rows)


def training_table(groups, cfg):
    first = next(groups[(cfg, g)] for g in GAPS if (cfg, g) in groups)
    rows = ["| readout | gap | error, epoch 1 → last (½‖target − prediction‖²) | train acc | learning rates chosen |",
            "|---|---|---|---|---|"]
    for key in readouts(first)[1:]:
        for g in GAPS:
            ms = groups.get((cfg, g))
            if not ms or f"{key}_mse_first" not in ms[0]:
                continue
            lrs = collections.Counter(f"{m[key + '_lr']:g}" for m in ms)
            rows.append(f"| {label(key)} | {g} | {stat([m[key + '_mse_first'] for m in ms])['mean']:.3f} → "
                        f"{stat([m[key + '_mse_last'] for m in ms])['mean']:.3f} | "
                        f"{cell(ms, key + '_train_accuracy', False)} | "
                        + ", ".join(f"{k} ×{v}" for k, v in sorted(lrs.items())) + " |")
    return "\n".join(rows)


def credit_table(groups, cfg, gaps=GAPS):
    first = next(groups[(cfg, g)] for g in GAPS if (cfg, g) in groups)
    if not any(k.endswith("_corr_dw_visible") for k in first[0]):
        return ""
    rows = ["| readout | gap | corr(Δw, direction before the blank) | corr(Δw, direction in the state) | "
            "corr(Δw, direction in the outputs at the prediction) | corr(Δw, ridge weights) | "
            "eligibility from before the readout | Δw on the 10% most selective neurons before the blank |",
            "|---|---|---|---|---|---|---|---|"]
    for key in readouts(first)[1:]:
        for g in gaps:
            ms = groups.get((cfg, g))
            if not ms or f"{key}_corr_dw_visible" not in ms[0]:
                continue
            c = lambda m: cell(ms, f"{key}_{m}", bold=False, fmt="{:+.2f}") if m.startswith("corr") else \
                cell(ms, f"{key}_{m}", bold=False, fmt="{:.2f}")
            rows.append(f"| {label(key)} | {g} | {c('corr_dw_visible')} | {c('corr_dw_state')} | "
                        f"{c('corr_dw_readout')} | {c('corr_dw_ridge')} | {c('eligibility_pre_share')} | "
                        f"{c('dw_share_top_visible')} |")
    return "\n".join(rows)


def neurons_table(groups, cfg):
    rows = ["| gap | selective before the blank | selective in the state before the reappearance | "
            "selective in the outputs at the prediction |", "|---|---|---|---|"]
    for g in GAPS:
        ms = groups.get((cfg, g))
        if not ms or "neurons_selective_visible" not in ms[0]:
            continue
        rows.append(f"| {g} | {cell(ms, 'neurons_selective_visible', False, '{:.1f}')} | "
                    f"{cell(ms, 'neurons_selective_state', False, '{:.1f}')} | "
                    f"{cell(ms, 'neurons_selective_readout', False, '{:.1f}')} |")
    return "\n".join(rows) if len(rows) > 2 else ""


def paired(groups, cfg, a, b):
    out = []
    for g in GAPS:
        ms = groups.get((cfg, g))
        if not ms or a not in ms[0] or b not in ms[0]:
            continue
        d = np.array([m[a] - m[b] for m in ms])
        s = stat(d + 0.5)
        out.append(f"gap {g}: {s['mean'] - 0.5:+.3f} (99% lower {s['lower99'] - 0.5:+.3f}, "
                   f"{int(np.sum(d > 0))}/{len(d)})")
    return "; ".join(out)


def classes(groups):
    rows = ["| split | LEFT | RIGHT | trials (every configuration, gap and seed) |", "|---|---|---|---|"]
    allm = [m for ms in groups.values() for m in ms]
    for split in ("train", "val", "test"):
        left = {int(m[f"{split}_left"]) for m in allm}
        right = {int(m[f"{split}_right"]) for m in allm}
        rows.append(f"| {split} | {'/'.join(map(str, sorted(left)))} | {'/'.join(map(str, sorted(right)))} | "
                    f"{len(allm)} |")
    return "\n".join(rows)


def figure(groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    trace_colors = {"t0": "#8a8984", "t0.5": "#9ec5f0", "t0.8": "#2a78d6", "t0.9": "#1baf7a",
                    "t0.95": "#eda100", "t0.99": "#eb6834"}
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    x = np.arange(len(GAPS))

    def series(ax, cfg, metric, name, color, style="-"):
        means = []
        for g in GAPS:
            ms = groups.get((cfg, g))
            means.append(np.mean([m[metric] for m in ms]) if ms and metric in ms[0] else np.nan)
        if not np.all(np.isnan(means)):
            ax.plot(x, means, style, color=color, lw=2, marker="o", ms=5, label=name)

    for ax, cfg, title in ((axes[0], "E0, recovery 0.9", "E0, recovery 0.9 (video_memory baseline)"),
                           (axes[1], "E0, recovery 0.99", "E0, recovery 0.99")):
        series(ax, cfg, "probe_prediction_state", "probe: state at the prediction", "#4a3aa7", "--")
        series(ax, cfg, "ridge_accuracy", "ridge readout (offline)", ink, ":")
        for key, color in trace_colors.items():
            series(ax, cfg, f"{key}_accuracy", "online, " + ("no trace" if key == "t0" else f"trace {key[1:]}"),
                   color)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
    # Accuracy against the trace decay, per recovery, at gap 4.
    ax = axes[2]
    for rec, color in (("0.9", "#2a78d6"), ("0.95", "#1baf7a"), ("0.97", "#eda100"), ("0.99", "#eb6834")):
        ms = groups.get((f"E0, recovery {rec}", 4))
        if not ms:
            continue
        ks = [k for k in trace_colors if f"{k}_accuracy" in ms[0]]
        ax.plot(range(len(ks)), [np.mean([m[f"{k}_accuracy"] for m in ms]) for k in ks], "-o", color=color, lw=2,
                ms=5, label=f"recovery {rec}")
        ax.axhline(np.mean([m["ridge_accuracy"] for m in ms]), color=color, lw=1, ls=":")
    ax.set_xticks(range(len(TRACES)), TRACES)
    ax.set_xlabel("eligibility-trace decay per tick (dotted: ridge)", color=muted)
    ax.set_title("E0, gap 4: accuracy against the trace", loc="left", fontsize=11, color=ink)
    # Hidden learning.
    ax = axes[3]
    hid = sorted({c for c, g in groups if "hidden learns" in c})
    colors = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#008300"]
    for c, color in zip(hid, colors * 3):
        if "lr 0.01" not in c:
            continue
        series(ax, c, "probe_prediction_state", c.split("recovery ")[1].replace("hidden learns ", "") + ": probe",
               color, "--")
        series(ax, c, "t0.8_accuracy", c.split("recovery ")[1].replace("hidden learns ", "") + ": online 0.8",
               color)
    ax.set_title("Hidden layer learns too (lr 0.01)", loc="left", fontsize=11, color=ink)
    for i, ax in enumerate(axes):
        ax.axhline(0.5, color=muted, lw=1, ls=":")
        if i != 2:
            ax.set_xticks(x, [str(g) for g in GAPS])
            ax.set_xlabel("blank frames (gap)", color=muted)
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=7, labelcolor=muted)
    axes[0].set_ylabel("test accuracy (mean over seeds)", color=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--png")
    args = ap.parse_args()
    groups = load(args.files)
    cfgs = sorted({c for c, _ in groups}, key=lambda c: (("hidden" in c), ("error" in c), ("gain" in c), c))
    out = ["# video_memory_learned: summary\n",
           "Mean ± sd over seeds (variance = sd²); bold: one-sided 99% lower bound above 0.5. "
           "Horizon: the largest gap with that.\n",
           "## Class distribution (checked before training in every trial)\n", classes(groups) + "\n"]
    for cfg in cfgs:
        out.append(f"## {cfg}\n")
        out.append("### Test accuracy\n")
        out.append(accuracy_table(groups, cfg) + "\n")
        out.append("### Probes (network unchanged)\n")
        out.append(probe_table(groups, cfg) + "\n")
        first = next(groups[(cfg, g)] for g in GAPS if (cfg, g) in groups)
        keys = readouts(first)
        diffs = [f"- {label(k)} − immediate: {paired(groups, cfg, k + '_accuracy', 't0_accuracy')}"
                 for k in keys if k not in ("ridge", "t0") and "t0_accuracy" in first]
        diffs += [f"- {label(k)} − ridge: {paired(groups, cfg, k + '_accuracy', 'ridge_accuracy')}"
                  for k in keys if k != "ridge"]
        out.append("### Paired differences (same clips, seeds and initial weights)\n")
        out.append("\n".join(diffs) + "\n")
        n = neurons_table(groups, cfg)
        if n:
            out.append("### Neurons with |d'| > 0.5 for the direction (training clips; of 128)\n")
            out.append(n + "\n")
        c = credit_table(groups, cfg)
        if c:
            out.append("### Credit assignment (training clips; Δw = change of w_RIGHT − w_LEFT)\n")
            out.append(c + "\n")
        out.append("### Training\n")
        out.append(training_table(groups, cfg) + "\n")
        out.append("### Per class\n")
        out.append(class_table(groups, cfg) + "\n")
    text = "\n".join(out)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text)
    else:
        print(text)
    if args.png:
        figure(groups, args.png)


if __name__ == "__main__":
    main()
