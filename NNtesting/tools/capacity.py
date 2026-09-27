#!/usr/bin/env python3
"""Minimum architectures and capacity curves from nl_static results.

  capacity.py RESULTS.jsonl... [--out DIR] [--target 1e-3] [--fraction 0.8]

For every task (l0..l3, and l4 per K) and model, an architecture (depth x
width) counts as solving the task when at least --fraction of its seeds
reach test MSE <= --target (default: the success the runs recorded, i.e.
their own target_mse). When the runs swept several learning rates, each
model and architecture uses the one with the lowest median *validation*
MSE (the same rule for every model; the test set is never used to choose).

Reported per task and model, each found independently:
  min depth     smallest depth at which some width solves the task
  min width     smallest width at which some depth solves it
  min neurons   fewest neurons (hidden + output) of a solving architecture
  min params    fewest trainable parameters of a solving architecture
and, for the solving architecture with the fewest neurons: its median test
MSE, spikes per sample, active neurons per tick, event-driven synaptic
operations and inference time. Nothing is combined into a single score.

Writes (with --out): capacity.md (the tables), architectures.csv (every
task x model x architecture: chosen lr, seeds, success fraction, medians),
and capacity_neurons.svg / capacity_params.svg (l4: minimum size against K).

Standard library only.
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

MODELS_ORDER = ["relu", "er", "gate", "clamp", "linear"]


def task_label(params):
    t = params["task"]
    return f"l4 K={params['k']}" if t == "l4" else t


def task_sort_key(label):
    if label.startswith("l4"):
        return (4, int(label.split("=")[1]))
    return (int(label[1]), 0)


def load(paths):
    trials = defaultdict(list)  # (task, model, depth, width, lr) -> [metrics]
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:  # e.g. a line still being written
                    print(f"{path}:{number}: skipped, not JSON ({e})", file=sys.stderr)
                    continue
                if r.get("type") != "trial" or r.get("experiment") != "nl_static" or r.get("error"):
                    continue
                p = r["params"]
                model = "clamp" if p["model"] == "linear" else p["model"]
                key = (task_label(p), model, int(p["depth"]), int(p["width"]), float(p["lr"]))
                m = dict(r["metrics"])
                m["seed"] = r["seed"]
                m["target_mse"] = float(p["target_mse"])
                trials[key].append(m)
    return trials


def median(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(values) if values else float("nan")


def choose(trials, target, fixed_lr):
    """Per (task, model, depth, width): the chosen lr and its trials."""
    by_arch = defaultdict(dict)
    for (task, model, depth, width, lr), ms in trials.items():
        by_arch[(task, model, depth, width)][lr] = ms
    rows = []
    for (task, model, depth, width), lrs in by_arch.items():
        if fixed_lr is not None:
            if fixed_lr not in lrs:
                continue
            lr = fixed_lr
        else:
            lr = min(lrs, key=lambda x: (median([m["validation_mse"] for m in lrs[x]]), x))
        ms = lrs[lr]
        succ = [(m["test_mse"] <= (target if target is not None else m["target_mse"])) for m in ms]
        rows.append({
            "task": task, "model": model, "depth": depth, "width": width, "lr": lr, "seeds": len(ms),
            "success_fraction": sum(succ) / len(succ),
            "neurons": int(ms[0]["neurons"]), "parameters": int(ms[0]["parameters"]),
            "test_mse": median([m["test_mse"] for m in ms]),
            "test_nmse": median([m["test_nmse"] for m in ms]),
            "spikes_per_sample": median([m["spikes_per_sample"] for m in ms]),
            "active_neurons_mean": median([m["active_neurons_mean"] for m in ms]),
            "active_fraction": median([m["active_fraction"] for m in ms]),
            "event_synops_per_sample": median([m["event_synops_per_sample"] for m in ms]),
            "dense_synops_per_sample": median([m["dense_synops_per_sample"] for m in ms]),
            "inference_us": median([m["inference_us"] for m in ms]),
            "training_samples": median([m["training_samples"] for m in ms]),
        })
    return rows


def minima(rows, fraction):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["model"])].append(r)
    for key, rs in groups.items():
        ok = [r for r in rs if r["success_fraction"] >= fraction]
        best = min(rs, key=lambda r: (r["test_mse"], r["neurons"]))
        res = {"archs": len(rs), "solved": len(ok), "best": best}
        if ok:
            res["min_depth"] = min(r["depth"] for r in ok)
            res["min_width"] = min(r["width"] for r in ok)
            res["min_neurons"] = min(ok, key=lambda r: (r["neurons"], r["parameters"]))
            res["min_params"] = min(ok, key=lambda r: (r["parameters"], r["neurons"]))
        out[key] = res
    return out


def fmt(v, digits=3):
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.{digits}g}"
    return str(v)


def arch(r):
    return f"{r['depth']}×{r['width']}"


def report(rows, mins, fraction, target, fixed_lr):
    tasks = sorted({r["task"] for r in rows}, key=task_sort_key)
    models = [m for m in MODELS_ORDER if any(r["model"] == m for r in rows)]
    crit = f"test MSE ≤ {target:g}" if target is not None else "test MSE ≤ the runs' target_mse"
    lines = [
        "# Minimum architectures (nl_static)",
        "",
        f"Solved: {crit} in ≥ {fraction:.0%} of seeds. Learning rate: "
        + (f"fixed at {fixed_lr:g}." if fixed_lr is not None else "per model and architecture, lowest median validation MSE."),
        "",
        "## Minimum size per task",
        "",
        "| Task | Model | Solved archs | Min depth | Min width | Min neurons (arch) | Min params (arch) |",
        "|------|-------|--------------|-----------|-----------|--------------------|-------------------|",
    ]
    for t in tasks:
        for m in models:
            res = mins.get((t, m))
            if not res:
                continue
            if res["solved"]:
                n, p = res["min_neurons"], res["min_params"]
                lines.append(f"| {t} | {m} | {res['solved']}/{res['archs']} | {res['min_depth']} | {res['min_width']} | "
                             f"{n['neurons']} ({arch(n)}) | {p['parameters']} ({arch(p)}) |")
            else:
                lines.append(f"| {t} | {m} | 0/{res['archs']} | – | – | – | – |")
    lines += [
        "",
        "## At the smallest solving architecture (fewest neurons)",
        "",
        "| Task | Model | Arch | lr | Test MSE | Spikes/sample | Active/tick | Event synops/sample | Inference µs |",
        "|------|-------|------|----|----------|---------------|-------------|---------------------|--------------|",
    ]
    for t in tasks:
        for m in models:
            res = mins.get((t, m))
            if not res or not res["solved"]:
                continue
            r = res["min_neurons"]
            lines.append(f"| {t} | {m} | {arch(r)} | {r['lr']:g} | {fmt(r['test_mse'])} | {fmt(r['spikes_per_sample'])} | "
                         f"{fmt(r['active_neurons_mean'])} | {fmt(r['event_synops_per_sample'])} | {fmt(r['inference_us'])} |")
    lines += [
        "",
        "## Best test MSE per task (any architecture, median over seeds)",
        "",
        "| Task | " + " | ".join(models) + " |",
        "|------|" + "|".join("---" for _ in models) + "|",
    ]
    for t in tasks:
        cells = []
        for m in models:
            res = mins.get((t, m))
            cells.append(f"{fmt(res['best']['test_mse'])} ({arch(res['best'])})" if res else "–")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def svg_curve(mins, field, title, path):
    ks = sorted({int(t.split("=")[1]) for (t, _) in mins if t.startswith("l4")})
    models = [m for m in MODELS_ORDER if any(mm == m for (_, mm) in mins)]
    if not ks:
        return
    points = {}
    for m in models:
        pts = []
        for k in ks:
            res = mins.get((f"l4 K={k}", m))
            if res and res["solved"]:
                pts.append((k, res["min_neurons"]["neurons"] if field == "neurons" else res["min_params"]["parameters"]))
        points[m] = pts
    values = [v for pts in points.values() for _, v in pts] or [1]
    lo, hi = math.log2(max(1, min(values))) - 0.5, math.log2(max(values)) + 0.5
    W, H, L, B = 560, 360, 70, 50
    x = lambda k: L + (W - L - 20) * (ks.index(k) / max(1, len(ks) - 1))
    y = lambda v: H - B - (H - B - 30) * ((math.log2(v) - lo) / max(1e-9, hi - lo))
    colors = {"relu": "#1f77b4", "er": "#d62728", "gate": "#2ca02c", "clamp": "#7f7f7f", "linear": "#7f7f7f"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W/2}" y="18" text-anchor="middle">{title}</text>',
           f'<line x1="{L}" y1="{H-B}" x2="{W-20}" y2="{H-B}" stroke="black"/>',
           f'<line x1="{L}" y1="30" x2="{L}" y2="{H-B}" stroke="black"/>',
           f'<text x="{(L+W)/2}" y="{H-12}" text-anchor="middle">K (l4 components)</text>']
    for k in ks:
        out.append(f'<text x="{x(k)}" y="{H-B+16}" text-anchor="middle">{k}</text>')
    e = math.ceil(lo)
    while e <= hi:
        out.append(f'<text x="{L-6}" y="{y(2**e)+4}" text-anchor="end">{2**e}</text>')
        out.append(f'<line x1="{L}" y1="{y(2**e)}" x2="{W-20}" y2="{y(2**e)}" stroke="#eee"/>')
        e += 1
    for i, m in enumerate(models):
        pts = points[m]
        c = colors.get(m, "black")
        if len(pts) > 1:
            out.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'
                       % (c, " ".join(f"{x(k)},{y(v)}" for k, v in pts)))
        for k, v in pts:
            out.append(f'<circle cx="{x(k)}" cy="{y(v)}" r="3.5" fill="{c}"/>')
        out.append(f'<text x="{W-110}" y="{40+16*i}" fill="{c}">{m} ({len(pts)}/{len(ks)} solved)</text>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", help="directory for capacity.md, architectures.csv and the SVG curves")
    ap.add_argument("--target", type=float, help="success threshold on test MSE (default: each run's target_mse)")
    ap.add_argument("--fraction", type=float, default=0.8, help="fraction of seeds that must succeed (default 0.8)")
    ap.add_argument("--lr", type=float, help="use this learning rate only instead of choosing by validation")
    a = ap.parse_args()
    trials = load(a.files)
    if not trials:
        sys.exit("no nl_static trials found")
    rows = choose(trials, a.target, a.lr)
    mins = minima(rows, a.fraction)
    text = report(rows, mins, a.fraction, a.target, a.lr)
    print(text)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "capacity.md"), "w", encoding="utf-8") as f:
            f.write(text)
        rows.sort(key=lambda r: (task_sort_key(r["task"]), MODELS_ORDER.index(r["model"]), r["depth"], r["width"]))
        with open(os.path.join(a.out, "architectures.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        svg_curve(mins, "neurons", "Minimum neurons to solve l4 (sum of K sines)", os.path.join(a.out, "capacity_neurons.svg"))
        svg_curve(mins, "params", "Minimum parameters to solve l4 (sum of K sines)", os.path.join(a.out, "capacity_params.svg"))


if __name__ == "__main__":
    main()
