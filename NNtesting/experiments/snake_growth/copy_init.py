"""Copy-init for width growth (research log §35): new neurons copy an existing neuron
(Net2Net-style widening, WidthGrowth init="copy") instead of starting from random weights; and
grown neurons that start easier to fire (§38: a lower resting threshold, `rest*` arms, or a
threshold at the minimum right after growth, `start0` arms).

    python NNtesting/experiments/snake_growth/copy_init.py OUT_DIR                  # every setup
    python NNtesting/experiments/snake_growth/copy_init.py OUT_DIR --summary        # the table

Each run writes OUT_DIR/<setup>_s<seed>.json: the experiment's metrics, score curve, final sizes,
event log, and per width growth the evaluation it happened in and the scores of the windows just
before and just after it.
"""
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment as sg  # this folder's experiment.py

ARMS = {
    "fixed": {"grow": False},
    "random": {},                                                    # the current init
    "copy_fatigued": {"growth_init": "copy"},                        # Net2Net: split 0.5, noise 0.05
    "copy_active": {"growth_init": "copy", "copy_source": "active"},
    "copy_read": {"growth_init": "copy", "copy_source": "read"},
    "copy_random": {"growth_init": "copy", "copy_source": "random"},
    "copy_nosplit": {"growth_init": "copy", "copy_split": 0.0},      # incoming copied, starts unread
    "copy_takeover": {"growth_init": "copy", "copy_split": 1.0},     # the copy takes all the reading
    "copy_noise02": {"growth_init": "copy", "copy_noise": 0.2},
    "copy_noise05": {"growth_init": "copy", "copy_noise": 0.5},
    "copy_noise10": {"growth_init": "copy", "copy_noise": 1.0},
    # §38: grown neurons start easier to fire.
    "random_start0": {"grown_start": 0.0},
    "random_rest05": {"grown_resting": 0.5},
    "random_rest01": {"grown_resting": 0.1},
    "random_rest025": {"grown_resting": 0.25},
    "random_rest075": {"grown_resting": 0.75},
    "random_rest01_start0": {"grown_resting": 0.1, "grown_start": 0.0},
    "copy_fatigued_start0": {"growth_init": "copy", "grown_start": 0.0},
    "copy_fatigued_rest01": {"growth_init": "copy", "grown_resting": 0.1},
    "copy_fatigued_rest01_start0": {"growth_init": "copy", "grown_resting": 0.1, "grown_start": 0.0},
    "copy_active_start0": {"growth_init": "copy", "copy_source": "active", "grown_start": 0.0},
    "copy_active_rest01": {"growth_init": "copy", "copy_source": "active", "grown_resting": 0.1},
}
STARTS = {
    "one": {"start": 1},
    "four": {"start": 4},
    # §37's retuned trigger at the default recovery: 25% saturated ticks, 4 neurons per growth.
    "one05t": {"start": 1, "recovery": 0.5, "saturated_share": 0.25, "increment": 4},
}


def setups():
    return {f"{s}_{a}": dict(sp, **ap) for s, sp in STARTS.items() for a, ap in ARMS.items()}


def growth_windows(log, curve):
    """(evaluation, score before, score after) per width growth, from the event log."""
    out, k = [], 0
    for line in log:
        if line.startswith("evaluation "):
            k = int(line.split()[1].rstrip(":")) + 1
        elif re.search(r"saturated, width", line):
            # Growth during window k: compare window k-1 (before) with k+1 (the first whole window after).
            before = curve[k - 1] if k >= 1 else None
            after = curve[k + 1] if k + 1 < len(curve) else None
            out.append((k, before, after))
    return out


def job(args):
    out, name, seed, extra = args
    p = dict(sg.DEFAULTS, **sg.OLD_TRIGGER, **setups()[name], **extra)
    log = []
    agent, m = sg.develop(p, seed, report=log.append)
    m = {k: (v if k == "diversity" else [float(x) for x in v] if isinstance(v, list) else float(v))
         for k, v in m.items()}
    m.update(setup=name, seed=seed, params=p, sizes=agent.sizes(), log=log,
             growth=growth_windows(log, m["curve"]), copies=agent.width.copies)
    (Path(out) / f"{name}_s{seed}.json").write_text(json.dumps(m))
    return name, seed, m["final"]


def summary(out):
    runs = {}
    for f in sorted(Path(out).glob("*.json")):
        m = json.loads(f.read_text())
        runs.setdefault(m["setup"], []).append(m)
    print("| setup | seeds | final (last 3 windows) | mean of all windows | best window | neurons | widths | "
          "undos+prunes | change across a growth |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in setups():
        ms = runs.get(name)
        if not ms:
            continue
        final = np.array([m["final"] for m in ms])
        area = np.array([np.mean(m["curve"]) for m in ms])
        best = np.array([m["best_window"] for m in ms])
        size = np.array([m["size"] for m in ms])
        widths = np.array([m["width_events"] for m in ms])
        prunes = np.array([m["prune_events"] for m in ms])
        deltas = [a - b for m in ms for _, b, a in m["growth"] if a is not None and b is not None]
        se = final.std(ddof=1) / np.sqrt(len(final)) if len(final) > 1 else 0.0
        delta = f"{np.mean(deltas):+.2f} (n={len(deltas)})" if deltas else "-"
        print(f"| {name} | {len(ms)} | {final.mean():.2f} ± {se:.2f} | {area.mean():.2f} | {best.mean():.2f} | "
              f"{size.mean():.1f} | {widths.mean():.1f} | {prunes.mean():.1f} | {delta} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--setups", default=",".join(setups()))
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=None, help="games per run (default: the experiment's)")
    parser.add_argument("--recovery", type=float, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.out)
    else:
        extra = {k: v for k, v in (("games", args.games), ("recovery", args.recovery)) if v is not None}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        jobs = [(args.out, n, int(s), extra) for s in args.seeds.split(",") for n in args.setups.split(",")]
        with ProcessPoolExecutor(args.workers) as pool:
            for name, seed, final in pool.map(job, jobs):
                print(name, seed, round(final, 2), flush=True)
        summary(args.out)
