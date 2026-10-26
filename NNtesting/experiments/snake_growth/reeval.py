"""Growth re-evaluated on the current library (results/growth-reeval/, research log §36).

§30 and §31 ran at recovery 0.9 (or a few fixed values), with 3 seeds, the default growth trigger,
and the long runs started from 16 neurons only. This script repeats them where the library has
moved on and takes the mechanism further:

    short  §30's 1,500-game sweep at recovery 0.5 / 0.75 / 0.9, 10 seeds, ER(1/4/8/16) fixed and
           ER(1/4) growing; the growth trigger (saturated share, window, increment) at 0.5; 8-bit
           growing layers
    long   long_run.py from one neuron and from four with the whole mechanism (caps lifted,
           freeze_from=4, wide depth growth, weak pruning) at recovery 0.5 / 0.75 / 0.9

    python NNtesting/experiments/snake_growth/reeval.py short OUT_DIR [--seeds 1-10] [--setups a,b]
    python NNtesting/experiments/snake_growth/reeval.py long OUT_DIR [--seeds 1-5]
    python NNtesting/experiments/snake_growth/reeval.py summary OUT_DIR

Each run writes OUT_DIR/<setup>_s<seed>.json in the format of sweep.py (short) or long_run.py
(long). `summary` prints a Markdown table of every setup found in OUT_DIR.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import long_run
import sweep

REC = {"05": 0.5, "075": 0.75, "09": 0.9}

SHORT = {}
for tag, r in REC.items():
    SHORT[f"one_fixed_rec{tag}"] = {"start": 1, "grow": False, "recovery": r}
    SHORT[f"one_grow_rec{tag}"] = {"start": 1, "recovery": r}
    SHORT[f"four_fixed_rec{tag}"] = {"start": 4, "grow": False, "recovery": r}
    SHORT[f"four_grow_rec{tag}"] = {"start": 4, "recovery": r}
    SHORT[f"eight_fixed_rec{tag}"] = {"start": 8, "grow": False, "recovery": r}
    SHORT[f"sixteen_fixed_rec{tag}"] = {"start": 16, "grow": False, "recovery": r}
    SHORT[f"one_grow_int8_rec{tag}"] = {"start": 1, "recovery": r, "precision": "int8"}
# The growth trigger at the new default recovery: a lower saturated share, a shorter window, more
# neurons per growth.
for share in (0.1, 0.25):
    SHORT[f"one_grow_rec05_share{int(share * 100):02d}"] = {"start": 1, "recovery": 0.5, "saturated_share": share}
    SHORT[f"four_grow_rec05_share{int(share * 100):02d}"] = {"start": 4, "recovery": 0.5, "saturated_share": share}
SHORT["one_grow_rec05_window50"] = {"start": 1, "recovery": 0.5, "window": 50}
SHORT["one_grow_rec05_inc4"] = {"start": 1, "recovery": 0.5, "increment": 4}
SHORT["one_grow_rec05_share25_inc4"] = {"start": 1, "recovery": 0.5, "saturated_share": 0.25, "increment": 4}

FULL = dict(long_run.OPEN, freeze_from=4, deep_size=0, deep_minimum=3, prune_base=True, prune_weak=True)
LONG = {}
for tag, r in REC.items():
    for start, name in ((1, "s1"), (4, "s4")):
        LONG[f"{name}_open_rec{tag}"] = dict(long_run.OPEN, start=start, recovery=r)
        LONG[f"{name}_full_rec{tag}"] = dict(FULL, start=start, recovery=r)
    LONG[f"s16_full_rec{tag}"] = dict(FULL, start=16, recovery=r)
LONG["s1_full_rec05_share25"] = dict(FULL, start=1, recovery=0.5, saturated_share=0.25)
LONG["s1_full_rec05_share10"] = dict(FULL, start=1, recovery=0.5, saturated_share=0.1)
LONG["s1_full_rec05_share25_inc4"] = dict(FULL, start=1, recovery=0.5, saturated_share=0.25, increment=4)
LONG["s1_open_rec05_share25_inc4"] = dict(long_run.OPEN, start=1, recovery=0.5, saturated_share=0.25, increment=4)


def seeds(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def run(mode, out, names, seed_list, workers, stable, max_games):
    Path(out).mkdir(parents=True, exist_ok=True)
    if mode == "short":
        sweep.SETUPS.update(SHORT)
        jobs = [(out, n, s) for s in seed_list for n in names
                if not (Path(out) / f"{n}_s{s}.json").exists()]
        fn = sweep.job
    else:
        long_run.SETUPS.update(LONG)
        jobs = [(out, n, s, stable, max_games) for n in names for s in seed_list
                if not (Path(out) / f"{n}_s{s}.json").exists()]
        fn = long_run.job
    with ProcessPoolExecutor(workers) as pool:
        for result in pool.map(fn, jobs):
            print(*result, flush=True)


def summary(out):
    runs = {}
    for f in sorted(Path(out).glob("*_s*.json")):
        d = json.loads(f.read_text())
        runs.setdefault(d["setup"], []).append(d)
    if not runs:
        return
    if "rows" in next(iter(runs.values()))[0]:
        print("| setup | seeds | final sizes (total) | largest | games | last 500 games, apples/game "
              "| width / undo / depth / prune events | last size change at game |")
        print("|---|---|---|---|---|---|---|---|")
        for name, ds in runs.items():
            ds.sort(key=lambda d: d["seed"])
            scores = [np.mean([r["score"] for r in d["rows"][-10:]]) for d in ds]
            sizes = [d["rows"][-1]["sizes"] for d in ds]
            largest = max(sum(r["sizes"]) for d in ds for r in d["rows"])
            changes = []
            for d in ds:
                last = "-"
                for a, b in zip(d["rows"], d["rows"][1:]):
                    if a["sizes"] != b["sizes"]:
                        last = str(b["games"])
                changes.append(last)
            ev = [sum(len(d[k]) for d in ds) for k in ("width_events", "undo_events", "depth_events", "prune_events")]
            sz = " ".join(f"{s}({sum(s)})" for s in sizes)
            print(f"| {name} | {len(ds)} | {sz} | {largest} | {int(np.mean([d['rows'][-1]['games'] for d in ds]))} "
                  f"| {np.mean(scores):.2f} ± {np.std(scores, ddof=1) / np.sqrt(len(ds)) if len(ds) > 1 else 0:.2f} "
                  f"({min(scores):.2f}-{max(scores):.2f}) | {' / '.join(map(str, ev))} | {'/'.join(changes)} |")
    else:
        print("| setup | seeds | final apples/game (mean ± se, min-max) | best window | final sizes "
              "| width / prune / depth events |")
        print("|---|---|---|---|---|---|")
        for name, ds in runs.items():
            f = np.array([d["final"] for d in ds])
            se = f.std(ddof=1) / np.sqrt(len(f)) if len(f) > 1 else 0.0
            sizes = {}
            for d in ds:
                sizes[str(d["sizes"])] = sizes.get(str(d["sizes"]), 0) + 1
            sz = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]))
            ev = [int(sum(d[k] for d in ds)) for k in ("width_events", "prune_events", "depth_events")]
            print(f"| {name} | {len(ds)} | {f.mean():.2f} ± {se:.2f} ({f.min():.2f}-{f.max():.2f}) "
                  f"| {max(d['best_window'] for d in ds):.2f} | {sz} | {' / '.join(map(str, ev))} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("short", "long", "summary"))
    parser.add_argument("out")
    parser.add_argument("--setups", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stable", type=int, default=200)
    parser.add_argument("--max-games", type=int, default=50_000)
    args = parser.parse_args()
    if args.mode == "summary":
        summary(args.out)
    else:
        table = SHORT if args.mode == "short" else LONG
        names = args.setups.split(",") if args.setups else list(table)
        run(args.mode, args.out, names, seeds(args.seeds or ("1-10" if args.mode == "short" else "1-5")),
            args.workers, args.stable, args.max_games)
