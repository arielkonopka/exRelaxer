#!/usr/bin/env python3
"""Tables of the online-learning comparison (NNtesting/experiments/goe_rl/online.py).

    cd results/goe/rules && python3 summarize.py sweep.jsonl.gz final.jsonl.gz final_online.jsonl.gz > summary.md

Final test: 200 mazes and 100 rooms (the G3 fresh worlds among them), every world from a fresh
network with the learned weights. Per technique: the mean over its seeds of the mean reward per
maze, per room and per world, the paired difference to the frozen start per world (each world's
reward averaged over seeds, minus the start's on the same world) with a bootstrap 90 % interval,
and the G3 subset (the 30 mazes of the earlier test).
"""
import collections
import gzip
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "..",
                                               "NNtesting", "experiments", "goe_rl"))
import online as O  # noqa: E402


def boot(d, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return np.percentile(m, 5), np.percentile(m, 95)


def read(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return [json.loads(l) for l in f]


def main():
    sweep, final = read(sys.argv[1]), read(sys.argv[2])
    print("## Rate sweep (validation: 40 mazes, 50 rooms; one training stream of 300 worlds)\n")
    rates = sorted({l["rate"] for l in sweep if l["technique"] != "frozen"})
    print("| technique | " + " | ".join(f"{r:g}" for r in rates) + " |")
    print("|---|" + "---|" * len(rates))
    by = collections.defaultdict(dict)
    for l in sweep:
        by[l["technique"]][l["rate"]] = l
    for t, d in by.items():
        cells = []
        for r in rates:
            if t == "frozen":  # no rate: the start, in the first column
                l = d[0.0]
                if r != rates[0]:
                    cells.append("")
                    continue
            elif r not in d:
                cells.append("")
                continue
            else:
                l = d[r]
            s = l["valid"]["summary"]
            cells.append(f"{s['maze']:.1f} / {s['rooms']:.1f}")
        print(f"| {t} | " + " | ".join(cells) + " |")
    print("\nCells: mean reward per maze / per room.\n")

    print("## Final test (200 mazes, 100 rooms; 3 training seeds)\n")
    runs = collections.defaultdict(list)
    for l in final:
        if "test" in l:
            runs[l["technique"]].append(l)
    worlds = O.TEST
    maze = np.array([k == "maze" for k, _ in worlds])
    g3 = np.array([w in O.G3 and k == "maze" for w, (k, _) in zip(worlds, worlds)])
    start = np.array(runs["frozen"][0]["test"]["rewards"])
    rows = []
    for t, ls in runs.items():
        R = np.array([l["test"]["rewards"] for l in ls])
        per = R.mean(axis=0)
        diff = per - start
        lo, hi = boot(diff)
        rate = ls[0].get("rate", 0.0)
        seeds = " ".join(f"{r[maze].mean():.1f}" for r in R)
        rows.append((per.mean(), t, rate, per[maze].mean(), per[~maze].mean(), per[g3].mean(), diff.mean(), lo, hi,
                     seeds, np.mean([l.get("drift", {}).get("readouts", 0.0) for l in ls])))
    print("| technique | rate | per maze | per room | per world | vs start (90 % CI) | G3 30 mazes | maze per seed | readout drift |")
    print("|---|---|---|---|---|---|---|---|---|")
    for w, t, rate, m, r, g, d, lo, hi, seeds, drift in sorted(rows, reverse=True):
        print(f"| {t} | {rate:g} | {m:.1f} | {r:.2f} | {w:.2f} | {d:+.2f} ({lo:+.2f}, {hi:+.2f}) | {g:.1f} | {seeds} | {drift:.3g} |")
    if len(sys.argv) > 3:
        print("\n## Learning on inside each test world (each world starts from the trained weights)\n")
        print("| technique | seed | off: maze / room / world | on: maze / room / world | on - off per world |")
        print("|---|---|---|---|---|")
        for l in sorted(read(sys.argv[3]), key=lambda l: (l["technique"], l["seed"])):
            a, b = np.array(l["test"]["rewards"]), np.array(l["test_online"]["rewards"])
            print(f"| {l['technique']} | {l['seed']} | {a[maze].mean():.1f} / {a[~maze].mean():.2f} / {a.mean():.2f} | "
                  f"{b[maze].mean():.1f} / {b[~maze].mean():.2f} / {b.mean():.2f} | {(b - a).mean():+.2f} |")


if __name__ == "__main__":
    main()
