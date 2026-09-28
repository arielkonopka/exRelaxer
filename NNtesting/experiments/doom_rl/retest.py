#!/usr/bin/env python3
"""Retests the best configs a search found, on fresh seeds and more test
episodes, each against the same network untrained (train=0): how much of
its score is learning and how much E-R's own activity.

    python3 NNtesting/experiments/doom_rl/retest.py --search results/doom-search/defend_the_center \\
        --top 5 --seeds 10,11,12 --test 30 --out retest.jsonl
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search  # noqa: E402


def top_configs(path, top):
    rows = []
    with open(os.path.join(path, "evals.jsonl")) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rungs = max(r["rung"] for r in rows)
    by, cfg, scenario = collections.defaultdict(list), {}, rows[0]["scenario"]
    for r in rows:
        if r["rung"] == rungs and "error" not in r["result"]:
            by[r["key"]].append(r["result"]["reward"])
            cfg[r["key"]] = r["config"]
    full = [k for k, v in by.items() if len(v) >= rungs + 1]
    seen, out = set(), []
    for k in sorted(full, key=lambda k: -st.mean(by[k])):
        sig = json.dumps(cfg[k], sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            out.append((k, cfg[k], st.mean(by[k])))
        if len(out) == top:
            break
    train = max(r["train"] for r in rows)
    return out, scenario, train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seeds", default="10,11,12")
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--scenario", default="", help="default: the search's")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    configs, scenario, train = top_configs(args.search, args.top)
    scenario = args.scenario or scenario
    jobs = []
    for k, config, score in configs:
        for seed in [int(s) for s in args.seeds.split(",")]:
            for budget in (train, 0):
                jobs.append({"config": config, "key": k, "rung": -1, "seed": seed, "scenario": scenario,
                             "train": budget, "test": args.test, "search_score": score, "save": None})
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool, open(args.out, "a") as f:
        for r in pool.imap_unordered(search.evaluate, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            res = r["result"]
            print(f"{r['key']} seed {r['seed']} train {r['train']}: reward {res.get('reward', float('nan')):.2f} "
                  f"kills {res.get('kills', 0):.2f}", flush=True)


if __name__ == "__main__":
    main()
