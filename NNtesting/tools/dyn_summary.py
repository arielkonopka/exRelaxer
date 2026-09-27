#!/usr/bin/env python3
"""Summarises dyn_ladder and doom results: for each task, model, frame
window and width, picks the learning rate with the best median validation
score (doom: training win rate; the test set is never used to choose), then
prints the median test score over seeds, its range, and activity.

    python3 NNtesting/tools/dyn_summary.py results.jsonl [more.jsonl ...]
"""
import collections
import json
import statistics as st
import sys


def load(paths):
    runs = collections.defaultdict(list)
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line still being written
                if r.get("type") != "trial":
                    continue
                p, m = r["params"], r["metrics"]
                task = p.get("task") or p.get("scenario")
                key = (r["experiment"], task, p["model"], int(p["window"]), int(p["width"]))
                runs[key, p["lr"]].append(m)
    return runs


def main(paths):
    runs = load(paths)
    best = {}
    for (key, lr), ms in runs.items():
        choose = "validation_score" if key[0] == "dyn_ladder" else "train_win_rate"
        v = st.median(m[choose] for m in ms)
        if key not in best or v > best[key][0]:
            best[key] = (v, lr, ms)
    print(f"{'task':17} {'model':13} {'win':>3} {'width':>5} {'lr':>6} {'test':>6} {'min':>6} {'max':>6} "
          f"{'spikes':>7}  reference")
    for key in sorted(best):
        _, lr, ms = best[key]
        score = "test_score" if key[0] == "dyn_ladder" else "win_rate"
        s = [m[score] for m in ms]
        ref = ""
        for name in ("input_ceiling_accuracy", "chase_catch_rate", "oracle_agreement", "oracle_win_rate",
                     "oracle_no_lead_win_rate"):
            vals = [m[name] for m in ms if name in m]
            if vals:
                ref += f" {name.replace('_accuracy', '').replace('_rate', '')}={st.median(vals):.3f}"
        spikes = st.median(m["spikes_per_step"] for m in ms)
        print(f"{key[1]:17} {key[2]:13} {key[3]:3} {key[4]:5} {lr:>6} {st.median(s):6.3f} {min(s):6.3f} "
              f"{max(s):6.3f} {spikes:7.1f} {ref}  (n={len(s)})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
