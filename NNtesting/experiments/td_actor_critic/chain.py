#!/usr/bin/env python3
"""Validation step 1: a cue, then a reward N steps later. Does the TD critic
learn V(s) over the reservoir's features, and does the TD error move from
the reward to the cue (the "dopamine shift")?

One trial: a blank wait of 5..15 steps (so the cue's time cannot be
foreseen), the cue for one step, N blank steps, then reward 1 on the step
that ends the trial. With --catch, that share of trials has neither cue
nor reward, so the blank wait itself predicts less. The critic sees only the features; no actor.

    chain.py --out DIR [--delays 5,10,20,40] [--trials 400] [--repr er,csc]

Representations:
  csc     a tapped delay line: feature k is 1 exactly k steps after the cue
          (the textbook "complete serial compound"; checks TD itself)
  er      the frozen E-R reservoir's outputs and thresholds (common.Reservoir)
  er_out  its outputs only
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exrelaxer as exr
from exrelaxer.actor_critic import Critic
from common import Reservoir


def run(rep, delay, trials, seed, args):
    rng = np.random.default_rng(seed)
    exr.reseed(seed)
    if rep == "csc":
        size = 64
        res = None
    else:
        res = Reservoir({"cue": 1}, width=args.width, recovery=args.recovery, spread=args.spread,
                        features="both" if rep == "er" else "output")
        size = res.size
    critic = Critic(size, args.gamma, args.lam, args.rate, bias=args.bias)
    log = []
    for trial in range(trials):
        wait = int(rng.integers(args.wait_min, args.wait_max + 1))
        catch = rng.random() < args.catch  # a catch trial: no cue, no reward
        if res is not None and args.reset:
            res.reset()
        critic.reset()
        deltas, values = [], []
        steps = wait + 1 + delay
        for t in range(steps):
            cue = 1.0 if t == wait and not catch else 0.0
            since = t - wait
            terminal = t == steps - 1
            if res is None:
                x = np.zeros(size)
                if 0 <= since < size and not catch:
                    x[since] = 1.0
            else:
                x = res.step({"cue": [cue]})
            r = 1.0 if terminal and not catch else 0.0
            d, v = critic.step(x, r, terminal)
            deltas.append(d)
            values.append(v)
        if catch:
            continue
        log.append({"v_pre": values[wait - 1],"trial": trial, "delta_cue": deltas[wait], "delta_reward": deltas[-1],
                    "delta_pre": float(np.mean(np.abs(deltas[1:wait]))) if wait > 1 else 0.0,
                    "values": values[wait:]})
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--delays", default="5,10,20,40")
    ap.add_argument("--repr", default="csc,er,er_out")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--rate", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--recovery", type=float, default=0.9)
    ap.add_argument("--spread", type=float, default=0.0, help="recovery drawn from recovery +- spread")
    ap.add_argument("--wait-min", type=int, default=5)
    ap.add_argument("--wait-max", type=int, default=15)
    ap.add_argument("--catch", type=float, default=0.0, help="share of trials without cue and reward")
    ap.add_argument("--reset", type=int, default=1, help="reset the reservoir's state between trials (0: one continuous stream)")
    ap.add_argument("--bias", type=int, default=0, help="the critic learns a bias (V of the blank wait then is not 0)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    exr.set_threads(1)
    summary = []
    for rep in args.repr.split(","):
        for delay in map(int, args.delays.split(",")):
            rows = []
            for s in range(args.seeds):
                log = run(rep, delay, args.trials, s + 1, args)
                first, last = log[:5], log[-50:]
                v_end = np.mean([l["values"] for l in last], axis=0)
                rows.append({
                    "delta_cue_first": float(np.mean([l["delta_cue"] for l in first])),
                    "delta_cue_last": float(np.mean([l["delta_cue"] for l in last])),
                    "delta_reward_first": float(np.mean([l["delta_reward"] for l in first])),
                    "delta_reward_last": float(np.mean([l["delta_reward"] for l in last])),
                    "delta_pre_last": float(np.mean([l["delta_pre"] for l in last])),
                    "v_pre_last": float(np.mean([l["v_pre"] for l in last])),
                    "v_after_cue": float(v_end[0]), "v_before_reward": float(v_end[-2]),
                    "v_min_between": float(v_end[:-1].min()),
                    "v_trace": [round(float(v), 4) for v in v_end],
                    "curve_cue": [round(l["delta_cue"], 4) for l in log],
                    "curve_reward": [round(l["delta_reward"], 4) for l in log],
                })
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if not isinstance(rows[0][k], list)}
            ideal = args.gamma ** delay  # V right after the cue when V is exact
            agg.update(repr=rep, delay=delay, ideal_cue=ideal,
                       shift=bool(agg["delta_reward_last"] < 0.5 * agg["delta_reward_first"] and
                                  agg["delta_cue_last"] > 0.5 * ideal))
            summary.append(agg)
            with open(os.path.join(args.out, f"{rep}_N{delay}.json"), "w") as f:
                json.dump({"summary": agg, "seeds": rows}, f)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
