#!/usr/bin/env python3
"""Validation step 2: Snake, learned by an actor-critic whose modulator is the
TD error (delta) or, as the control, the raw reward. Same network, games and
seeds for both, and the original experiment (snake_py: sign readouts,
error-driven) as the reference (13.6 apples per test game in the research log).

    state (23) -> eye (pass-through) -> mix (64 frozen random E-R)
    features = state + mix outputs (+ mix thresholds) -> critic V(s), 3 actor readouts

    snake_td.py --out FILE.jsonl --arm td|reward|original [--seeds 1-5] [--set key=value ...]

Each line of the output: arm, seed, settings, test apples/steps, training curve (apples per
block of 20 games), mean |delta|, action entropy, clamped share.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "snake"))
import exrelaxer as exr
from exrelaxer.actor_critic import Agent
import game as snake  # snake/game.py
import experiment as original  # snake/experiment.py (snake_py)
from common import recovery_jitter

DEFAULTS = {
    "width": 10, "height": 10, "mix": 64, "train": 200, "test": 50,
    "eat": 1.0, "die": -1.0, "approach": 0.1,
    "features": "output",       # mix features: output, threshold, both
    "recovery": 0.5, "spread": 0.0,
    "gamma": 0.97, "lam": 0.9, "critic_rate": 0.1, "actor_rate": 0.02, "clip": 1.0,
    "post": "taken", "select": "softmax", "tau": 0.1, "epsilon": 0.05,
}


class Features:
    def __init__(self, p):
        self.net = net = exr.Network()
        state = exr.Shape(1, 1, snake.STATE_SIZE)
        eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(state), False, False))
        net.add_inputs(eye, state)
        spec = exr.LayerSpec.dense(p["mix"], False, False, frozen=True)
        spec.recovery_jitter = recovery_jitter(p["recovery"], p["spread"])
        self.mix = net.add_layer("mix", spec)
        net.connect(eye, self.mix)
        net.add_output(self.mix)
        self.kind = p["features"]
        self.size = snake.STATE_SIZE + p["mix"] * (2 if self.kind == "both" else 1)

    def __call__(self, state):
        self.net.set_inputs(state)
        self.net.step()
        parts = [np.asarray(state, dtype=np.float64)]
        if self.kind in ("output", "both"):
            parts.append(np.asarray(self.net.layer_output(self.mix), dtype=np.float64).ravel())
        if self.kind in ("threshold", "both"):
            parts.append(np.asarray(self.net.neuron_state(self.mix)["threshold"], dtype=np.float64))
        return np.concatenate(parts)


def reward_of(p, outcome, before, after):
    if outcome == snake.ATE:
        return p["eat"]
    if outcome == snake.DIED:
        return p["die"]
    if outcome == snake.STARVED:
        return 0.0
    return p["approach"] if after < before else -p["approach"]


def play(feats, agent, p, seed, games, learn):
    results, summaries = [], []
    for i in range(games):
        g = snake.Game(p["width"], p["height"], (seed * 1000003 + i) & snake.MASK)
        agent.log.new()
        agent.end_episode()
        a = agent.step(feats(g.state()), 0.0, False, learn, learn)
        outcome = snake.MOVED
        while True:
            before = g.apple_distance()
            outcome = g.step(a)
            r = reward_of(p, outcome, before, g.apple_distance())
            a = agent.step(feats(g.state()), r, g.over, learn, learn)
            if g.over:
                break
        results.append((g.score, g.steps, outcome == snake.DIED))
        summaries.append(agent.log.summary())
    return np.array(results, dtype=np.float64), summaries


def run_ac(p, seed, modulator):
    exr.reseed(seed)
    feats = Features(p)
    agent = Agent(feats.size, 3, p["gamma"], p["lam"], p["critic_rate"], p["actor_rate"], clip=p["clip"],
                  modulator=modulator, post=p["post"], select=p["select"], tau=p["tau"], epsilon=p["epsilon"],
                  rng=np.random.default_rng(seed))
    train, logs = play(feats, agent, p, 2000 + seed, p["train"], True)
    test, _ = play(feats, agent, p, 1000 + seed, p["test"], False)
    blocks = [float(train[k:k + 20, 0].mean()) for k in range(0, len(train), 20)]
    return {
        "test_apples": float(test[:, 0].mean()), "test_steps": float(test[:, 1].mean()),
        "test_deaths": float(test[:, 2].mean()), "test_best": float(test[:, 0].max()),
        "train_apples": float(train[:, 0].mean()), "curve": blocks,
        "mean_abs_delta": [round(float(np.mean([l["mean_abs_delta"] for l in logs[k:k + 20]])), 4) for k in range(0, len(logs), 20)],
        "mean_delta": [round(float(np.mean([l["mean_delta"] for l in logs[k:k + 20]])), 4) for k in range(0, len(logs), 20)],
        "entropy": [round(float(np.mean([l["entropy"] for l in logs[k:k + 20]])), 4) for k in range(0, len(logs), 20)],
        "clamped": agent.actor.clamped(),
        "critic_w_max": float(np.abs(agent.critic.w).max()),
    }


def run_original(p, seed):
    """snake_py as it is (sign readouts, error-driven, epsilon 0.05, lr 0.03)."""
    q = {"width": p["width"], "height": p["height"], "mix": p["mix"], "train": p["train"], "test": p["test"],
         "lr": 0.03, "explore": 0.05, "eat": p["eat"], "die": p["die"], "approach": p["approach"], "reward": "error"}
    exr.reseed(seed)
    learner = original.Player(q)
    train = original.play(learner, q, 2000 + seed, q["train"], q["lr"], q["explore"])
    test = original.play(learner, q, 1000 + seed, q["test"], 0.0, 0.0)
    blocks = [float(train[k:k + 20, 0].mean()) for k in range(0, len(train), 20)]
    return {"test_apples": float(test[:, 0].mean()), "test_steps": float(test[:, 1].mean()),
            "test_deaths": float(test[:, 2].mean()), "test_best": float(test[:, 0].max()),
            "train_apples": float(train[:, 0].mean()), "curve": blocks}


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=("td", "reward", "original"))
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    exr.set_threads(1)
    p = dict(DEFAULTS)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in p:
            raise SystemExit(f"unknown setting {k}")
        p[k] = parse_value(v)
    lo, hi = (int(s) for s in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
    with open(args.out, "a") as f:
        for seed in range(lo, hi + 1):
            t0 = time.time()
            try:
                with np.errstate(over="raise", invalid="raise"):
                    res = run_original(p, seed) if args.arm == "original" else run_ac(p, seed, args.arm)
            except (FloatingPointError, ValueError) as e:  # the critic or actor diverged
                res = {"diverged": str(e), "test_apples": 0.0, "train_apples": 0.0, "curve": []}
            row = {"arm": args.arm, "seed": seed, "settings": p, "seconds": round(time.time() - t0, 1), **res}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(args.arm, seed, "test", round(res["test_apples"], 2), "train", round(res["train_apples"], 2),
                  "curve", [round(c, 1) for c in res["curve"]], flush=True)


if __name__ == "__main__":
    main()
