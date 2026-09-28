#!/usr/bin/env python3
"""Long-running search for an E-R network with habituation that plays Doom.

The network is always E-R with fading habituation; everything else is
searched: topology (depth, width, feedback lines, reservoir), E-R and
habituation settings, ticks per game step, screen pooling and the learning
settings. Each candidate is built and trained by doom_rl (experiment.py)
exactly as `nntest.py run doom_rl` would, and scored by its mean test
reward (the shaped reward of rewards.py) after training.

Method: asynchronous successive halving with evolution.
  rung 0   train `budgets[0]` episodes, 1 seed
  rung 1   train `budgets[1]` episodes, 2 seeds   (the top third of rung 0)
  rung 2   train `budgets[2]` episodes, 3 seeds   (the top third of rung 1)
New candidates are half random and half mutations of the best configs at
the highest rung reached so far. Workers run in parallel.

Everything is appended to <out>/evals.jsonl as it finishes, so the search
can be stopped and resumed at any time (same --out); <out>/best.md is
rewritten after every evaluation, and the trained networks of the top rung
are saved in <out>/nets/.

    python3 NNtesting/experiments/doom_rl/search.py --out results/doom-search --workers 4
    python3 NNtesting/experiments/doom_rl/search.py --out ... --scenario map01 --hours 12
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (kind, choices or (low, high))
SPACE = {
    "depth": ("choice", [1, 2, 3, 4]),
    "width": ("choice", [64, 128, 256, 512]),
    "feedback": ("choice", ["none", "recurrent", "topdown", "both"]),
    "feedback_width": ("choice", [16, 32, 64]),
    "reservoir": ("choice", [0, 128, 256, 512]),
    "reservoir_recurrent": ("choice", [64, 128, 256]),
    "recurrent_scale": ("uniform", (0.2, 1.0)),
    "ticks": ("choice", [2, 3, 4, 6, 9, 12]),
    "pool": ("choice", [4, 8]),
    "habituation_decay": ("uniform", (0.5, 0.99)),
    "habituation_fade_after": ("choice", [1, 2, 3, 4]),
    "habituation_tolerance": ("loguniform", (0.01, 0.3)),
    "rule": ("choice", ["sign", "trace"]),
    "trace": ("uniform", (0.5, 0.97)),
    "reward_mode": ("choice", ["error", "always"]),
    "lr": ("loguniform", (0.001, 0.1)),
    "explore": ("uniform", (0.05, 0.4)),
    "baseline": ("loguniform", (0.001, 0.1)),
}
FIXED = {"model": "er", "habituation": True, "sound": True, "learn_hidden": False}


def sample(rng):
    c = {}
    for k, (kind, v) in SPACE.items():
        if kind == "choice":
            c[k] = rng.choice(v)
        elif kind == "uniform":
            c[k] = round(rng.uniform(*v), 4)
        else:
            c[k] = round(math.exp(rng.uniform(math.log(v[0]), math.log(v[1]))), 5)
    return c


def mutate(parent, rng, rate=0.3):
    c = dict(parent)
    keys = [k for k in SPACE if rng.random() < rate] or [rng.choice(list(SPACE))]
    for k in keys:
        kind, v = SPACE[k]
        if kind == "choice":
            i = v.index(c[k]) if c[k] in v else 0
            c[k] = v[max(0, min(len(v) - 1, i + rng.choice([-1, 1])))] if rng.random() < 0.7 else rng.choice(v)
        elif kind == "uniform":
            c[k] = round(min(v[1], max(v[0], c[k] + rng.gauss(0, 0.15 * (v[1] - v[0])))), 4)
        else:
            c[k] = round(min(v[1], max(v[0], c[k] * math.exp(rng.gauss(0, 0.5)))), 5)
    return c


def key(config):
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def evaluate(job):
    """Runs in a worker: trains and tests one config on one seed."""
    sys.path.insert(0, HERE)
    import numpy as np
    import exrelaxer as exr
    import experiment as e
    exr.set_threads(1)
    p = {k: v[0] for k, v in e.PARAMS.items()}
    p.update(FIXED)
    p.update(job["config"])
    p.update(scenario=job["scenario"], train=job["train"], test=job["test"])
    seed = job["seed"]
    start = time.time()
    try:
        exr.reseed(seed)
        rng = np.random.default_rng(seed)
        player = e.Player(p)
        game = e.make_game(p, 1000 + seed)
        train, _ = e.play(game, player, p, p["train"], p["lr"], p["explore"], rng)
        game.close()
        game = e.make_game(p, 5000 + seed)
        test, parts = e.play(game, player, p, p["test"], 0.0, 0.0, rng, meter=True)
        game.close()
        result = {"train_reward": train["reward"], **test,
                  "spikes_per_step": player.spikes / max(player.ticks_seen, 1) * p["ticks"],
                  "hidden_neurons": player.neurons, **{"reward_" + k: v for k, v in parts.items()}}
        if job.get("save"):
            os.makedirs(os.path.dirname(job["save"]), exist_ok=True)
            player.net.save(job["save"])
    except Exception:
        result = {"error": traceback.format_exc(limit=3)}
    return {**job, "result": result, "seconds": time.time() - start, "finished": time.time()}


class Search:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.path = os.path.join(args.out, "evals.jsonl")
        os.makedirs(args.out, exist_ok=True)
        self.configs, self.scores, self.running = {}, {}, set()  # scores[(key, rung)] = {seed: reward}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("scenario") != args.scenario:
                        continue
                    self.record(r)

    def record(self, r):
        k = key(r["config"])
        self.configs[k] = r["config"]
        reward = r["result"].get("reward", -1e9) if "error" not in r["result"] else -1e9
        self.scores.setdefault((k, r["rung"]), {})[r["seed"]] = reward

    def seeds(self, rung):
        return list(range(rung + 1))

    def complete(self, rung):
        """Configs with every seed of `rung` done: {key: mean reward}."""
        need = set(self.seeds(rung))
        return {k: sum(s.values()) / len(s) for (k, r), s in self.scores.items() if r == rung and set(s) >= need}

    def next_job(self):
        budgets = self.args.budgets
        # Promote: the top third of a rung that is not yet in the next one.
        for rung in reversed(range(len(budgets) - 1)):
            done = self.complete(rung)
            if len(done) < 3:
                continue
            ranked = sorted(done, key=done.get, reverse=True)[:max(1, len(done) // 3)]
            for k in ranked:
                for seed in self.seeds(rung + 1):
                    if seed not in self.scores.get((k, rung + 1), {}) and (k, rung + 1, seed) not in self.running:
                        return self.job(k, rung + 1, seed)
        # Unfinished seeds of a rung in progress.
        for (k, rung), s in list(self.scores.items()):
            for seed in self.seeds(rung):
                if seed not in s and (k, rung, seed) not in self.running:
                    return self.job(k, rung, seed)
        # A new candidate: mutate a good one or sample at random.
        parents = []
        for rung in reversed(range(len(budgets))):
            done = self.complete(rung)
            if len(done) >= 2:
                parents = sorted(done, key=done.get, reverse=True)[:5]
                break
        if parents and self.rng.random() < 0.5:
            config = mutate(self.configs[self.rng.choice(parents)], self.rng)
        else:
            config = sample(self.rng)
        k = key(config)
        self.configs[k] = config
        return self.job(k, 0, 0)

    def job(self, k, rung, seed):
        self.running.add((k, rung, seed))
        top = rung == len(self.args.budgets) - 1
        return {"config": self.configs[k], "key": k, "rung": rung, "seed": seed, "scenario": self.args.scenario,
                "train": self.args.budgets[rung], "test": self.args.test,
                "save": os.path.join(self.args.out, "nets", f"{k}_s{seed}.exr") if top else None}

    def finish(self, r):
        self.running.discard((r["key"], r["rung"], r["seed"]))
        self.record(r)
        with open(self.path, "a") as f:
            f.write(json.dumps(r) + "\n")
        self.report()

    def report(self):
        lines = [f"# Doom search: {self.args.scenario}", "",
                 f"{sum(len(s) for s in self.scores.values())} evaluations, {len(self.configs)} configs. "
                 "Score: mean test reward (higher is better).", ""]
        for rung in reversed(range(len(self.args.budgets))):
            done = self.complete(rung)
            if not done:
                continue
            lines += [f"## Rung {rung}: {self.args.budgets[rung]} training episodes, {rung + 1} seed(s), "
                      f"{len(done)} configs", "", "| reward | config |", "|---|---|"]
            for k in sorted(done, key=done.get, reverse=True)[:10]:
                lines.append(f"| {done[k]:.2f} | `{k}` {json.dumps(self.configs[k], sort_keys=True)} |")
            lines.append("")
        with open(os.path.join(self.args.out, "best.md"), "w") as f:
            f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder (resumes if it exists)")
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], default=[100, 300, 900],
                    help="training episodes per rung")
    ap.add_argument("--test", type=int, default=10, help="test episodes per evaluation")
    ap.add_argument("--hours", type=float, default=0, help="stop starting new jobs after this long (0: never)")
    ap.add_argument("--evals", type=int, default=0, help="stop after this many new evaluations (0: never)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    search = Search(args)
    deadline = time.time() + args.hours * 3600 if args.hours else float("inf")
    started = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        pending = []

        def submit():
            nonlocal started
            pending.append(pool.apply_async(evaluate, (search.next_job(),)))
            started += 1

        for _ in range(args.workers):
            submit()
        while pending:
            time.sleep(1)
            for a in [a for a in pending if a.ready()]:
                pending.remove(a)
                r = a.get()
                search.finish(r)
                res = r["result"]
                print(f"{time.strftime('%H:%M:%S')} rung {r['rung']} seed {r['seed']} {r['key']} "
                      f"reward {res.get('reward', float('nan')):.2f} kills {res.get('kills', 0):.1f} "
                      f"({r['seconds']:.0f} s){' ERROR' if 'error' in res else ''}", flush=True)
                if time.time() < deadline and (not args.evals or started < args.evals):
                    submit()


if __name__ == "__main__":
    main()
