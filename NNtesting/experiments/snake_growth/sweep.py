"""The runs behind results/snake-growth/ (doc/snake_growth_guide.md, research log §30).

    python NNtesting/experiments/snake_growth/sweep.py OUT_DIR            # every setup, 3 seeds
    python NNtesting/experiments/snake_growth/sweep.py OUT_DIR --saturation  # the saturation table

Each run writes OUT_DIR/<setup>_s<seed>.json (metrics, score curve, final sizes, event log).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment as sg  # this folder's experiment.py
import exrelaxer as exr

SETUPS = {
    "one_fixed": {"start": 1, "grow": False},
    "one_grow": {"start": 1},
    "one_grow_plastic": {"start": 1, "freeze_old": False},
    "one_grow_rec099": {"start": 1, "recovery": 0.99},
    "four_fixed": {"start": 4, "grow": False},
    "four_grow": {"start": 4},
    "four_grow_plastic": {"start": 4, "freeze_old": False},
    "eight_fixed": {"start": 8, "grow": False},
}


def job(args):
    out, name, seed = args
    p = dict(sg.DEFAULTS, **SETUPS[name])
    log = []
    agent, m = sg.develop(p, seed, report=log.append)
    m = {k: ([float(x) for x in v] if isinstance(v, list) else float(v)) for k, v in m.items()}
    m.update(setup=name, seed=seed, params=p, sizes=agent.sizes(), log=log)
    (Path(out) / f"{name}_s{seed}.json").write_text(json.dumps(m))
    return name, seed, m["final"]


def saturation(sizes=(1, 2, 4, 8, 16), recoveries=(0.9, 0.99), games=30):
    """Share of ticks on which an untrained ER(n) population was saturated, playing with 30%
    random moves."""
    for recovery in recoveries:
        for n in sizes:
            agent = sg.Agent(dict(sg.DEFAULTS, start=n, recovery=recovery))
            monitor = exr.ActivityMonitor(exr.ActivitySpec(window=200))
            rng, saturated = sg.snake.Rng(5), []
            for i in range(games):
                g = sg.snake.Game(10, 10, i)
                while not g.over:
                    action = agent.choose(g.state())
                    monitor.observe(agent.net, agent.er)
                    saturated.append(monitor.saturated_tick)
                    g.step(action if rng.uniform() > 0.3 else rng.below(3))
            print(f"recovery {recovery} ER({n}): saturated ticks {np.mean(saturated):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--setups", default=",".join(SETUPS))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--saturation", action="store_true")
    args = parser.parse_args()
    if args.saturation:
        exr.reseed(1)
        saturation()
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        jobs = [(args.out, n, int(s)) for s in args.seeds.split(",") for n in args.setups.split(",")]
        with ProcessPoolExecutor(args.workers) as pool:
            for name, seed, final in pool.map(job, jobs):
                print(name, seed, round(final, 2), flush=True)
