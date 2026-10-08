"""How big does a growing Snake network get? (results/snake-growth-long/, research log §31)

Starts the snake_growth agent from `start` E-R neurons and lets it play game after game (each game
ends at the snake's death), learning and developing all the time, until its size has not changed
for `stable` evaluation windows or `max_games` games have been played. Each run writes
OUT_DIR/<setup>_s<seed>.json with the score and the size after every 50-game window.

    python NNtesting/experiments/snake_growth/long_run.py OUT_DIR [--setups a,b] [--seeds 1,2,3]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment as sg  # this folder's experiment.py
import exrelaxer as exr
from exrelaxer.development import Plateau

OPEN = {"max_width": 4096, "max_depth": 64}   # caps far beyond anything reached
SETUPS = {
    "s16_fixed": {"start": 16, "grow": False},                     # control: no growth or pruning
    "s16_fixed_rec099": {"start": 16, "grow": False, "recovery": 0.99},
    "s16_capped": {"start": 16},                                   # experiment.py's caps (16 wide, 2 deep)
    "s16_open": dict(OPEN, start=16),
    "s16_open_noundo": dict(OPEN, start=16, undo=0),
    "s16_open_rec099": dict(OPEN, start=16, recovery=0.99),
    "s16_fixed_rec05": {"start": 16, "grow": False, "recovery": 0.5},
    "s16_open_rec05": dict(OPEN, start=16, recovery=0.5),
    "s16_fixed_rec075": {"start": 16, "grow": False, "recovery": 0.75},
    "s16_open_rec075": dict(OPEN, start=16, recovery=0.75),
    "s16_open_rec05_freeze4": dict(OPEN, start=16, recovery=0.5, freeze_from=4),
    "s16_open_rec075_freeze4": dict(OPEN, start=16, recovery=0.75, freeze_from=4),
    "s16_open_freeze4": dict(OPEN, start=16, freeze_from=4),
    "s16_open_rec099_noundo": dict(OPEN, start=16, recovery=0.99, undo=0),
}


def long_run(p, seed, stable, max_games):
    exr.reseed(seed)
    agent = sg.Agent(p)
    base = sg.play(agent, p, 50_000 + seed, p["eval_games"], learning=False)[:, 0].mean()
    plateau = Plateau(baseline=base, margin=p["margin"], patience=p["patience"])
    agent.last_score = base
    rows, unchanged, k, games = [], 0, 0, 0
    while games < max_games and unchanged < stable:
        before = agent.sizes()
        g = sg.play(agent, p, seed * 7919 + k, p["eval_games"], learning=True)
        games += len(g)
        score = float(g[:, 0].mean())
        agent.evaluated(score)
        if p["grow"] and plateau.update(score) and len(agent.layers) < p["max_depth"]:
            agent.deepen()
            plateau.restart()
        sizes = agent.sizes()
        unchanged = unchanged + 1 if sizes == before else 0
        rows.append({
            "games": games, "ticks": agent.ticks, "score": score, "steps": float(g[:, 1].mean()),
            "best_game": int(g[:, 0].max()), "sizes": sizes,
            "frozen": sum(len(agent.net.frozen_neurons(l)) for l in agent.layers),
            "width": len(agent.width.events), "pruned": len(agent.pruning.events),
            "undone": len(agent.undo_events), "depth": len(agent.depth_events),
        })
        k += 1
    return agent, {
        "baseline": float(base), "rows": rows, "stopped": "stable" if unchanged >= stable else "cap",
        "width_events": agent.width.events, "prune_events": agent.pruning.events,
        "undo_events": agent.undo_events, "depth_events": agent.depth_events,
    }


def job(args):
    out, name, seed, stable, max_games = args
    p = dict(sg.DEFAULTS, **SETUPS[name])
    agent, m = long_run(p, seed, stable, max_games)
    m.update(setup=name, seed=seed, params=p)
    (Path(out) / f"{name}_s{seed}.json").write_text(json.dumps(m, default=lambda o: o.item() if hasattr(o, "item") else list(o)))
    agent.net.save(str(Path(out) / f"{name}_s{seed}.exr"))
    last = m["rows"][-1]
    return name, seed, last["games"], last["sizes"], np.mean([r["score"] for r in m["rows"][-10:]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--setups", default=",".join(SETUPS))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--stable", type=int, default=200, help="windows without a size change that end a run")
    parser.add_argument("--max-games", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(args.out, n, int(s), args.stable, args.max_games)
            for n in args.setups.split(",") for s in args.seeds.split(",")]
    with ProcessPoolExecutor(args.workers) as pool:
        for name, seed, games, sizes, score in pool.map(job, jobs):
            print(f"{name} s{seed}: {games} games, sizes {sizes}, last-500 score {score:.2f}", flush=True)
