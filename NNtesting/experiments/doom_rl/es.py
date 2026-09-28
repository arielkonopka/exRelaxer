#!/usr/bin/env python3
"""Evolves the action readouts of an E-R network with habituation to play
Doom: an evolution strategy (OpenAI-ES: antithetic Gaussian perturbations,
centred-rank fitness, Adam) on the readout weights, scored directly by the
shaped game reward of rewards.py. The hidden E-R layers stay the frozen
random network doom_rl builds; no learning rule runs during play.

Why: in doom_rl the reward-driven readout rules did not beat the untrained
network (research log §19), so the credit-assignment problem is skipped
here: every candidate plays the same episodes, and the better half pulls
the weights its way.

Every generation appends a line to <out>/log.jsonl (mean and best
population reward, the current weights' reward on fixed validation
episodes, spikes); <out>/state.npz holds the weights and optimizer state
(the run resumes from it), and <out>/best.exr the network with the best
validation weights so far.

    python3 NNtesting/experiments/doom_rl/es.py --out results/doom-es/dtc --workers 4 --generations 300
    python3 NNtesting/experiments/doom_rl/es.py --out ... --config '{"depth": 2, "feedback": "recurrent"}'
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULTS = {"model": "er", "habituation": True, "habituation_tolerance": 0.05, "habituation_decay": 0.9,
            "habituation_fade_after": 2, "sound": True, "depth": 2, "width": 128, "feedback": "recurrent",
            "ticks": 6, "pool": 8, "reservoir": 0}

_worker = {}


def params(config, scenario):
    import experiment as e
    p = {k: v[0] for k, v in e.PARAMS.items()}
    p.update(DEFAULTS)
    p.update(config)
    p["scenario"] = scenario
    return p


def init_worker(p, net_seed):
    import exrelaxer as exr
    import experiment as e
    exr.set_threads(1)
    _worker.update(p=p, net_seed=net_seed, game=e.make_game(p, 1))


def build(p, net_seed, theta):
    import exrelaxer as exr
    import experiment as e
    exr.reseed(net_seed)  # the same frozen hidden network every time
    player = e.Player(p)
    if theta is not None:
        fan = len(theta) // len(player.readouts)
        for i, r in enumerate(player.readouts):
            player.net.set_weights(r, 0, theta[i * fan:(i + 1) * fan].astype(np.float32))
    return player


def readout_weights(p, net_seed):
    player = build(p, net_seed, None)
    return np.concatenate([np.asarray(player.net.weights(r, 0), dtype=np.float64) for r in player.readouts])


def play(task):
    """Plays `episodes` episodes from `seed` with weights theta; returns (reward, kills, spikes per step)."""
    import experiment as e
    theta, seed, episodes = task
    p, game = _worker["p"], _worker["game"]
    player = build(p, _worker["net_seed"], theta)  # fresh E-R state for every candidate
    game.set_seed(int(seed))
    stats, _ = e.play(game, player, p, episodes, 0.0, 0.0, np.random.default_rng(0), meter=True)
    return stats["reward"], stats["kills"], player.spikes / max(player.ticks_seen, 1) * p["ticks"]


def centred_ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--config", default="{}", help="doom_rl parameters as JSON (over the E-R defaults here)")
    ap.add_argument("--net-seed", type=int, default=0, help="seed of the frozen random network")
    ap.add_argument("--pairs", type=int, default=12, help="antithetic pairs per generation")
    ap.add_argument("--sigma", type=float, default=0.1, help="perturbation size, relative to the weights' RMS")
    ap.add_argument("--lr", type=float, default=0.03, help="Adam step, relative to the weights' RMS")
    ap.add_argument("--episodes", type=int, default=2, help="episodes per candidate")
    ap.add_argument("--validation", type=int, default=10, help="fixed validation episodes per generation")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = params(json.loads(args.config), args.scenario)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "params": p}, f, indent=1)

    state_path = os.path.join(args.out, "state.npz")
    theta0 = readout_weights(p, args.net_seed)
    scale = float(np.sqrt(np.mean(theta0 ** 2)))
    if os.path.exists(state_path):
        s = np.load(state_path)
        theta, m, v, gen, best = s["theta"], s["m"], s["v"], int(s["gen"]), float(s["best"])
    else:
        theta, m, v, gen, best = theta0.copy(), np.zeros_like(theta0), np.zeros_like(theta0), 0, -np.inf
    rng = np.random.default_rng(args.seed + gen)
    sigma, lr = args.sigma * scale, args.lr * scale

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_worker, initargs=(p, args.net_seed)) as pool:
        while gen < args.generations:
            start = time.time()
            eps = rng.standard_normal((args.pairs, len(theta)))
            seed = int(rng.integers(1, 1 << 30))  # every candidate plays the same episodes
            tasks = [(theta + sigma * e, seed, args.episodes) for e in eps] + \
                    [(theta - sigma * e, seed, args.episodes) for e in eps]
            tasks.append((theta, 777777, args.validation))  # validation: fixed episodes, current weights
            results = pool.map(play, tasks)
            val, val_kills, val_spikes = results.pop()
            rewards = np.array([r[0] for r in results])
            ranks = centred_ranks(rewards)
            grad = (ranks[:args.pairs] - ranks[args.pairs:]) @ eps / (2 * args.pairs * sigma)
            # Adam, ascending.
            gen += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            step = lr * (m / (1 - 0.9 ** gen)) / (np.sqrt(v / (1 - 0.999 ** gen)) + 1e-8)
            if val > best:
                best = val
                build(p, args.net_seed, theta).net.save(os.path.join(args.out, "best.exr"))
                np.save(os.path.join(args.out, "best_theta.npy"), theta)
            theta = theta + step
            np.savez(state_path, theta=theta, m=m, v=v, gen=gen, best=best)
            line = {"generation": gen, "validation_reward": val, "validation_kills": val_kills,
                    "validation_spikes_per_step": val_spikes, "population_mean": float(rewards.mean()),
                    "population_best": float(rewards.max()),
                    "kills_mean": float(np.mean([r[1] for r in results])), "best_validation": best,
                    "seconds": time.time() - start}
            with open(os.path.join(args.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"gen {gen}: validation {val:.2f} (kills {val_kills:.1f}), population {rewards.mean():.2f} "
                  f"best {rewards.max():.2f}, {line['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
