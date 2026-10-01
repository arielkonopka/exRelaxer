#!/usr/bin/env python3
"""Evolves the goe_rl network on Gardens of Eris: OpenAI-ES (antithetic
Gaussian perturbations, centred-rank fitness, Adam) on its weights, scored
by the game reward of experiment.py (score, minus w_death per avatar lost),
as doom_rl/es.py does for Doom. `evolve` "all" (the default here, the best
setting on Doom) evolves the hidden layers' weights and the readouts;
"readout" only the readouts. Steps are relative to each layer's weight RMS.

Every candidate of a generation plays the same worlds; the current weights
also play fixed validation worlds every generation. <out>/log.jsonl gets a
line per generation, <out>/state.npz holds the weights and optimizer state
(the run resumes from it), <out>/best.exr the network with the best
validation so far (and best_theta.npy its weights).

    python3 NNtesting/experiments/goe_rl/es.py --out results/goe-es/er_d1 --workers 4 --generations 300
    python3 NNtesting/experiments/goe_rl/es.py --out ... --config '{"recurrent": true, "habituation": false}'
    python3 NNtesting/experiments/goe_rl/es.py --out results/goe-es/er_reservoir_grow \
        --config NNtesting/experiments/goe_rl/models/er_reservoir_grow.json

--config takes JSON or a file of it (models/ holds the designed networks,
doc/goe.md). With grow_to (in the config, or --grow-to) the network starts
at `depth` hidden layers and gains one on top, up to grow_to, as soon as it
learns: when the mean validation reward of the last grow_window generations
beats the first grow_window at this depth by grow_margin. The new layer's
weights start random, and every old weight is carried over by the input it
reads, those from the new layer starting at zero (_shared/wiring.py), so the
grown network plays as before up to float rounding. A frozen reservoir
never evolves.
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
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))

VALIDATION_SEED = 777777
_worker = {}


GROW = {"grow_to": 0, "grow_window": 20, "grow_margin": 1.0}


def params(config):
    import experiment as e
    p = {k: v[0] for k, v in e.PARAMS.items()}
    p["evolve"] = "all"
    p.update(GROW)
    p.update(config)
    return p


def read_config(text):
    """--config: JSON, or a file holding it (e.g. models/er_reservoir.json)."""
    if os.path.isfile(text):
        with open(text) as f:
            config = json.load(f)
    else:
        config = json.loads(text)
    config.pop("description", None)  # the model files say what they are
    return config


def init_worker(p, net_seed):
    import exrelaxer as exr
    import experiment as e
    exr.set_threads(1)
    _worker.update(p=p, net_seed=net_seed, game=e.make_game(p))


def groups(player, evolve):
    """The weight vectors that evolve, as groups of (layer, neuron): the
    readouts, then (evolve == "all") each hidden layer's neurons."""
    out = [[(r, 0) for r in player.readouts]]
    if evolve == "all":
        hidden = player.evolving(player.p) if player.wiring is not None else player.layers
        out += [[(h, i) for i in range(player.net.layer_size(h))] for h in hidden]
    return out


def keyed(player, evolve):
    """(key, sources) per evolving neuron in flat order, for wiring.carry."""
    return [((player.wiring.names[layer], i), player.wiring.sources(layer, i))
            for group in groups(player, evolve) for layer, i in group]


def grow(p, net_seed, arrays):
    """One hidden layer more on top: the deeper network's parameters, and the
    flat arrays (weights, Adam's moments) carried over onto it."""
    import wiring
    q = dict(p, depth=p["depth"] + 1)
    old, new = build(p, net_seed, None), build(q, net_seed, None)
    fresh = initial_weights(q, net_seed)[0]
    return q, wiring.carry(keyed(old, p["evolve"]), keyed(new, q["evolve"]), arrays, fresh)


def build(p, net_seed, theta):
    import exrelaxer as exr
    import experiment as e
    exr.reseed(net_seed)  # the same random network every time; theta replaces what evolves
    player = e.Player(p)
    player.p = p
    if theta is not None:
        at = 0
        for group in groups(player, p["evolve"]):
            for layer, i in group:
                n = len(player.net.weights(layer, i))
                player.net.set_weights(layer, i, theta[at:at + n].astype(np.float32))
                at += n
    return player


def initial_weights(p, net_seed):
    """The network's own weights of everything that evolves, and for each weight its group's RMS."""
    player = build(p, net_seed, None)
    theta, rms = [], []
    for group in groups(player, p["evolve"]):
        w = np.concatenate([np.asarray(player.net.weights(layer, i), dtype=np.float64) for layer, i in group])
        theta.append(w)
        rms.append(np.full(len(w), np.sqrt(np.mean(w ** 2))))
    return np.concatenate(theta), np.concatenate(rms)


def play(task):
    """Plays one game per seed with weights theta; returns (reward, score, avatars lost, spikes per step)."""
    import experiment as e
    theta, seeds = task
    p = _worker["p"]
    player = build(p, _worker["net_seed"], theta)  # fresh E-R state for every candidate
    r = e.play(_worker["game"], player, p, seeds, meter=True)
    return r["reward"], r["score"], r["avatars_lost"] + r["dead"], player.spikes / max(player.ticks_seen, 1) * p["ticks"]


def centred_ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}", help="goe_rl parameters as JSON, and evolve (all or readout)")
    ap.add_argument("--net-seed", type=int, default=0, help="seed of the random starting network")
    ap.add_argument("--pairs", type=int, default=12, help="antithetic pairs per generation")
    ap.add_argument("--sigma", type=float, default=0.05, help="perturbation size, relative to the weights' RMS")
    ap.add_argument("--lr", type=float, default=0.02, help="Adam step, relative to the weights' RMS")
    ap.add_argument("--episodes", type=int, default=2, help="worlds per candidate")
    ap.add_argument("--validation", type=int, default=6, help="fixed validation worlds per generation")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grow-to", type=int, default=None, help="grow up to this many hidden layers (over the config)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = params(read_config(args.config))
    if args.grow_to is not None:
        p["grow_to"] = args.grow_to
    if p["grow_to"]:
        from experiment import designed
        if p["evolve"] != "all":
            raise ValueError("grow_to needs evolve all: a frozen random layer would change with the depth")
        p["readout_from"] = "all"  # the readouts read the new layer, from zero weights
        if not designed(p):
            raise ValueError("grow_to needs the designed network (readout_from all)")
        if p.get("feedback_first", 0) and p.get("feedback_first_from", "all") != "all":
            raise ValueError("grow_to needs feedback_first_from all: a top-only rung would move to every new layer")
    state_path = os.path.join(args.out, "state.npz")
    if os.path.exists(state_path):
        saved = np.load(state_path)
        if "depth" in saved.files:
            p["depth"] = int(saved["depth"])  # a grown run resumes at its depth
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "params": p}, f, indent=1)

    theta0, scale = initial_weights(p, args.net_seed)
    vals, grown_at = [], 0
    if os.path.exists(state_path):
        s = np.load(state_path)
        theta, m, v, gen, best = s["theta"], s["m"], s["v"], int(s["gen"]), float(s["best"])
        vals = list(s["vals"]) if "vals" in s.files else []
        grown_at = int(s["grown_at"]) if "grown_at" in s.files else 0
    else:
        theta, m, v, gen, best = theta0.copy(), np.zeros_like(theta0), np.zeros_like(theta0), 0, -np.inf
    rng = np.random.default_rng(args.seed + gen)
    sigma, lr = args.sigma * scale, args.lr * scale
    validation = [VALIDATION_SEED + i for i in range(args.validation)]

    while gen < args.generations:
        with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(p, args.net_seed)) as pool:
            while gen < args.generations:
                start = time.time()
                eps = rng.standard_normal((args.pairs, len(theta)))
                first = int(rng.integers(1, 1 << 30))
                seeds = [first + i for i in range(args.episodes)]  # every candidate plays the same worlds
                tasks = [(theta + sigma * e, seeds) for e in eps] + [(theta - sigma * e, seeds) for e in eps]
                tasks.append((theta, validation))
                results = pool.map(play, tasks)
                val, val_score, val_lost, val_spikes = results.pop()
                rewards = np.array([r[0] for r in results])
                ranks = centred_ranks(rewards)
                grad = (ranks[:args.pairs] - ranks[args.pairs:]) @ eps / (2 * args.pairs * sigma)
                gen += 1
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad ** 2
                step = lr * (m / (1 - 0.9 ** gen)) / (np.sqrt(v / (1 - 0.999 ** gen)) + 1e-8)
                if val > best:
                    best = val
                    build(p, args.net_seed, theta).net.save(os.path.join(args.out, "best.exr"))
                    np.save(os.path.join(args.out, "best_theta.npy"), theta)
                    with open(os.path.join(args.out, "best.json"), "w") as f:
                        json.dump({"generation": gen, "validation": val, "depth": p["depth"]}, f)
                theta = theta + step
                vals.append(val)
                w = p["grow_window"]
                since = vals[grown_at:]
                grew = (p["depth"] < p["grow_to"] and len(since) >= 2 * w
                        and np.mean(since[-w:]) - np.mean(since[:w]) >= p["grow_margin"])
                if grew:
                    p, (theta, m, v) = grow(p, args.net_seed, (theta, m, v))
                    scale = initial_weights(p, args.net_seed)[1]
                    sigma, lr = args.sigma * scale, args.lr * scale
                    grown_at = len(vals)
                    with open(os.path.join(args.out, "config.json"), "w") as f:
                        json.dump({"args": vars(args), "params": p}, f, indent=1)
                np.savez(state_path, theta=theta, m=m, v=v, gen=gen, best=best, vals=np.array(vals),
                         grown_at=grown_at, depth=p["depth"])
                line = {"generation": gen, "depth": p["depth"], "validation_reward": val, "validation_score": val_score,
                        "validation_avatars_lost": val_lost, "validation_spikes_per_step": val_spikes,
                        "population_mean": float(rewards.mean()), "population_best": float(rewards.max()),
                        "best_validation": best, "seconds": time.time() - start}
                with open(os.path.join(args.out, "log.jsonl"), "a") as f:
                    f.write(json.dumps(line) + "\n")
                print(f"gen {gen}: validation {val:.1f} (score {val_score:.1f}, lost {val_lost:.2f}, "
                      f"{val_spikes:.1f} spikes), population {rewards.mean():.1f} best {rewards.max():.1f}, "
                      f"{line['seconds']:.0f} s", flush=True)
                if grew:
                    print(f"gen {gen}: grown to {p['depth']} hidden layers", flush=True)
                    break  # the workers restart with the deeper network


if __name__ == "__main__":
    main()
