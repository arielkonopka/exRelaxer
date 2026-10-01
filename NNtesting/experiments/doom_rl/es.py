#!/usr/bin/env python3
"""Evolves an E-R network with habituation to play Doom: an evolution
strategy (OpenAI-ES: antithetic Gaussian perturbations, centred-rank
fitness, Adam) on its weights, scored directly by the shaped game reward of
rewards.py. With `evolve` "readout" (the default) only the action readouts
evolve and the hidden E-R layers stay the random network doom_rl builds;
with "all" the hidden layers' weights evolve too (steps are relative to
each layer's weight RMS). No learning rule runs during play.

Why: in doom_rl the reward-driven readout rules did not beat the untrained
network (research log §20), so the credit-assignment problem is skipped
here: every candidate plays the same episodes, and the better half pulls
the weights its way.

Every generation appends a line to <out>/log.jsonl (mean and best
population reward, the current weights' reward on fixed validation
episodes, spikes); <out>/state.npz holds the weights and optimizer state
(the run resumes from it), and <out>/best.exr the network with the best
validation weights so far.

    python3 NNtesting/experiments/doom_rl/es.py --out results/doom-es/dtc --workers 4 --generations 300
    python3 NNtesting/experiments/doom_rl/es.py --out ... --config '{"depth": 2, "feedback": "recurrent"}'
    python3 NNtesting/experiments/doom_rl/es.py --scenario map01 --out ... --init <readout best_theta.npy> \
        --config '{"depth": 1, "feedback": "none", "evolve": "all"}'
    python3 NNtesting/experiments/doom_rl/es.py --out ... --grow-to 4 \
        --config '{"depth": 1, "feedback": "none", "evolve": "all", "feedback_first": 16, "feedback_first_model": "relu"}'

With --grow-to the network gains a hidden layer on top whenever it learns;
with the feedback ladder (feedback_first, see experiment.py) each new layer
also brings its rung back to h1.
"""
import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULTS = {"model": "er", "habituation": True, "habituation_tolerance": 0.05, "habituation_decay": 0.9,
            "habituation_fade_after": 2, "sound": True, "depth": 2, "width": 128, "feedback": "recurrent",
            "ticks": 6, "pool": 8, "reservoir": 0, "evolve": "readout"}

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
    # The pool ends its workers with SIGTERM; without closing its game a
    # worker leaves the Doom engine running, forever, and es.py hangs at exit
    # (the engine holds the pipe multiprocessing waits on).
    signal.signal(signal.SIGTERM, close_game)


def close_game(signum=None, frame=None):
    game = _worker.pop("game", None)
    if game is not None:
        game.close()
    os._exit(0)


def groups(player, evolve):
    """The weight vectors that evolve, as groups of (layer, neuron): the
    readouts, then (evolve == "all") each hidden layer's neurons."""
    out = [[(r, 0) for r in player.readouts]]
    if evolve == "all":
        out += [[(h, i) for i in range(player.net.layer_size(h))] for h in player.layers + getattr(player, "rungs", [])]
    return out


def build(p, net_seed, theta):
    import exrelaxer as exr
    import experiment as e
    exr.reseed(net_seed)  # the same random network every time; theta replaces what evolves
    player = e.Player(p)
    if theta is not None:
        at = 0
        for group in groups(player, p.get("evolve", "readout")):
            for layer, i in group:
                n = len(player.net.weights(layer, i))
                player.net.set_weights(layer, i, theta[at:at + n].astype(np.float32))
                at += n
    return player


def initial_weights(p, net_seed):
    """The network's own weights of everything that evolves, and for each
    weight the RMS of its group, so that steps are relative to each group's scale."""
    player = build(p, net_seed, None)
    theta, rms = [], []
    for group in groups(player, p.get("evolve", "readout")):
        w = np.concatenate([np.asarray(player.net.weights(layer, i), dtype=np.float64) for layer, i in group])
        theta.append(w)
        rms.append(np.full(len(w), np.sqrt(np.mean(w ** 2))))
    return np.concatenate(theta), np.concatenate(rms), len(theta[0])


def split(player, evolve, theta):
    """theta cut into one array per evolving (layer, neuron), grouped as groups()."""
    out, at = [], 0
    for group in groups(player, evolve):
        out.append([])
        for layer, i in group:
            n = len(player.net.weights(layer, i))
            out[-1].append(theta[at:at + n])
            at += n
    return out


def grow_ladder(p, q, net_seed, arrays):
    """grow() for the feedback ladder, where the new layer also adds a rung
    (neurons in h1, or a rung layer that h1 reads) and so new inputs to old
    neurons: every weight is matched by the input it reads (Player.sources).
    Old weights keep their values, weights from new inputs start at zero, and
    new neurons start as the network's random ones, so the grown network
    plays as before up to float rounding (the extra zero-weight inputs change
    the order of summation by about 1e-7; an E-R neuron at its threshold can
    tip the other way, so long games may part after a while)."""
    old, new = build(p, net_seed, None), build(q, net_seed, None)
    fresh = initial_weights(q, net_seed)[0]
    def keyed(player):
        return [(player.names[layer], i) for group in groups(player, p["evolve"]) for layer, i in group]
    old_keys, new_keys = keyed(old), keyed(new)
    lengths = [len(new.sources[k]) for k in new_keys]
    out = []
    for k, a in enumerate(arrays):
        at, before = 0, {}
        for key in old_keys:
            n = len(old.sources[key])
            before[key] = dict(zip(old.sources[key], a[at:at + n]))
            at += n
        parts, at = [], 0
        for key, n in zip(new_keys, lengths):
            if key in before:
                parts.append(np.array([before[key].get(s, 0.0) for s in new.sources[key]]))
            else:
                parts.append(fresh[at:at + n] if k == 0 else np.zeros(n))
            at += n
        out.append(np.concatenate(parts))
    return out


def grow(p, net_seed, arrays):
    """Adds a hidden layer on top: returns the deeper network's parameters and
    `arrays` (theta, Adam's m and v) mapped onto it. The readouts read every
    layer (readout_from all), and their weights from the new layer start at
    zero, so the grown network plays exactly as before; the new layer's own
    weights start as the network's random ones."""
    q = dict(p, depth=p["depth"] + 1)
    if p.get("feedback_first", 0):
        return q, grow_ladder(p, q, net_seed, arrays)
    old, new = build(p, net_seed, None), build(q, net_seed, None)
    fresh = split(new, q["evolve"], initial_weights(q, net_seed)[0])
    out = []
    for k, a in enumerate(arrays):
        parts = split(old, p["evolve"], a)
        grown = [np.concatenate([r, np.zeros(len(f) - len(r))]) for r, f in zip(parts[0], fresh[0])]
        grown += [w for layer in parts[1:] for w in layer]
        grown += [f if k == 0 else np.zeros_like(f) for f in fresh[-1]]
        out.append(np.concatenate(grown))
    return q, out


def play(task):
    """Plays `episodes` episodes from `seed` with weights theta; returns (reward, kills, spikes per step)."""
    import experiment as e
    theta, seed, episodes = task
    p, game = _worker["p"], _worker["game"]
    player = build(p, _worker["net_seed"], theta)  # fresh E-R state for every candidate
    game.set_seed(int(seed))
    stats, _ = e.play(game, player, p, episodes, 0.0, 0.0, np.random.default_rng(0), meter=True)
    return stats["reward"], stats["kills"], player.spikes / max(player.ticks_seen, 1) * p["ticks"]


def retry(what, write, tries=5):
    """Shared folders can fail a write now and then (EIO): retry, then carry on."""
    for i in range(tries):
        try:
            return write()
        except OSError as e:
            if i == tries - 1:
                print(f"warning: {what} not written: {e}", flush=True)
            time.sleep(2 * (i + 1))


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
    ap.add_argument("--grow-to", type=int, default=0,
                    help="add hidden layers, one at a time, up to this depth, each once the current network learns")
    ap.add_argument("--grow-window", type=int, default=20,
                    help="generations averaged to decide that the network learns")
    ap.add_argument("--grow-margin", type=float, default=1.0,
                    help="validation gain over the first window at this depth that counts as learning")
    ap.add_argument("--init", default="", help="start from these weights (a best_theta.npy of the same network)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = params(json.loads(args.config), args.scenario)
    if args.grow_to:
        if p["evolve"] != "all":
            raise ValueError("--grow-to needs evolve all: a frozen random layer would change with the depth")
        if p.get("feedback_first", 0) and p.get("feedback_first_from", "all") != "all":
            raise ValueError("--grow-to needs feedback_first_from all: a top-only rung would move to every new layer")
        p["readout_from"] = "all"  # the new layer joins the readouts with zero weights
    state_path = os.path.join(args.out, "state.npz")
    if os.path.exists(state_path):
        saved = np.load(state_path)
        if "depth" in saved.files:
            p["depth"] = int(saved["depth"])  # a grown run resumes at its depth
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "params": p}, f, indent=1)

    theta0, scale, readouts = initial_weights(p, args.net_seed)
    if os.path.exists(state_path):
        s = np.load(state_path)
        theta, m, v, gen, best = s["theta"], s["m"], s["v"], int(s["gen"]), float(s["best"])
        vals = list(s["vals"]) if "vals" in s.files else []
        grown_at = int(s["grown_at"]) if "grown_at" in s.files else 0
    else:
        start = theta0.copy()
        if args.init:
            init = np.load(args.init)
            if init.size == theta0.size:
                start = init.astype(np.float64)
            elif init.size == readouts:  # evolved readouts only: the rest starts as the network's own
                start[:readouts] = init
            else:
                raise ValueError(f"--init has {init.size} weights, this network {theta0.size} ({readouts} readout)")
        theta, m, v, gen, best = start, np.zeros_like(theta0), np.zeros_like(theta0), 0, -np.inf
        vals, grown_at = [], 0
    rng = np.random.default_rng(args.seed + gen)
    sigma, lr = args.sigma * scale, args.lr * scale

    ctx = mp.get_context("spawn")
    while gen < args.generations:
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
                    net = build(p, args.net_seed, theta).net
                    retry("best.exr", lambda: net.save(os.path.join(args.out, "best.exr")))
                    retry("best_theta.npy", lambda: np.save(os.path.join(args.out, "best_theta.npy"), theta))
                    retry("best.json", lambda: json.dump({"generation": gen, "validation": val, "depth": p["depth"]},
                                                         open(os.path.join(args.out, "best.json"), "w")))
                theta = theta + step
                vals.append(val)
                w = args.grow_window
                since = vals[grown_at:]
                grew = (p["depth"] < args.grow_to and len(since) >= 2 * w
                        and np.mean(since[-w:]) - np.mean(since[:w]) >= args.grow_margin)
                if grew:
                    p, (theta, m, v) = grow(p, args.net_seed, (theta, m, v))
                    _, scale, readouts = initial_weights(p, args.net_seed)
                    sigma, lr = args.sigma * scale, args.lr * scale
                    grown_at = len(vals)
                retry("state.npz", lambda: np.savez(state_path, theta=theta, m=m, v=v, gen=gen, best=best,
                                                    vals=np.array(vals), grown_at=grown_at, depth=p["depth"]))
                line = {"generation": gen, "depth": p["depth"], "validation_reward": val, "validation_kills": val_kills,
                        "validation_spikes_per_step": val_spikes, "population_mean": float(rewards.mean()),
                        "population_best": float(rewards.max()),
                        "kills_mean": float(np.mean([r[1] for r in results])), "best_validation": best,
                        "seconds": time.time() - start}
                def append():
                    with open(os.path.join(args.out, "log.jsonl"), "a") as f:
                        f.write(json.dumps(line) + "\n")
                retry("log.jsonl", append)
                print(f"gen {gen}: validation {val:.2f} (kills {val_kills:.1f}), population {rewards.mean():.2f} "
                      f"best {rewards.max():.2f}, {line['seconds']:.0f} s", flush=True)
                if grew:
                    print(f"gen {gen}: grown to {p['depth']} hidden layers", flush=True)
                    break  # the workers restart with the deeper network


if __name__ == "__main__":
    main()
