#!/usr/bin/env python3
"""How far back does each hidden E-R layer of an evolved agent remember?

Plays games with an agent saved by es.py and records, at every step, each
hidden layer's output (mean over the step's ticks), the screen the network
saw (pooled to 5 x 4), each layer's E-R thresholds after the step (the
state a neuron's history leaves, whether or not it fires), the action it took, and whether it was hit or made a
kill. Then, for each lag k, a ridge readout is fitted on some games and
tested on the others:

- `past screen`: the screen k steps ago from one layer's state now;
- `past action`: the action k steps ago (accuracy);
- `memory beyond the input`: the gain in R^2 for the screen k steps ago
  when the layer's state is added to the current input (the pooled
  screen the network sees, 15 x 20). The screen
  changes slowly, so the current screen alone already predicts much of a
  recent one; only this gain is information the layer carries about the
  past.

    python3 NNtesting/experiments/doom_rl/memprobe.py --agent DIR [--theta state.npz] --out probe.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(folder, theta):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    p = cfg["params"]
    path = os.path.join(folder, theta)
    if path.endswith(".npz"):  # es.py's state: the current weights, at the current depth
        s = np.load(path)
        p["depth"] = int(s["depth"]) if "depth" in s.files else p["depth"]
        return p, cfg.get("args", {}).get("net_seed", 0), s["theta"]
    return p, cfg.get("args", {}).get("net_seed", 0), np.load(path)


def record(p, net_seed, theta, games, seed, minutes):
    import vizdoom as vzd
    import exrelaxer as exr
    import es
    import experiment as e
    exr.set_threads(1)
    p = dict(p, episode_tics=min(p["episode_tics"], int(minutes * 60 * 35)))
    game = e.make_game(p, seed)
    buttons = np.eye(len(e.ACTIONS), dtype=int).tolist()
    rng = np.random.default_rng(0)
    out = []
    for g in range(games):
        player = es.build(p, net_seed, theta)
        net, layers = player.net, player.layers
        acc = [np.zeros(net.layer_size(h)) for h in layers]

        class Summing:  # the network, with each layer's output summed over the step's ticks
            def __getattr__(self, name):
                return getattr(net, name)

            def step(self):
                net.step()
                for i, h in enumerate(layers):
                    acc[i] += np.asarray(net.layer_output(h)).ravel()
        player.net = Summing()
        game.new_episode()
        rec = dict(states=[[] for _ in layers], thr=[[] for _ in layers], screen=[], eye=[], action=[], hurt=[], kill=[])
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        while not game.is_episode_finished():
            state = game.get_state()
            for a_ in acc:
                a_[:] = 0.0
            a = player.act(state, rng, 0.0)
            for i, h in enumerate(layers):
                rec["states"][i].append(acc[i] / player.ticks)
                rec["thr"][i].append(np.asarray(net.neuron_state(h)["threshold"]))
            s = state.screen_buffer.astype(np.float32) / 255.0
            rec["screen"].append(s.reshape(4, 30, 5, 32).mean(axis=(1, 3)).ravel())
            f = player.pool  # the network's own input
            rec["eye"].append(s.reshape(120 // f, f, 160 // f, f).mean(axis=(1, 3)).ravel())
            rec["action"].append(a)
            game.make_action(buttons[a], e.FRAME_SKIP)
            h = game.get_game_variable(vzd.GameVariable.HEALTH)
            k = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            rec["hurt"].append(float(h < health))
            rec["kill"].append(float(k > kills))
            health, kills = h, k
        out.append({k: (np.array(v) if k not in ("states", "thr") else [np.array(x) for x in v]) for k, v in rec.items()})
        print(f"game {g + 1}: {len(out[-1]['action'])} steps", flush=True)
    game.close()
    return out


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(Xs.shape[1]), Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


def lagged(games, feats, target, k):
    """Features at t and target at t - k, over all t >= k of every game."""
    X = np.concatenate([f(g)[k:] for g in games for f in [feats]])
    Y = np.concatenate([target(g)[:len(target(g)) - k] for g in games])
    return X, Y


def r2(Y, P):
    Y, P = Y.reshape(len(Y), -1), P.reshape(len(P), -1)
    return float(1.0 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent", required=True)
    ap.add_argument("--theta", default="best_theta.npy", help="weights file, or es.py's state.npz")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--lags", default="0,1,2,3,4,6,8,12,16")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p, net_seed, theta = load(args.agent, args.theta)
    games = record(p, net_seed, theta, args.games, args.seed, args.minutes)
    train, test = games[: 2 * len(games) // 3], games[2 * len(games) // 3:]
    n_actions = int(max(a for g in games for a in g["action"])) + 1
    onehot = lambda g: np.eye(max(n_actions, 8))[g["action"]]
    screen = lambda g: g["screen"]
    eye = lambda g: g["eye"]
    depth = len(games[0]["states"])
    sources = {f"layer {i + 1}": (lambda i: lambda g: g["states"][i])(i) for i in range(depth)}
    sources["all layers"] = lambda g: np.concatenate(g["states"], axis=1)
    for i in range(depth):
        sources[f"thresholds {i + 1}"] = (lambda i: lambda g: g["thr"][i])(i)
    sources["all thresholds"] = lambda g: np.concatenate(g["thr"], axis=1)
    sources["all outputs and thresholds"] = lambda g: np.concatenate(g["states"] + g["thr"], axis=1)
    lags = [int(x) for x in args.lags.split(",")]
    res = dict(agent=args.agent, theta=args.theta, depth=depth, games=args.games, seed=args.seed,
               steps=int(sum(len(g["action"]) for g in games)),
               spikes_per_layer=[float(np.mean([np.count_nonzero(np.abs(g["states"][i]) > 1e-6, axis=1).mean()
                                                for g in games])) for i in range(depth)],
               lags=lags, screen={}, action={}, beyond_input={}, input_only=[])
    for k in lags:
        Xi, Y = lagged(train, eye, screen, k)
        Ti, Yt = lagged(test, eye, screen, k)
        base = r2(Yt, ridge_fit(Xi, Y, args.ridge)(Ti))
        res["input_only"].append(round(base, 4))
        for name, f in sources.items():
            X, _ = lagged(train, f, screen, k)
            T, _ = lagged(test, f, screen, k)
            res["screen"].setdefault(name, []).append(round(r2(Yt, ridge_fit(X, Y, args.ridge)(T)), 4))
            both = ridge_fit(np.hstack([X, Xi]), Y, args.ridge)(np.hstack([T, Ti]))
            res["beyond_input"].setdefault(name, []).append(round(r2(Yt, both) - base, 4))
            Xa, Ya = lagged(train, f, onehot, k)
            Ta, Yta = lagged(test, f, onehot, k)
            pred = ridge_fit(Xa, Ya, args.ridge)(Ta).argmax(1)
            if name == "layer 1":
                res.setdefault("action_majority", []).append(round(float(Yta.mean(0).max()), 4))
            res["action"].setdefault(name, []).append(round(float((pred == Yta.argmax(1)).mean()), 4))
        print(f"lag {k}: input {base:.3f} | " + " | ".join(
            f"{n}: screen {res['screen'][n][-1]:.3f} +{res['beyond_input'][n][-1]:.3f} act {res['action'][n][-1]:.2f}"
            for n in sources), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
