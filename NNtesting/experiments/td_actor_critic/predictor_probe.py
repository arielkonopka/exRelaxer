"""Can the plastic layer learn to predict the next frame? (Phase 36 diagnostic.)

Phase 35's `plastic_teacher=view` layer never learned its prediction in play (error 0.95-0.99 of
the mean picture's at the end). This probe records random games (actions held for ~10 steps,
strike and orient reflexes from labels, as in detector_probe.py) on the training levels and keeps
for every step the frozen vision pool the plastic layer reads (500 values) and the coarse target
picture (6 x 8 blocks of the pooled screen, centred, unit variance, clipped to +-3). Then it
compares, on held-out games (normalised MSE: 1.0 = predicting the mean picture):

  copy         the current picture as the prediction of the next
  ridge        a linear map from the pool to the next picture (fitted on the training games)
  ridge_now    the same map to the current picture (how much of the picture the pool holds)
  online ...   doom_td's plastic layer (e-prop E-R, feedback-alignment linear readout) trained
               online, one pass in game order, at several rates; timing "lag" applies the error
               one step late as in play, "aligned" right after the prediction (target known)

    python3 predictor_probe.py --games 30 --out frames.npz
    python3 predictor_probe.py --load frames.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doom_td as dt  # noqa: E402
import exrelaxer as exr  # noqa: E402


def target_of(eye, shape, block):
    h, w = shape
    v = np.asarray(eye, dtype=np.float64).reshape(h // block, block, w // block, block).mean(axis=(1, 3)).ravel()
    return np.clip((v - v.mean()) / (v.std() + 1e-6), -3.0, 3.0)


def collect(games, seed, minutes, levels, block):
    p = dict(dt.DEFAULTS)
    p.update(features="both_pool", attack_reflex=1, orient_reflex=1, until_death=minutes, detector=1)
    shape = (120 // p["pool"], 160 // p["pool"])
    pools, targets, gid = [], [], []
    for g in range(games):
        level = levels[g % len(levels)]
        game = dt.make_game(level, seed, p["skill"], labels=True, until_death=minutes)
        learner = dt.Learner(p, game.get_available_buttons_size(), "random", seed)
        names = [b.name for b in game.get_available_buttons()]
        attack, n = names.index("ATTACK"), len(names)
        rng = np.random.default_rng(seed * 1000 + g)
        game.set_seed(seed * 1000 + g)
        game.new_episode()
        learner.res.reset()
        learner.last_pos, learner.path_last, learner.boredom_level, learner.view = None, None, 0.0, None
        learner.anchor_err = []
        a = rng.integers(n)
        while not game.is_episode_finished():
            state = game.get_state()
            learner.features(state, 100.0, dt.PlaceCells.read(game), False)
            pools.append(learner.view.astype(np.float32))
            targets.append(target_of(learner.eye_now, shape, block).astype(np.float32))
            gid.append(g)
            if rng.random() < 0.1:
                a = rng.integers(n)
            buttons = [0] * n
            buttons[a] = 1
            where = dt.monster_ahead(state, p["attack_cone"])
            if where == 0:
                buttons[attack] = 1
            elif where is not None:
                buttons[names.index("TURN_LEFT" if where < 0 else "TURN_RIGHT")] = 1
            game.make_action(buttons, dt.FRAME_SKIP)
        game.close()
        print(f"game {g} ({level}): {len(gid)} frames", flush=True)
    return np.array(pools), np.array(targets), np.array(gid)


def pairs(gid):
    """Indices t with t+1 in the same game."""
    return np.where(gid[:-1] == gid[1:])[0]


def nmse(pred, y):
    return float(np.mean((pred - y) ** 2) / np.mean(y ** 2))


def ridge(x, y, lam=1.0):
    xm = np.c_[x, np.ones(len(x))]
    return np.linalg.solve(xm.T @ xm + lam * np.eye(xm.shape[1]), xm.T @ y)


def apply(w, x):
    return np.c_[x, np.ones(len(x))] @ w


class Online:
    def __init__(self, n_in, n, n_out, trace, seed):
        exr.reseed(seed)
        net = self.net = exr.Network()
        self.h = net.add_layer("plastic", exr.LayerSpec.dense(n, False, True, learning_rule=exr.LearningRule.eprop(trace)))
        net.add_inputs(self.h, n_in, "pool")
        self.o = net.add_layer("out", exr.LayerSpec.dense(n_out, False, False,
                                                          learning_rule=exr.LearningRule.feedback_alignment()))
        net.connect(self.h, self.o)
        net.add_output(self.o)

    def step(self, x):
        self.net.set_inputs("pool", np.asarray(x, dtype=np.float32))
        self.net.step()
        return np.asarray(self.net.layer_output(self.o), dtype=np.float64).ravel().copy()

    def reset(self):
        self.net.reset_state(self.h)
        self.net.reset_traces(self.h)


def run_online(pools, targets, gid, train, test, n, rate, timing, trace=0.9, seed=1):
    m = Online(pools.shape[1], n, targets.shape[1], trace, seed)
    curve = []
    for phase, games in (("train", train), ("test", test)):
        errs = []
        for g in games:
            idx = np.where(gid == g)[0]
            m.reset()
            prev = None
            for t in idx[:-1]:
                pred = m.step(pools[t])
                y = targets[t + 1]
                if phase == "train":
                    if timing == "aligned":
                        m.net.apply_error((y - pred).astype(np.float32), rate)
                    elif prev is not None:  # lag: the previous prediction's error, after this step
                        m.net.apply_error((targets[t] - prev).astype(np.float32), rate)
                errs.append(((pred - y) ** 2).mean() / (y ** 2).mean())
                prev = pred
            if phase == "train":
                curve.append(float(np.mean(errs[-len(idx):])))
        if phase == "test":
            test_err = float(np.mean(errs))
    w_abs = float(np.mean([np.abs(m.net.weights(m.h, i)).mean() for i in range(n)]))
    return {"test": test_err, "train_first": curve[0], "train_last": curve[-1], "w_abs": w_abs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--levels", default="map01,map03,map04")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--out", default="frames.npz")
    ap.add_argument("--load")
    ap.add_argument("--rates", default="0.01,0.001,0.0001")
    ap.add_argument("--sizes", default="32")
    ap.add_argument("--report", default="predictor_probe.json")
    args = ap.parse_args()
    if args.load:
        d = np.load(args.load)
        pools, targets, gid = d["pools"], d["targets"], d["gid"]
    else:
        pools, targets, gid = collect(args.games, args.seed, args.minutes, args.levels.split(","), args.block)
        np.savez_compressed(args.out, pools=pools, targets=targets, gid=gid)
    games = np.unique(gid)
    train, test = games[: int(len(games) * 0.7)], games[int(len(games) * 0.7):]
    tr, te = np.isin(gid, train), np.isin(gid, test)
    pr = pairs(gid)
    ptr, pte = pr[tr[pr]], pr[te[pr]]
    res = {"frames": int(len(gid)), "train_games": len(train), "test_games": len(test)}
    res["copy"] = nmse(targets[pte], targets[pte + 1])
    w = ridge(pools[ptr], targets[ptr + 1])
    res["ridge"] = nmse(apply(w, pools[pte]), targets[pte + 1])
    w0 = ridge(pools[tr], targets[tr])
    res["ridge_now"] = nmse(apply(w0, pools[te]), targets[te])
    print(json.dumps(res), flush=True)
    for n in [int(s) for s in args.sizes.split(",")]:
        for rate in [float(r) for r in args.rates.split(",")]:
            for timing in ("lag", "aligned"):
                r = run_online(pools, targets, gid, train, test, n, rate, timing)
                key = f"online n{n} rate {rate} {timing}"
                res[key] = r
                print(key, json.dumps(r), flush=True)
    json.dump(res, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
