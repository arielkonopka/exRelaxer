"""Which view can a simple detector find monsters in? (Phase 10 diagnostic.)

Plays random games on map01 with the strike and orienting reflexes driven by ViZDoom's labels
(so monsters come into view), records for every step the screen (60 x 80 gray), the frozen
vision pool the TD learner reads (10 filters x 5 rows x 10 columns) and which of 10 screen
columns hold a monster's centre. Then fits per-column detectors (weights shared across
columns) on the first games and tests them on the rest:

  pool      the 50 pool features of the column (what doom_td's ThreatDetector reads)
  pool3     the column and its two neighbours (150)
  eye       the 30 x 4 strip of the 30 x 40 image the reservoir sees
  screen    the 60 x 8 strip of a 60 x 80 image
  screen+h  screen strip through a 64-unit hidden layer (an MLP)

Reports AUC and recall at 1% false columns.

    python3 detector_probe.py --games 40 --out probe.npz          # collect and fit
    python3 detector_probe.py --load probe.npz                    # fit again
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doom_td as dt  # noqa: E402


def collect(games, seed, minutes, color=False):
    p = dict(dt.DEFAULTS)
    p.update(features="both_pool", attack_reflex=1, orient_reflex=1, until_death=minutes, detector=1)
    game = dt.make_game("map01", seed, p["skill"], labels=True, until_death=minutes, color=color)
    learner = dt.Learner(p, game.get_available_buttons_size(), "random", seed)
    names = [b.name for b in game.get_available_buttons()]
    attack = names.index("ATTACK")
    rng = np.random.default_rng(seed)
    n = len(names)
    screens, pools, cols, gid, colours = [], [], [], [], []
    for g in range(games):
        game.set_seed(seed * 1000 + g)
        game.new_episode()
        learner.res.reset()
        learner.last_pos, learner.path_last, learner.boredom_level, learner.view = None, None, 0.0, None
        learner.anchor_err = []
        a = rng.integers(n)
        while not game.is_episode_finished():
            state = game.get_state()
            learner.features(state, 100.0, dt.PlaceCells.read(game), False)
            screens.append(dt.gray(state.screen_buffer).reshape(60, 2, 80, 2).mean(axis=(1, 3)).astype(np.uint8))
            if color:
                colours.append(dt.colour_columns(state.screen_buffer).astype(np.float32))
            pools.append(learner.view.astype(np.float32))
            cols.append(dt.label_columns(state))
            gid.append(g)
            if rng.random() < 0.1:
                a = rng.integers(n)  # random actions held for ~10 steps
            buttons = [0] * n
            buttons[a] = 1
            where = dt.monster_ahead(state, p["attack_cone"])
            if where == 0:
                buttons[attack] = 1
            elif where is not None:
                buttons[names.index("TURN_LEFT" if where < 0 else "TURN_RIGHT")] = 1
            game.make_action(buttons, dt.FRAME_SKIP)
        print(f"game {g}: {len(gid)} frames, {int(np.sum(cols))} monster columns", flush=True)
    return np.array(screens), np.array(pools), np.array(cols), np.array(gid), np.array(colours)


def column_rows(kind, screens, pools, colours=None):
    """(frames, 10, features) per kind."""
    f = len(pools)
    if kind == "colour":
        return colours
    if kind == "colour3":
        pad = np.pad(colours, ((0, 0), (1, 1), (0, 0)))
        return np.concatenate([pad[:, :-2], pad[:, 1:-1], pad[:, 2:]], axis=2)
    if kind == "pool3colour3":
        return np.concatenate([column_rows("pool3", screens, pools), column_rows("colour3", screens, pools, colours)], 2)
    if kind == "pool":
        return pools.reshape(f, 10, 5, 10).transpose(0, 3, 1, 2).reshape(f, 10, 50)
    if kind == "pool3":
        x = column_rows("pool", screens, pools)
        pad = np.pad(x, ((0, 0), (1, 1), (0, 0)))
        return np.concatenate([pad[:, :-2], pad[:, 1:-1], pad[:, 2:]], axis=2)
    if kind == "eye":
        eye = screens.reshape(f, 30, 2, 40, 2).mean(axis=(2, 4)) / 255.0
        return eye.reshape(f, 30, 10, 4).transpose(0, 2, 1, 3).reshape(f, 10, 120)
    if kind.startswith("screen"):
        s = screens / 255.0
        return s.reshape(f, 60, 10, 8).transpose(0, 2, 1, 3).reshape(f, 10, 480)
    raise ValueError(kind)


def fit(x, y, hidden=0, epochs=300, lr=0.01, seed=0):
    """Logistic regression or one-hidden-layer MLP, full batch Adam, positives weighted to balance."""
    rng = np.random.default_rng(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-3
    x = (x - mu) / sd
    pos = max(1.0, (y == 0).sum() / max(1, (y > 0).sum()))
    wt = np.where(y > 0, pos, 1.0)
    d = x.shape[1]
    params = {}
    if hidden:
        params["W1"] = rng.normal(0, 1 / np.sqrt(d), (d, hidden))
        params["b1"] = np.zeros(hidden)
        params["w"] = rng.normal(0, 1 / np.sqrt(hidden), hidden)
    else:
        params["w"] = np.zeros(d)
    params["b"] = np.zeros(1)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v2 = {k: np.zeros_like(v) for k, v in params.items()}
    for t in range(1, epochs + 1):
        if hidden:
            h = np.tanh(x @ params["W1"] + params["b1"])
        else:
            h = x
        p = 1 / (1 + np.exp(-(h @ params["w"] + params["b"])))
        g = (p - y) * wt / wt.sum()
        grads = {"w": h.T @ g + 1e-4 * params["w"], "b": np.array([g.sum()])}
        if hidden:
            gh = np.outer(g, params["w"]) * (1 - h ** 2)
            grads["W1"] = x.T @ gh + 1e-4 * params["W1"]
            grads["b1"] = gh.sum(0)
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v2[k] = 0.999 * v2[k] + 0.001 * grads[k] ** 2
            params[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)

    def predict(z):
        z = (z - mu) / sd
        hh = np.tanh(z @ params["W1"] + params["b1"]) if hidden else z
        return 1 / (1 + np.exp(-(hh @ params["w"] + params["b"])))
    return predict


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = (y > 0).sum(), (y == 0).sum()
    return (ranks[y > 0].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg)


def recall_at(score, y, false_rate=0.01):
    thr = np.quantile(score[y == 0], 1 - false_rate)
    return float((score[y > 0] > thr).mean()), float(thr)


def evaluate(screens, pools, cols, gid, colours=None, train_frac=0.7):
    games = np.unique(gid)
    cut = games[int(len(games) * train_frac)]
    tr, te = gid < cut, gid >= cut
    print(f"frames train {tr.sum()} test {te.sum()}; monster columns {cols.mean():.3f} of all, "
          f"frames with a monster {(cols.sum(1) > 0).mean():.2f}")
    out = {}
    kinds = (("pool", 0), ("pool3", 0), ("eye", 0), ("screen", 0), ("pool3+h", 64), ("screen+h", 64))
    if colours is not None and len(colours):
        kinds = (("pool3+h", 64), ("colour", 0), ("colour3+h", 64), ("pool3colour3", 0), ("pool3colour3+h", 64))
    for kind, hidden in kinds:
        x = column_rows(kind.split("+")[0], screens, pools, colours)
        xtr, ytr = x[tr].reshape(-1, x.shape[2]), cols[tr].ravel()
        xte, yte = x[te].reshape(-1, x.shape[2]), cols[te].ravel()
        predict = fit(xtr, ytr, hidden=hidden, epochs=400 if hidden else 300)
        s_tr, s_te = predict(xtr), predict(xte)
        r, thr = recall_at(s_te, yte)
        out[kind] = dict(auc_train=auc(s_tr, ytr), auc=auc(s_te, yte), recall_1pct=r,
                         recall_half=float((s_te[yte > 0] > 0.5).mean()), false_half=float((s_te[yte == 0] > 0.5).mean()))
        print(f"{kind:9s} AUC train {out[kind]['auc_train']:.3f} test {out[kind]['auc']:.3f}  "
              f"recall at 1% false {r:.2f}  at p>0.5: recall {out[kind]['recall_half']:.2f} "
              f"false {out[kind]['false_half']:.3f}", flush=True)
    return out


def online(pools, cols, gid, configs, train_frac=0.7, epochs=1):
    """Streams the training games once (or `epochs` times) through doom_td's ThreatDetector,
    learning every frame, as it would in a childhood; then scores the test games."""
    games = np.unique(gid)
    cut = games[int(len(games) * train_frac)]
    tr, te = np.where(gid < cut)[0], np.where(gid >= cut)[0]
    out = {}
    for name, kw in configs.items():
        det = dt.ThreatDetector(**kw)
        for _ in range(epochs):
            for i in tr:
                pr, xh = det.probs(pools[i])
                det.learn(xh, pr, cols[i])
        score = np.array([det.probs(pools[i])[0] for i in te]).ravel()
        y = cols[te].ravel()
        r, thr = recall_at(score, y)
        out[name] = dict(auc=auc(score, y), recall_1pct=r, recall_half=float((score[y > 0] > 0.5).mean()),
                         false_half=float((score[y == 0] > 0.5).mean()))
        print(f"online {name:24s} AUC {out[name]['auc']:.3f}  recall at 1% false {r:.2f} (threshold {thr:.2f})  "
              f"at p>0.5: recall {out[name]['recall_half']:.2f} false {out[name]['false_half']:.4f}", flush=True)
    return out


ONLINE = {
    "logistic pos5 (phase 10)": dict(rate=0.05),
    "logistic n1 replay8": dict(rate=0.05, pos_weight=1, neighbours=1, replay=8),
    "mlp64 n1 replay8 r.05": dict(rate=0.05, pos_weight=1, neighbours=1, hidden=64, replay=8),
    "mlp64 n1 replay8 r.2": dict(rate=0.2, pos_weight=1, neighbours=1, hidden=64, replay=8),
    "mlp64 n1 replay32 r.05": dict(rate=0.05, pos_weight=1, neighbours=1, hidden=64, replay=32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--minutes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="probe.npz")
    ap.add_argument("--load", default=None)
    ap.add_argument("--color", action="store_true", help="RGB screen; adds colour-opponent column features")
    ap.add_argument("--online", action="store_true", help="only the online (doom_td) detectors")
    args = ap.parse_args()
    if args.load:
        d = np.load(args.load)
        data = d["screens"], d["pools"], d["cols"], d["gid"], (d["colours"] if "colours" in d else None)
    else:
        data = collect(args.games, args.seed, args.minutes, args.color)
        np.savez_compressed(args.out, screens=data[0], pools=data[1], cols=data[2], gid=data[3], colours=data[4])
    if args.online:
        out = online(data[1], data[2], data[3], ONLINE)
    else:
        out = evaluate(*data)
    import json
    with open(os.path.splitext(args.load or args.out)[0] + ("_online" if args.online else "") + ".json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
