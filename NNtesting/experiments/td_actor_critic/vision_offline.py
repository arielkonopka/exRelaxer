"""Learn vision offline (phase 38): record games, fit what to look for, freeze it.

The TD agent's only learned part is its readout of a frozen random reservoir; §72-§75 showed that
a plastic layer learning in play does not extract more from the frozen vision pool, although the
pool holds the picture. Here a stronger learner (a numpy MLP with Adam, many passes) is fitted
offline on recorded play, then frozen and read by the agent (doom_td `vision_model=path`).

Collect: random games (actions held ~10 steps, strike and orient reflexes from labels, the door
reflex pressing USE) on each level; per step the frozen vision pool (500 values) the agent reads,
and the targets
  monsters  which of 10 screen columns hold a monster's centre (labels)
  depth     log depth per column (the middle third of the depth buffer, 10 columns)
  hurt      health drops within the next 8 steps
  exit      walking distance to the exit, divided by the level's largest reachable distance
            (exitmap.py, ledges respected); NaN where the level has no exit field
Fit: an MLP pool -> hidden (ReLU) -> all targets (logistic for monsters and hurt, linear for the
others), trained on the training levels, tested on frames of levels it never saw.

    python3 vision_offline.py collect --levels map01,map03,map04,map02,map05,map06 --games 20 --out frames.npz
    python3 vision_offline.py fit --data frames.npz[,more.npz] --train map01,map03,map04 --hidden 128 --out vision.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doom_td as dt  # noqa: E402
import vizdoom as vzd  # noqa: E402

HURT_STEPS = 8
COLUMNS = 10


def depth_columns(depth, k=COLUMNS):
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape
    band = d[h // 3: 2 * h // 3]
    return np.log1p(np.array([np.median(band[:, i * w // k:(i + 1) * w // k]) for i in range(k)]))


def doorway_columns(depth, k=40, jump=0.25, widest=8):
    """Narrow openings in view: runs of fine screen columns (k across, log median depth of rows
    h/4..h/2, above the weapon sprite) deeper than the columns on both sides of the run by at least `jump` (log units)
    and at most `widest` columns wide. Returns [(centre column 0..k-1, width, jump)], biggest
    jump first; runs touching a screen edge are not openings (one side is unseen)."""
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape
    band = d[h // 4: h // 2]
    col = np.log1p(np.array([np.median(band[:, i * w // k:(i + 1) * w // k]) for i in range(k)]))
    out = []
    i = 0
    while i < k:
        j = i
        # a run: columns at least `jump` deeper than the column left of where it starts
        if i > 0 and col[i] - col[i - 1] >= jump:
            left = col[i - 1]
            while j + 1 < k and col[j + 1] - left >= jump:
                j += 1
            if j + 1 < k and col[j] - col[j + 1] >= jump and j - i + 1 <= widest:
                run = col[i:j + 1]
                out.append(((i + j) / 2.0, j - i + 1, float(run.min() - max(left, col[j + 1]))))
            i = j + 1
        else:
            i += 1
    out.sort(key=lambda g: -g[2])
    return out


def collect_level(level, games, seed, minutes):
    p = dict(dt.DEFAULTS)
    p.update(features="both_pool", attack_reflex=1, orient_reflex=1, until_death=minutes, detector=1)
    q = dt.doom_rl_params(level, p["skill"], minutes, labels=True)
    q["depth"] = True
    game = dt.drl.make_game(q, seed)
    learner = dt.Learner(p, game.get_available_buttons_size(), "random", seed)
    names = [b.name for b in game.get_available_buttons()]
    attack, use, n = names.index("ATTACK"), names.index("USE"), len(names)
    out = {k: [] for k in ("pool", "monsters", "depth", "health", "exit", "game")}
    field, span = None, None
    for g in range(games):
        rng = np.random.default_rng(seed * 1000 + g)
        game.set_seed(seed * 1000 + g)
        game.new_episode()
        learner.res.reset()
        learner.last_pos, learner.path_last, learner.boredom_level, learner.view = None, None, 0.0, None
        learner.anchor_err = []
        a = rng.integers(n)
        while not game.is_episode_finished():
            state = game.get_state()
            if field is None:
                field = dt.drl.exit_field(level, state, heights=True)
                if field is not None:
                    reach = field.dist[np.isfinite(field.dist)]
                    span = float(reach.max()) if reach.size else None
            pos = dt.PlaceCells.read(game)
            learner.features(state, 100.0, pos, False)
            out["pool"].append(learner.view.astype(np.float32))
            out["monsters"].append(dt.label_columns(state, COLUMNS).astype(np.float32))
            out["depth"].append(depth_columns(state.depth_buffer).astype(np.float32))
            out["health"].append(game.get_game_variable(vzd.GameVariable.HEALTH))
            d = field(pos[0], pos[1]) if field is not None and span else np.nan
            out["exit"].append(np.float32(d / span if np.isfinite(d) else np.nan))
            out["game"].append(g)
            if rng.random() < 0.1:
                a = rng.integers(n)
            buttons = [0] * n
            buttons[a] = 1
            where = dt.monster_ahead(state, p["attack_cone"])
            if where == 0:
                buttons[attack] = 1
            elif where is not None:
                buttons[names.index("TURN_LEFT" if where < 0 else "TURN_RIGHT")] = 1
            if rng.random() < 0.1:
                buttons[use] = 1
            game.make_action(buttons, dt.FRAME_SKIP)
        print(f"{level} game {g}: {len(out['game'])} frames", flush=True)
    game.close()
    out = {k: np.array(v) for k, v in out.items()}
    # hurt: health falls within the next HURT_STEPS steps of the same game
    h, gid = out["health"], out["game"]
    hurt = np.zeros(len(h), np.float32)
    for t in range(len(h)):
        end = t + 1
        while end < len(h) and end <= t + HURT_STEPS and gid[end] == gid[t]:
            if h[end] < h[end - 1]:
                hurt[t] = 1.0
                break
            end += 1
    out["hurt"] = hurt
    return out


def collect(args):
    parts = []
    for i, level in enumerate(args.levels.split(",")):
        o = collect_level(level, args.games, args.seed + i, args.minutes)
        o["level"] = np.array([level] * len(o["game"]))
        o["game"] = o["game"] + 1000 * i
        parts.append(o)
    data = {k: np.concatenate([o[k] for o in parts]) for k in parts[0]}
    np.savez_compressed(args.out, **data)


class MLP:
    """pool -> ReLU hidden -> [monsters (logistic), depth (linear), hurt (logistic), exit (linear)]."""

    def __init__(self, n_in, hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = np.zeros(n_in), np.ones(n_in)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, 2 * COLUMNS + 2))
        self.b2 = np.zeros(2 * COLUMNS + 2)
        self.ymu, self.ysd = np.zeros(COLUMNS + 1), np.ones(COLUMNS + 1)  # depth and exit scaling

    def hidden(self, x):
        return np.maximum(0.0, ((x - self.mu) / self.sd) @ self.W1 + self.b1)

    def heads(self, z):
        o = z @ self.W2 + self.b2
        k = COLUMNS
        return o[:, :k], o[:, k:2 * k], o[:, 2 * k], o[:, 2 * k + 1]

    def predict(self, x):
        m, d, h, e = self.heads(self.hidden(x))
        sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
        return sig(m), d * self.ysd[:COLUMNS] + self.ymu[:COLUMNS], sig(h), e * self.ysd[COLUMNS] + self.ymu[COLUMNS]

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in ("mu", "sd", "W1", "b1", "W2", "b2", "ymu", "ysd")})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(d["W1"].shape[0], d["W1"].shape[1])
        for k in d.files:
            setattr(m, k, d[k])
        return m


def fit_mlp(x, mon, depth, hurt, ex, hidden, epochs, lr, seed, pos_weight=5.0):
    m = MLP(x.shape[1], hidden, seed)
    m.mu, m.sd = x.mean(0), x.std(0) + 1e-6
    y = np.c_[depth, ex]
    ok = np.isfinite(y[:, -1])
    m.ymu = np.r_[depth.mean(0), np.nanmean(ex) if ok.any() else 0.0]
    m.ysd = np.r_[depth.std(0) + 1e-6, (np.nanstd(ex) + 1e-6) if ok.any() else 1.0]
    yd = (depth - m.ymu[:COLUMNS]) / m.ysd[:COLUMNS]
    ye = np.where(ok, (np.nan_to_num(ex) - m.ymu[COLUMNS]) / m.ysd[COLUMNS], 0.0)
    params = ["W1", "b1", "W2", "b2"]
    mom = {k: np.zeros_like(getattr(m, k)) for k in params}
    vel = {k: np.zeros_like(getattr(m, k)) for k in params}
    rng = np.random.default_rng(seed)
    step, batch = 0, 256
    for ep in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), batch):
            j = order[i:i + batch]
            xb = (x[j] - m.mu) / m.sd
            a1 = xb @ m.W1 + m.b1
            z = np.maximum(0.0, a1)
            om, od, oh, oe = m.heads(z)
            pm, ph = 1 / (1 + np.exp(-om)), 1 / (1 + np.exp(-oh))
            wm = np.where(mon[j] > 0, pos_weight, 1.0)
            wh = np.where(hurt[j] > 0, pos_weight, 1.0)
            g = np.zeros((len(j), 2 * COLUMNS + 2))
            g[:, :COLUMNS] = wm * (pm - mon[j])
            g[:, COLUMNS:2 * COLUMNS] = od - yd[j]
            g[:, 2 * COLUMNS] = wh * (ph - hurt[j])
            g[:, 2 * COLUMNS + 1] = ok[j] * (oe - ye[j])
            g /= len(j)
            grads = {"W2": z.T @ g, "b2": g.sum(0)}
            dz = (g @ m.W2.T) * (a1 > 0)
            grads["W1"], grads["b1"] = xb.T @ dz, dz.sum(0)
            step += 1
            for k in params:
                mom[k] = 0.9 * mom[k] + 0.1 * grads[k]
                vel[k] = 0.999 * vel[k] + 0.001 * grads[k] ** 2
                mh, vh = mom[k] / (1 - 0.9 ** step), vel[k] / (1 - 0.999 ** step)
                setattr(m, k, getattr(m, k) - lr * mh / (np.sqrt(vh) + 1e-8))
    return m


def auc(score, y):
    y = y.astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def r2(pred, y):
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(1 - np.mean((pred[ok] - y[ok]) ** 2) / np.var(y[ok]))


def evaluate(m, d, mask):
    pm, pd_, ph, pe = m.predict(d["pool"][mask])
    out = {"frames": int(mask.sum()),
           "monsters_auc": auc(pm.ravel(), d["monsters"][mask].ravel()),
           "depth_r2": r2(pd_.ravel(), d["depth"][mask].ravel()),
           "hurt_auc": auc(ph, d["hurt"][mask]),
           "exit_r2": r2(pe, d["exit"][mask])}
    ex = d["exit"][mask]
    ok = np.isfinite(ex)
    out["exit_corr"] = float(np.corrcoef(pe[ok], ex[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    return out


def fit(args):
    parts = [dict(np.load(f)) for f in args.data.split(",")]
    for i, part in enumerate(parts):
        part["game"] = part["game"] % 1000 + 1000 * i
    d = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    train = np.isin(d["level"], args.train.split(","))
    games = np.unique(d["game"][train])
    held = np.isin(d["game"], games[::5])  # every 5th training game held out
    tr = train & ~held
    m = fit_mlp(d["pool"][tr], d["monsters"][tr], d["depth"][tr], d["hurt"][tr], d["exit"][tr],
                args.hidden, args.epochs, args.lr, args.seed)
    m.save(args.out)
    report = {"hidden": args.hidden, "epochs": args.epochs, "train": evaluate(m, d, tr),
              "held_games": evaluate(m, d, train & held)}
    for level in np.unique(d["level"]):
        if level not in args.train.split(","):
            report[str(level)] = evaluate(m, d, d["level"] == level)
    print(json.dumps(report, indent=1))
    json.dump(report, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--levels", default="map01,map03,map04,map02,map05,map06")
    c.add_argument("--games", type=int, default=20)
    c.add_argument("--minutes", type=float, default=2.0)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", default="frames.npz")
    f = sub.add_parser("fit")
    f.add_argument("--data", default="frames.npz")
    f.add_argument("--train", default="map01,map03,map04")
    f.add_argument("--hidden", type=int, default=128)
    f.add_argument("--epochs", type=int, default=20)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--out", default="vision.npz")
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else fit(args)


if __name__ == "__main__":
    main()
