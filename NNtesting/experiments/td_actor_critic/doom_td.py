#!/usr/bin/env python3
"""Validation steps 3 and 4: ViZDoom `basic` and `defend_the_center`, learned
online (learning always on) by the TD actor-critic over a frozen E-R reservoir.

    screen 160 x 120 gray -> pooled 40 x 30 ("eye", 1200 values in [0, 1])
       -> frozen random E-R reservoir (reads the eye and itself one tick late;
          recovery spread over recovery +- spread)
       features = reservoir outputs + thresholds (+ the eye with features=both_eye)
          |-> critic V(s)              TD(lambda)
          |-> one actor readout per button, softmax, delta x (onehot - pi) x pre

The scenario's own buttons and reward (basic: -1 per tic, +100 kill, -5 a
missed shot; defend_the_center: +1 kill, -1 death), times reward_scale so
that |r| <= 1. Whole maps (--scenario map01): doom_rl's 8 buttons and its
shaped reward (doom_rl/rewards.py) x 0.2. Frame skip 4. A kill that ends `basic` and a death are
terminal (V = 0); the time limit is not (the last state bootstraps).

    doom_td.py --out DIR --scenario basic --arm td|reward|random [--episodes 1000] [--seed 1] [--set k=v]

With --set until_death=M a game has no time limit: it ends at death (or after M minutes).

DIR/log.jsonl: one line per training episode (return, kills, length and survival seconds, mean |delta|, mean V,
entropy, off-greedy share, spikes per step); DIR/test.json: greedy test episodes on fresh
seeds after training; DIR/values_<episode>.json: a V(s) and delta trace every `trace_every`.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exrelaxer as exr
import vizdoom as vzd
from exrelaxer.actor_critic import Agent
from common import Reservoir

sys.path.insert(0, os.path.join(HERE, "..", "doom_rl"))
import experiment as drl  # doom_rl: whole maps, its buttons and reward shaper

FRAME_SKIP = 4
DEFAULTS = {
    "width": 256, "model": "er", "recurrent_scale": 0.5, "recovery": 0.745, "spread": 0.245, "ticks": 1,
    "features": "both",  # reservoir: output, threshold, both; both_eye: + the pooled screen; eye: the screen only; pool, both_pool: the frozen vision features (+ reservoir); oracle: the monster's screen column (no vision, a control)
    "pool": 4, "skill": None,
    "vision": "gabor",  # gabor: frozen Gabor + centre-surround bank, max-pooled, before the reservoir; none: raw pixels
    "gamma": 0.97, "lam": 0.9, "critic_rate": 0.03, "actor_rate": 0.03, "tau": 0.03, "clip": 1.0,
    "post": "advantage", "entropy": 0.0, "center": 0.0,
    "reward_scale": None, "test": 30, "trace_every": 100,
    "until_death": 0,  # > 0: no time limit, games end at death or after this many minutes
    # Phase 2, homeostatic drives. reward: game (the scenario's), drive (drive reduction
    # D(h_prev) - D(h_now), D(h) = |target - h| / 100) or drive_level (-D(h_now) x drive_cost every step).
    "reward": "game", "drives": 0,  # drives 1: health / 100 is an input of the reservoir and a raw feature
    "drive_gain": 5.0, "health_target": 100.0, "drive_cost": 0.05,
    # Phase 3, a place layer: place=1 appends place cells (tile coding of the map position,
    # `tilings` offset grids of `place_size` map units hashed into `place_bins`) and
    # `headings` head-direction cells to the features (place_input=1: also into the reservoir).
    "place": 0, "place_size": 128.0, "tilings": 4, "place_bins": 1024, "headings": 8, "place_input": 0,
    # A lifetime novelty bonus: novelty / sqrt(1 + N) for entering a `novelty_cell`-unit square
    # entered N times before in all training games (not reset between games; none in tests).
    "novelty": 0.0, "novelty_cell": 64.0,
    # Learned place: motion=1 feeds self-motion (forward and sideways speed, turning) into the
    # reservoir; vplace=K appends K learned place cells, prototypes of the reservoir's state
    # and the vision features found by online competitive learning (no game coordinates).
    "motion": 0, "motion_gain": 5.0, "vplace": 0, "vplace_rate": 0.02, "vplace_active": 4,
    "vplace_settle": 0,
    # A door reflex: door_reflex=1 presses USE (with the chosen action) whenever the last step
    # moved the player less than 1 map unit although it tried to move: pushing against a wall.
    "door_reflex": 0,
    # Path integration: path=1 gives the place cells a position integrated from self-motion
    # (each step's forward and sideways movement and turn, each with path_noise relative noise),
    # starting from (0, 0, 0) at every spawn, instead of the game's coordinates.
    "path": 0, "path_noise": 0.05,
    # path=2: anchored by landmarks. `landmarks` view prototypes (the vision features, online
    # competitive learning) remember the integrated pose where they win; a life starts lost (no
    # place cells) and is placed when a view is recognised (similarity > landmark_match, the
    # prototype seen >= landmark_seen times while placed); placed, recognised views pull the
    # estimate back by landmark_pull. The first life ever is placed at its spawn.
    "landmarks": 256, "landmark_match": 0.97, "landmark_seen": 5, "landmark_pull": 0.1, "landmark_agree": 64.0,
    # landmark_seq=L > 1: a landmark is the sequence of the last L distinct winning views, not one
    # view; landmark_spread > 0: a landmark is trusted only while the poses it was seen at stay
    # within that many map units (standard deviation), so look-alikes seen in two places are dropped.
    "landmark_seq": 1, "landmark_spread": 0.0,
    # landmark_guided=N: during the first N lives the agent is told its spawn pose (a guided
    # childhood: landmarks learn correct poses); after that it starts lost as before.
    "landmark_guided": 0,
    # Reasons to go on (whole maps): approach = doom_rl's w_approach (reward per 64 units of new
    # closest walking distance to the exit); boredom = c: a drive that grows while no new square
    # is entered, r -= c * min(1, steps since a new square / boredom_steps), also felt (a raw feature).
    "approach": 0.0, "boredom": 0.0, "boredom_steps": 100,
    # A killer instinct: attack_reflex=1 adds ATTACK to the chosen action whenever a monster's
    # centre is within the middle `attack_cone` of the screen width; kill_weight sets doom_rl's
    # w_kill (reward per kill, default 1).
    "attack_reflex": 0, "attack_cone": 0.125, "kill_weight": 1.0,
    # orient_reflex=1: also add TURN_LEFT / TURN_RIGHT toward the biggest monster on screen when
    # it is outside the cone (the orienting reflex: the head turns to a threat).
    "orient_reflex": 0,
    # A learned threat detector: detector=N games during which a logistic detector, one set of
    # weights shared by the 10 screen columns of the frozen vision features (10 filters x 5
    # rows), learns from the labels which columns hold a monster; from game N on the reflexes
    # use the detector (threshold detector_threshold) instead of the labels.
    # detector_hidden: tanh units before the logistic unit (0 = logistic); detector_neighbours:
    # columns on each side it also reads; detector_pos: weight of a column with a monster.
    "detector": 0, "detector_rate": 0.05, "detector_threshold": 0.5, "detector_hidden": 0,
    "detector_neighbours": 0, "detector_pos": 5.0, "detector_replay": 0,
}
SCALES = {"my_way_home": 1.0, "basic": 0.01, "defend_the_center": 1.0, "map01": 0.2, "health_gathering": 0.01,
          "health_gathering_supreme": 0.01}


MAPS = ("map01",)  # whole levels: doom_rl's 8 buttons and shaped reward (doom_rl/rewards.py)


def doom_rl_params(scenario, skill, until_death, approach=0.0, labels=False, kill=None):
    """doom_rl's parameters for a whole map, without sound."""
    q = {k: v[0] for k, v in drl.PARAMS.items()}
    q.update(scenario=scenario, sound=False, color=False, w_approach=float(approach), labels=bool(labels))
    if kill is not None:
        q["w_kill"] = float(kill)
    if skill is not None:
        q["skill"] = int(skill)
    if until_death > 0:
        q["episode_tics"] = int(until_death * 60 * 35)
    return q


def make_game(scenario, seed, skill=None, labels=False, until_death=0, approach=0.0):
    if scenario in MAPS:
        return drl.make_game(doom_rl_params(scenario, skill, until_death, approach, labels), seed)
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_sound_enabled(False)
    game.set_labels_buffer_enabled(labels)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE):
        game.add_available_game_variable(v)
    if skill is not None:
        game.set_doom_skill(skill)
    if until_death > 0:
        # Play until killed: no time limit but a safety cap of `until_death` minutes.
        game.set_episode_timeout(int(until_death * 60 * 35))
    game.set_seed(seed)
    game.init()
    return game


ORACLE_BINS = 17


def oracle_features(state):
    """A control that skips vision: where the nearest monster's centre is on the screen, one-hot
    in 16 columns (0 when none is seen), plus 1 (a bias)."""
    x = np.zeros(ORACLE_BINS)
    x[-1] = 1.0
    for l in state.labels:
        if l.object_name != "DoomPlayer":
            x[min(15, int((l.x + l.width / 2) * 16 / 160))] = 1.0
    return x


class PlaceCells:
    """Place and head-direction cells from the map position: `tilings` grids of squares
    `size` map units wide, each shifted by size / tilings, so a position lights one square per
    grid (hashed into `bins`; nearby positions share squares), plus `headings` cells tuned to
    the facing direction (max(0, cos)^2 bumps). A linear critic over them is a smoothed table
    of values per place, as in Foster, Morris & Dayan (2000)."""

    def __init__(self, size=128.0, tilings=4, bins=1024, headings=8):
        self.size, self.tilings, self.bins, self.headings = float(size), int(tilings), int(bins), int(headings)
        self.n = self.bins + self.headings
        self.angles = np.arange(self.headings) * 2 * np.pi / max(1, self.headings)

    def __call__(self, x, y, angle):
        v = np.zeros(self.n)
        for t in range(self.tilings):
            off = t * self.size / self.tilings
            i, j = int(np.floor((x + off) / self.size)), int(np.floor((y + off) / self.size))
            v[hash((t, i, j)) % self.bins] = 1.0
        if self.headings:
            v[self.bins:] = np.maximum(0.0, np.cos(np.radians(angle) - self.angles)) ** 2
        return v

    @staticmethod
    def read(game):
        return tuple(game.get_game_variable(g) for g in
                     (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE))


class Landmarks:
    """Recognised views anchor path integration, as landmarks reset grid cells: each view
    prototype keeps a running mean of the pose (x, y and heading as cos / sin) at which it won
    while the agent knew where it was."""

    def __init__(self, k, rate=0.02, match=0.97, seen=5, pull=0.1, seq=1, spread=0.0):
        self.view = LearnedPlace(k, 0, rate, 1, settle=20000)
        self.match, self.seen, self.pull = float(match), int(seen), float(pull)
        self.seq, self.spread = int(seq), float(spread)
        # Per landmark (a view, or a sequence of views): count, running mean of x, y, cos, sin,
        # and the running variance of x and y.
        self.stats = {}
        self.recent = []

    def reset(self):
        self.recent = []

    def recognise(self, view, learn):
        """The current landmark (the winning view, or the last `seq` distinct winning views) and
        whether it is recognised: similar enough, seen often enough, and (with `spread`) always
        seen at about the same place."""
        self.view(view, learn)
        w = self.view.winner
        if w is None:
            return None, False
        sim = float(self.view.W[w] @ (view / (np.linalg.norm(view) + 1e-9)))
        if sim <= self.match:
            return None, False
        if not self.recent or self.recent[-1] != w:
            self.recent = (self.recent + [w])[-self.seq:]
        if len(self.recent) < self.seq:
            return None, False
        key = tuple(self.recent)
        st = self.stats.get(key)
        known = st is not None and st[0] >= self.seen
        if known and self.spread > 0:
            known = np.sqrt(st[5] + st[6]) < self.spread
        return key, known

    def remember(self, key, est):
        a = np.radians(est[2])
        target = np.array([est[0], est[1], np.cos(a), np.sin(a)])
        st = self.stats.setdefault(key, np.zeros(7))
        st[0] += 1
        rate = 1.0 / min(st[0], 20.0)
        d = target - st[1:5]
        st[1:5] += rate * d
        st[5] += rate * (d[0] * (target[0] - st[1]) - st[5])  # running variance of x
        st[6] += rate * (d[1] * (target[1] - st[2]) - st[6])

    def recall(self, key):
        x, y, c, s = self.stats[key][1:5]
        return np.array([x, y, np.degrees(np.arctan2(s, c)) % 360.0])


class LearnedPlace:
    """Place cells learned without coordinates: K prototypes of the (unit-length) feature vector,
    learned by online competitive learning (the winner moves toward the input, a local
    Hebbian rule); the `active` closest prototypes fire (1), the rest stay 0. With the
    reservoir's thresholds in the input, a prototype stands for "what I see and what I just
    did", which in a level mostly means where I am."""

    def __init__(self, k, size, rate=0.02, active=4, rng=None, settle=0):
        self.k, self.rate, self.active, self.settle = int(k), float(rate), int(active), float(settle)
        self.updates = 0
        self.W = None
        self.size = size
        self.filled = 0
        self.winner = None
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, x, learn=True):
        x = np.asarray(x, dtype=np.float64)
        x = x / (np.linalg.norm(x) + 1e-9)
        if self.W is None:
            self.W = np.zeros((self.k, len(x)))
        code = np.zeros(self.k)
        if self.filled < self.k:  # the first K distinct states seed the prototypes
            if learn:
                self.W[self.filled] = x
                self.filled += 1
            return code
        sims = self.W @ x
        self.winner = int(np.argmax(sims))
        top = np.argpartition(-sims, self.active - 1)[:self.active]
        code[top] = 1.0
        if learn:
            w = int(np.argmax(sims))
            rate = self.rate * self.settle / (self.settle + self.updates) if self.settle > 0 else self.rate
            self.updates += 1
            self.W[w] += rate * (x - self.W[w])
            self.W[w] /= np.linalg.norm(self.W[w]) + 1e-9
        return code


MONSTERS = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
            "BaronOfHell", "HellKnight", "Revenant", "Arachnotron", "Fatso", "PainElemental", "Archvile",
            "Cyberdemon", "SpiderMastermind", "WolfensteinSS", "MarineChainsawVzd"}


def monster_ahead(state, cone):
    """0 when a monster's centre is within the middle `cone` of the screen width (labels buffer),
    -1 / +1 when the biggest monster on screen is to the left / right, None when none is seen."""
    width = state.screen_buffer.shape[-1]
    lo, hi = width * (0.5 - cone / 2), width * (0.5 + cone / 2)
    seen = [l for l in state.labels or [] if l.object_name in MONSTERS]
    if not seen:
        return None
    if any(lo <= l.x + l.width / 2 <= hi for l in seen):
        return 0
    big = max(seen, key=lambda l: l.width * l.height)
    return -1 if big.x + big.width / 2 < lo else 1


class ThreatDetector:
    """Monster-or-not for each screen column, from the frozen vision features of that column and
    `neighbours` columns on each side (weights shared across columns, like a convolution): a
    logistic unit, or with `hidden` > 0 one tanh hidden layer before it. Taught online by the
    labels during a childhood, then on its own. Monsters are rare (about 0.2% of columns), so a
    column with one counts `pos_weight` times in the error."""

    def __init__(self, channels=10, rows=5, columns=10, rate=0.05, hidden=0, neighbours=0, pos_weight=5.0,
                 seed=0, replay=0, replay_size=2000):
        self.c, self.r, self.k, self.rate = channels, rows, columns, float(rate)
        # replay > 0: keep the last replay_size columns with and without a monster and relearn
        # `replay` of each kind every tick (a rehearsal, as in sleep replay), so the rare
        # positives are seen many times.
        self.replay, self.replay_size = int(replay), int(replay_size)
        self.pos_buf, self.neg_buf = [], []
        self.rng = np.random.default_rng(seed + 7)
        self.n, self.hidden, self.pos_weight = int(neighbours), int(hidden), float(pos_weight)
        d = channels * rows * (2 * self.n + 1)
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, self.hidden)) if self.hidden else None
        self.b1 = np.zeros(self.hidden)
        self.w = rng.normal(0, 1 / np.sqrt(self.hidden), self.hidden) if self.hidden else np.zeros(d)
        self.b = -2.0
        self.mean, self.var = None, None

    def columns(self, pool):
        x = np.asarray(pool, dtype=np.float64).reshape(self.c, self.r, self.k)
        x = x.transpose(2, 0, 1).reshape(self.k, -1)  # one row of features per column
        if self.mean is None:
            self.mean, self.var = x.mean(0), x.var(0) + 1e-3
        else:  # slow running statistics over all columns seen
            self.mean += 0.001 * (x.mean(0) - self.mean)
            self.var += 0.001 * (((x - self.mean) ** 2).mean(0) - self.var)
        x = (x - self.mean) / np.sqrt(self.var + 1e-6)
        if self.n:
            pad = np.pad(x, ((self.n, self.n), (0, 0)))
            x = np.concatenate([pad[i:i + self.k] for i in range(2 * self.n + 1)], axis=1)
        return x

    def probs(self, pool):
        x = self.columns(pool)
        p, h = self.forward(x)
        return p, (x, h)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1) if self.hidden else x
        return 1.0 / (1.0 + np.exp(-(h @ self.w + self.b))), h

    def learn(self, xh, p, target):
        x, h = xh
        if self.replay:
            for row, t in zip(x, target):
                buf = self.pos_buf if t > 0 else self.neg_buf
                if len(buf) < self.replay_size:
                    buf.append(row)
                else:
                    buf[self.rng.integers(self.replay_size)] = row
            if self.pos_buf:
                pi = self.rng.integers(len(self.pos_buf), size=self.replay)
                ni = self.rng.integers(len(self.neg_buf), size=self.replay)
                x = np.concatenate([x, np.array([self.pos_buf[i] for i in pi]), np.array([self.neg_buf[i] for i in ni])])
                target = np.concatenate([target, np.ones(self.replay), np.zeros(self.replay)])
                p, h = self.forward(x)
        g = (target - p) * np.where(target > 0, self.pos_weight, 1.0)
        if self.hidden:
            gh = np.outer(g, self.w) * (1 - h ** 2)
            self.W1 += self.rate * (x.T @ gh) / len(g)
            self.b1 += self.rate * gh.mean(0)
        self.w += self.rate * (g @ h) / len(g)
        self.b += self.rate * g.mean()

    def where(self, p, cone, threshold):
        """As monster_ahead: 0 for a monster in the middle cone, -1 / +1 for the strongest one left /
        right, None for none."""
        k = len(p)
        lo, hi = int(np.floor(k * (0.5 - cone / 2))), int(np.ceil(k * (0.5 + cone / 2)))
        hot = p > threshold
        if not hot.any():
            return None
        if hot[lo:hi].any():
            return 0
        return -1 if int(np.argmax(p)) < lo else 1


def label_columns(state, k=10):
    """Which of k screen columns hold a monster's centre (the detector's teacher)."""
    t = np.zeros(k)
    width = state.screen_buffer.shape[-1]
    for l in state.labels or []:
        if l.object_name in MONSTERS:
            t[min(k - 1, int((l.x + l.width / 2) * k / width))] = 1.0
    return t


def pool_screen(screen, pool):
    return screen.reshape(120 // pool, pool, 160 // pool, pool).mean(axis=(1, 3)).ravel().astype(np.float32) / 255.0


class Learner:
    def __init__(self, p, buttons, arm, seed):
        exr.reseed(seed)
        self.eye_size = (120 // p["pool"]) * (160 // p["pool"])
        kind = {"both_eye": "both", "eye": "none", "oracle": "none", "both_pool": "both", "pool": "none",
                "place": "none"}.get(
            p["features"], p["features"])
        self.oracle = p["features"] == "oracle"
        self.drives = bool(p["drives"])
        extra = ORACLE_BINS if self.oracle else self.eye_size if p["features"] in ("both_eye", "eye") else 0
        self.place = PlaceCells(p["place_size"], p["tilings"], p["place_bins"], p["headings"]) \
            if p["place"] or p["features"] == "place" else None
        direct = {"drive": 1} if self.drives else {}
        if p["motion"]:
            direct["motion"] = 3
        if self.place is not None:
            extra += self.place.n
            if p["place_input"]:
                direct["place"] = self.place.n
        self.res = Reservoir({"eye": self.eye_size}, width=p["width"], recurrent_scale=p["recurrent_scale"],
                             recovery=p["recovery"], spread=p["spread"], model=p["model"], features=kind, ticks=p["ticks"],
                             vision=dict(height=120 // p["pool"], width=160 // p["pool"]) if p["vision"] == "gabor" else None,
                             with_pool=p["features"] in ("both_pool", "pool"),
                             extra=extra + int(self.drives) + int(p["boredom"] > 0), direct=direct or None,
                             direct_gain=p["motion_gain"] if p["motion"] else p["drive_gain"])
        self.last_pos = None
        self.boredom_level = 0.0
        self.landmarks = Landmarks(p["landmarks"], match=p["landmark_match"], seen=p["landmark_seen"],
                                   pull=p["landmark_pull"], seq=p["landmark_seq"],
                                   spread=p["landmark_spread"]) if p["path"] == 2 else None
        self.view = None  # the last vision features (landmarks recognise them one tick late)
        self.detector = ThreatDetector(rate=p["detector_rate"], hidden=p["detector_hidden"],
                                       neighbours=p["detector_neighbours"], pos_weight=p["detector_pos"],
                                       replay=p["detector_replay"], seed=seed) if p["detector"] else None
        self.games = 0
        self.lives = 0
        self.origin = None  # the first life's spawn: the shared frame (for the error measure)
        self.anchor_err = []
        self.vplace = LearnedPlace(p["vplace"], 0, p["vplace_rate"], p["vplace_active"],
                                                                      settle=p["vplace_settle"]) if p["vplace"] else None
        self.arm, self.p, self.buttons = arm, p, buttons
        self.rng = np.random.default_rng(seed)
        self.agent = Agent(self.res.size + p["vplace"], buttons, p["gamma"], p["lam"], p["critic_rate"], p["actor_rate"],
                           clip=p["clip"], modulator="reward" if arm == "reward" else "td", post=p["post"],
                           tau=p["tau"], entropy=p["entropy"], center=p["center"], rng=self.rng)

    def features(self, state, health=None, pos=None, learn=True):
        inputs, raw = {}, []
        if self.p["motion"]:
            inputs["motion"] = self.motion(pos)
        if self.place is not None:
            if self.p["path"] == 2:
                est = self.anchored(pos, learn)
                cells = self.place(*est) if est is not None else np.zeros(self.place.n)
            else:
                cells = self.place(*(self.integrate(pos) if self.p["path"] else pos))
            if self.p["place_input"]:
                inputs["place"] = cells
        if self.drives:
            inputs["drive"] = [health / 100.0]
        if self.oracle:
            inputs["eye"] = np.zeros(self.eye_size)
            raw.append(oracle_features(state))
        else:
            inputs["eye"] = pool_screen(state.screen_buffer, self.p["pool"])
            if self.p["features"] in ("both_eye", "eye"):
                raw.append(inputs["eye"])
        if self.drives:
            raw.append([health / 100.0])
        if self.place is not None:
            raw.append(cells)
        if self.p["boredom"] > 0:
            raw.append([self.boredom_level])
        x = self.res.step(inputs, raw=np.concatenate(raw) if raw else None)
        if self.landmarks is not None or self.detector is not None:
            self.view = np.asarray(self.res.net.layer_output(self.res.pool), dtype=np.float64).ravel()
        if self.vplace is not None:
            # The learned place cells read the reservoir and the vision features, not the raw inputs.
            x = np.concatenate([x, self.vplace(x[:len(x) - self.res.extra], learn)])
        return x

    def anchored(self, pos, learn):
        """path=2: integrate self-motion once placed; get placed (or corrected) by recognised views.
        Returns the pose estimate in the shared frame, or None while lost."""
        first = getattr(self, "path_last", None) is None
        if first:
            self.lives += 1
            self.placed = self.lives == 1
            if self.origin is None:
                self.origin = pos
        if first:
            self.candidate = None
            self.landmarks.reset()
        last = getattr(self, "path_last", None)
        est = self.integrate(pos)  # moves self.path_est (meaningful only while placed)
        if first and 1 < self.lives <= self.p["landmark_guided"]:
            self.path_est[:] = self.true_pose(pos)
            self.placed = True
        c = getattr(self, "candidate", None)
        if c is not None and last is not None:
            # The candidate pose moves with the agent too (same dead reckoning, its own heading).
            dx, dy = pos[0] - last[0], pos[1] - last[1]
            a = np.radians(last[2])
            fwd, side = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
            b = np.radians(c[2])
            c[0] += fwd * np.cos(b) - side * np.sin(b)
            c[1] += fwd * np.sin(b) + side * np.cos(b)
            c[2] = (c[2] + (pos[2] - last[2] + 180.0) % 360.0 - 180.0) % 360.0
        lm = self.landmarks
        if self.view is not None and not first:
            w, known = lm.recognise(self.view, learn)
            if w is not None:
                if known and not self.placed:
                    # A guess from one view is checked against the next recognised view: placed
                    # only when two agree (after integrating the movement between them) within
                    # landmark_agree map units, so one look-alike room cannot misplace the agent.
                    r = lm.recall(w)
                    c = getattr(self, "candidate", None)
                    if c is not None and np.hypot(*(c[:2] - r[:2])) < self.p["landmark_agree"] and w != self.candidate_w:
                        self.path_est[:] = r
                        self.placed = True
                        self.candidate = None
                    else:
                        self.candidate, self.candidate_w = r.copy(), w
                elif known and lm.pull > 0:
                    r = lm.recall(w)
                    e = self.path_est
                    e[:2] += lm.pull * (r[:2] - e[:2])
                    e[2] = (e[2] + lm.pull * ((r[2] - e[2] + 180.0) % 360.0 - 180.0)) % 360.0
                if self.placed and learn:
                    lm.remember(w, self.path_est)
        if self.placed:
            self.anchor_err.append(self.frame_error(pos))
            return tuple(self.path_est)
        return None

    def true_pose(self, pos):
        """The true pose in the shared frame (the first life's spawn at the origin, facing 0)."""
        ox, oy, oa = self.origin
        a = np.radians(oa)
        dx, dy = pos[0] - ox, pos[1] - oy
        return np.array([dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a), (pos[2] - oa) % 360.0])

    def frame_error(self, pos):
        """Distance between the estimate and the true position in the shared frame (the first
        life's spawn at the origin facing 0)."""
        ox, oy, oa = self.origin
        a = np.radians(oa)
        dx, dy = pos[0] - ox, pos[1] - oy
        tx, ty = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
        return float(np.hypot(self.path_est[0] - tx, self.path_est[1] - ty))

    def integrate(self, pos):
        """Dead reckoning: the position and heading the agent would estimate from its own
        movement since the spawn (with noise), in its own frame (the spawn is the origin, facing
        angle 0). Only the step's movement relative to the facing direction is used."""
        last, self.path_last = getattr(self, "path_last", None), pos
        if last is None:
            self.path_est = np.zeros(3)
            return tuple(self.path_est)
        dx, dy = pos[0] - last[0], pos[1] - last[1]
        a = np.radians(last[2])
        fwd, side = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
        turn = (pos[2] - last[2] + 180.0) % 360.0 - 180.0
        noise = 1.0 + self.p["path_noise"] * self.rng.standard_normal(3)
        fwd, side, turn = fwd * noise[0], side * noise[1], turn * noise[2]
        e = self.path_est
        b = np.radians(e[2])
        e[0] += fwd * np.cos(b) - side * np.sin(b)
        e[1] += fwd * np.sin(b) + side * np.cos(b)
        e[2] = (e[2] + turn) % 360.0
        return tuple(e)

    def motion(self, pos):
        """Self-motion since the last step, as the vestibular sense would give it: forward and
        sideways speed (map units / 20) and turning (degrees / 45), clipped to [-1, 1]."""
        last, self.last_pos = self.last_pos, pos
        if last is None or pos is None:
            return np.zeros(3)
        dx, dy = pos[0] - last[0], pos[1] - last[1]
        a = np.radians(pos[2])
        turn = (pos[2] - last[2] + 180.0) % 360.0 - 180.0
        v = np.array([dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a), 0.0]) / 20.0
        v[2] = turn / 45.0
        return np.clip(v, -1.0, 1.0)

    def novelty_bonus(self, pos):
        """Count-based novelty over the whole life of the agent (kept on the agent, so a
        checkpoint carries it): only a square entered from another one counts."""
        counts = self.agent.__dict__.setdefault("novelty_counts", {})
        size = self.p["novelty_cell"]
        cell = (int(pos[0] // size), int(pos[1] // size))
        if cell == getattr(self, "_last_cell", None):
            return 0.0
        self._last_cell = cell
        n = counts.get(cell, 0)
        counts[cell] = n + 1
        return self.p["novelty"] / np.sqrt(1.0 + n)

    def drive(self, health):
        """D(h) = |target - h| / 100: how far health is from where the body wants it."""
        return abs(self.p["health_target"] - max(0.0, health)) / 100.0

    def act(self, x, reward, terminal, learn):
        a = self.agent.step(x, reward, terminal, learn=learn and self.arm != "random", explore=learn)
        if self.arm == "random" and not terminal:
            a = int(self.rng.integers(self.buttons))
        return a


def episode(game, learner, scale, learn, trace=False, shaper=None):
    """One game. `shaper` (doom_rl's rewards.Shaper, whole maps): the reward comes from it
    instead of the scenario's own."""
    game.new_episode()
    learner.res.reset()
    learner.agent.end_episode()
    learner.agent.log.new()
    learner.res.spikes = 0
    eye = np.eye(learner.buttons, dtype=int).tolist()
    names = [str(b).split(".")[-1] for b in game.get_available_buttons()]
    use = names.index("USE") if "USE" in names else None
    moving = {i for i, n in enumerate(names) if n.startswith("MOVE_")}
    reflex = bool(learner.p["door_reflex"]) and use is not None
    attack = names.index("ATTACK") if "ATTACK" in names else None
    instinct = bool(learner.p["attack_reflex"]) and attack is not None
    strikes = 0
    det_hits = det_false = det_true = 0
    if learn:
        learner.games += 1
    blocked, reflexes = False, 0
    state = game.get_state()
    if shaper is not None:
        shaper.reset(state)
    kind = learner.p["reward"]
    health = game.get_game_variable(vzd.GameVariable.HEALTH)
    medkits, game_total, novelty = 0, 0.0, 0.0
    pos = PlaceCells.read(game)
    cells = set()
    learner.last_pos = None
    learner.path_last = None
    learner.boredom_level = 0.0
    learner.view = None
    learner.anchor_err = []
    fields = []  # test games: (winning learned place cell, x, y)
    x = learner.features(state, health, pos, learn)
    a = learner.act(x, 0.0, False, learn)
    steps = 0
    while True:
        buttons = eye[a]
        if reflex and blocked:
            buttons = list(buttons)
            buttons[use] = 1
            reflexes += 1
        where = monster_ahead(state, learner.p["attack_cone"]) if instinct and state is not None else None
        det = learner.detector
        if det is not None and instinct and state is not None and learner.view is not None:
            probs, xcols = det.probs(learner.view)
            truth = label_columns(state)
            guess = probs > learner.p["detector_threshold"]
            det_hits += int((guess & (truth > 0)).sum())
            det_false += int((guess & (truth == 0)).sum())
            det_true += int(truth.sum())
            if learn and learner.games < learner.p["detector"]:
                det.learn(xcols, probs, truth)  # childhood: the labels teach
            else:
                where = det.where(probs, learner.p["attack_cone"], learner.p["detector_threshold"])
        if where == 0:
            buttons = list(buttons)
            buttons[attack] = 1
            strikes += 1
        elif where is not None and learner.p["orient_reflex"]:
            buttons = list(buttons)
            buttons[names.index("TURN_LEFT" if where < 0 else "TURN_RIGHT")] = 1
        before_pos = PlaceCells.read(game)
        r = game.make_action(buttons, FRAME_SKIP) * scale
        game_total += r / scale
        steps += 1
        done = game.is_episode_finished()
        before, health = health, game.get_game_variable(vzd.GameVariable.HEALTH)
        medkits += int(health > before)
        if kind == "drive":
            r = learner.drive(before) - learner.drive(health)
        elif kind == "drive_level":
            r = -learner.drive(health) * learner.p["drive_cost"]
        if shaper is not None:
            if done:
                r = shaper.end(game) * scale
            else:
                state = game.get_state()
                r = shaper.step(state) * scale
                if learner.p["boredom"] > 0:
                    learner.boredom_level = min(1.0, shaper.since_new / learner.p["boredom_steps"])
                    if learn:
                        r -= learner.p["boredom"] * learner.boredom_level
                        novelty -= learner.p["boredom"] * learner.boredom_level  # intrinsic, not the game's
        if done:
            if game.get_episode_time() >= game.get_episode_timeout() and not game.is_player_dead():
                # A time limit is not a terminal state: the last state seen bootstraps
                # (V(s_T) as if play went on), else running out the clock would
                # look like relief from the living cost (delta ~ -V > 0).
                learner.agent.step(x, r, False, learn=learn and learner.arm != "random", explore=False)
                learner.agent.end_episode()
            else:
                learner.act(np.zeros(len(learner.agent.critic.w)), r, True, learn)
            break
        if shaper is None:
            state = game.get_state()
        pos = PlaceCells.read(game)
        blocked = a in moving and np.hypot(pos[0] - before_pos[0], pos[1] - before_pos[1]) < 1.0
        cells.add((int(pos[0] // 64), int(pos[1] // 64)))
        if learn and learner.p["novelty"] > 0:
            bonus = learner.novelty_bonus(pos)
            r += bonus
            novelty += bonus
        x = learner.features(state, health, pos, learn)
        if not learn and learner.vplace is not None and learner.vplace.winner is not None:
            fields.append((learner.vplace.winner, pos[0], pos[1]))
        a = learner.act(x, r, False, learn)
    total = game_total if shaper is None else (sum(learner.agent.log.rewards) - novelty) / scale
    kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)  # still readable after the end
    died = int(game.is_player_dead())
    s = learner.agent.log.summary()
    s.update(game_reward=float(total), medkits=medkits, visited=len(cells), steps=steps,
             novelty=novelty, reflexes=reflexes, strikes=strikes,
             det_recall=det_hits / max(1, det_true), det_false=det_false / max(1, steps),
             placed=len(learner.anchor_err) / max(1, steps), anchor_err=float(np.mean(learner.anchor_err)) if learner.anchor_err else -1.0, lifetime_cells=len(getattr(learner.agent, "novelty_counts", {})), seconds=steps * FRAME_SKIP / 35.0, died=died, kills=float(kills),
             spikes=learner.res.spikes / max(1, steps * learner.res.ticks))
    if shaper is not None:
        s.update({k: float(shaper.counts[k]) for k in ("kills", "items", "doors", "keys", "exits", "cells", "damage")})
        s["distance"] = float(shaper.distance)
        s["approach_reward"] = float(shaper.totals.get("approach", 0.0))
        s["closest"] = float(shaper.closest) if shaper.closest is not None else -1.0
    if fields:
        s["fields"] = fields
    if trace:
        s["v_trace"] = [round(v, 4) for v in learner.agent.log.values]
        s["d_trace"] = [round(d, 4) for d in learner.agent.log.deltas]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default="basic")
    ap.add_argument("--arm", default="td", choices=("td", "reward", "random"))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--checkpoint", type=int, default=0,
                    help="save the learner every N episodes and resume from DIR/checkpoint.pkl when restarted")
    args = ap.parse_args()
    exr.set_threads(1)
    p = dict(DEFAULTS)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in p:
            raise SystemExit(f"unknown setting {k}")
        p[k] = type(p[k])(v) if p[k] is not None else float(v)
    scale = p["reward_scale"] if p["reward_scale"] is not None else SCALES.get(args.scenario, 1.0)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "test.json")):
        print(args.out, "already done", flush=True)
        return
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"scenario": args.scenario, "arm": args.arm, "episodes": args.episodes, "seed": args.seed,
                   "reward_scale": scale, **p}, f, indent=1)
    log_path = os.path.join(args.out, "log.jsonl")
    ckpt_path = os.path.join(args.out, "checkpoint.pkl")
    start = 0
    saved = None
    if args.checkpoint > 0 and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            saved = pickle.load(f)
        start = saved["episode"]
    # A resumed run plays new games from the restart on (the game's seed moves with the episode).
    game = make_game(args.scenario, 10_000 + args.seed + 7919 * start, p["skill"], p["features"] == "oracle" or bool(p["attack_reflex"]),
                     p["until_death"])
    learner = Learner(p, game.get_available_buttons_size(), args.arm, args.seed)
    if saved is not None:
        learner.agent, learner.rng = saved["agent"], saved["rng"]
        learner.vplace = saved.get("vplace", learner.vplace)
        if saved.get("landmarks") is not None:
            learner.landmarks, learner.lives, learner.origin = saved["landmarks"]
        learner.detector = saved.get("detector", learner.detector)
        learner.games = saved.get("games", learner.games)
        with open(log_path) as f:
            kept = f.readlines()[:start]
        with open(log_path, "w") as f:
            f.writelines(kept)
        print(f"{args.out} resumed at episode {start}", flush=True)
    else:
        open(log_path, "w").close()
    shaper = drl.make_shaper(doom_rl_params(args.scenario, p["skill"], p["until_death"], p["approach"],
                                                   kill=p["kill_weight"])) if args.scenario in MAPS else None
    t0 = time.time()
    with open(log_path, "a") as log:
        for ep in range(start, args.episodes):
            if args.checkpoint > 0 and ep > start and ep % args.checkpoint == 0:
                log.flush()
                with open(ckpt_path + ".tmp", "wb") as f:
                    pickle.dump({"episode": ep, "agent": learner.agent, "rng": learner.rng, "vplace": learner.vplace,
                                 "landmarks": (learner.landmarks, learner.lives, learner.origin),
                                 "detector": learner.detector, "games": learner.games}, f)
                os.replace(ckpt_path + ".tmp", ckpt_path)
            trace = ep % p["trace_every"] == 0 or ep == args.episodes - 1
            s = episode(game, learner, scale, True, trace, shaper)
            s["episode"] = ep
            s["clamped"] = learner.agent.actor.clamped()
            s["critic_w_max"] = float(np.abs(learner.agent.critic.w).max())
            if trace:
                with open(os.path.join(args.out, f"values_{ep}.json"), "w") as f:
                    json.dump({"v": s.pop("v_trace"), "delta": s.pop("d_trace"), "return": s["return"]}, f)
            log.write(json.dumps(s) + "\n")
            if ep % 50 == 0:
                log.flush()
                print(f"{args.out} ep {ep} game {s['game_reward']:.1f} kills {s['kills']} s {s['seconds']:.0f} |d| {s['mean_abs_delta']:.3f} "
                      f"H {s['entropy']:.2f} t {time.time() - t0:.0f}s", flush=True)
    game.close()
    # Greedy test on fresh episodes, no learning.
    test_game = make_game(args.scenario, 20_000 + args.seed, p["skill"], p["features"] == "oracle" or bool(p["attack_reflex"]), p["until_death"])
    rows = [episode(test_game, learner, scale, False, shaper=shaper) for _ in range(p["test"])]
    test_game.close()
    summary = {k: float(np.mean([r[k] for r in rows]))
               for k in ("game_reward", "kills", "medkits", "visited", "steps", "seconds", "died", "spikes")}
    summary["success"] = float(np.mean([r["game_reward"] > 0.5 for r in rows]))  # my_way_home: the vest found
    for k in ("items", "doors", "exits", "cells", "distance", "approach_reward", "closest", "placed",
              "det_recall", "det_false"):
        if k in rows[0]:
            summary[k] = float(np.mean([r[k] for r in rows]))
    fields = [f for r in rows for f in r.pop("fields", [])]
    if fields:
        # How place-like the learned cells are: the spread of the positions where each cell wins
        # (weighted by its wins) over the spread of all positions; 1 = no place information.
        f = np.asarray(fields)
        spread = lambda q: float(np.sqrt(q[:, 1].var() + q[:, 2].var()))
        cells = [f[f[:, 0] == c] for c in np.unique(f[:, 0])]
        summary["field_spread"] = float(sum(len(c) * spread(c) for c in cells) / len(f) / max(spread(f), 1e-9))
        summary["cells_used"] = len(cells)
    summary["seconds_median"] = float(np.median([r["seconds"] for r in rows]))
    summary["seconds_max"] = float(np.max([r["seconds"] for r in rows]))
    np.save(os.path.join(args.out, "actor_W.npy"), learner.agent.actor.W)
    np.save(os.path.join(args.out, "critic_w.npy"), learner.agent.critic.w)
    with open(os.path.join(args.out, "test.json"), "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f)
    print(args.out, "TEST", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
