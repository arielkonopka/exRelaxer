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
}
SCALES = {"basic": 0.01, "defend_the_center": 1.0, "map01": 0.2}


MAPS = ("map01",)  # whole levels: doom_rl's 8 buttons and shaped reward (doom_rl/rewards.py)


def doom_rl_params(scenario, skill, until_death):
    """doom_rl's parameters for a whole map, without sound."""
    q = {k: v[0] for k, v in drl.PARAMS.items()}
    q.update(scenario=scenario, sound=False, color=False)
    if skill is not None:
        q["skill"] = int(skill)
    if until_death > 0:
        q["episode_tics"] = int(until_death * 60 * 35)
    return q


def make_game(scenario, seed, skill=None, labels=False, until_death=0):
    if scenario in MAPS:
        return drl.make_game(doom_rl_params(scenario, skill, until_death), seed)
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_sound_enabled(False)
    game.set_labels_buffer_enabled(labels)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
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


def pool_screen(screen, pool):
    return screen.reshape(120 // pool, pool, 160 // pool, pool).mean(axis=(1, 3)).ravel().astype(np.float32) / 255.0


class Learner:
    def __init__(self, p, buttons, arm, seed):
        exr.reseed(seed)
        self.eye_size = (120 // p["pool"]) * (160 // p["pool"])
        kind = {"both_eye": "both", "eye": "none", "oracle": "none", "both_pool": "both", "pool": "none"}.get(
            p["features"], p["features"])
        self.oracle = p["features"] == "oracle"
        self.res = Reservoir({"eye": self.eye_size}, width=p["width"], recurrent_scale=p["recurrent_scale"],
                             recovery=p["recovery"], spread=p["spread"], model=p["model"], features=kind, ticks=p["ticks"],
                             vision=dict(height=120 // p["pool"], width=160 // p["pool"]) if p["vision"] == "gabor" else None,
                             with_pool=p["features"] in ("both_pool", "pool"),
                             extra=ORACLE_BINS if self.oracle else self.eye_size if p["features"] in ("both_eye", "eye") else 0)
        self.arm, self.p, self.buttons = arm, p, buttons
        self.rng = np.random.default_rng(seed)
        self.agent = Agent(self.res.size, buttons, p["gamma"], p["lam"], p["critic_rate"], p["actor_rate"],
                           clip=p["clip"], modulator="reward" if arm == "reward" else "td", post=p["post"],
                           tau=p["tau"], entropy=p["entropy"], center=p["center"], rng=self.rng)

    def features(self, state):
        if self.oracle:
            return self.res.step({"eye": np.zeros(self.eye_size)}, raw=oracle_features(state))
        eye = pool_screen(state.screen_buffer, self.p["pool"])
        return self.res.step({"eye": eye}, raw=eye)

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
    state = game.get_state()
    if shaper is not None:
        shaper.reset(state)
    x = learner.features(state)
    a = learner.act(x, 0.0, False, learn)
    steps = 0
    while True:
        r = game.make_action(eye[a], FRAME_SKIP) * scale
        steps += 1
        done = game.is_episode_finished()
        if shaper is not None:
            if done:
                r = shaper.end(game) * scale
            else:
                state = game.get_state()
                r = shaper.step(state) * scale
        if done:
            if game.get_episode_time() >= game.get_episode_timeout() and not game.is_player_dead():
                # A time limit is not a terminal state: the last state seen bootstraps
                # (V(s_T) as if play went on), else running out the clock would
                # look like relief from the living cost (delta ~ -V > 0).
                learner.agent.step(x, r, False, learn=learn and learner.arm != "random", explore=False)
                learner.agent.end_episode()
            else:
                learner.act(np.zeros(learner.res.size), r, True, learn)
            break
        if shaper is None:
            state = game.get_state()
        x = learner.features(state)
        a = learner.act(x, r, False, learn)
    total = game.get_total_reward() if shaper is None else sum(learner.agent.log.rewards) / scale
    kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)  # still readable after the end
    died = int(game.is_player_dead())
    s = learner.agent.log.summary()
    s.update(game_reward=float(total), steps=steps, seconds=steps * FRAME_SKIP / 35.0, died=died, kills=float(kills),
             spikes=learner.res.spikes / max(1, steps * learner.res.ticks))
    if shaper is not None:
        s.update({k: float(shaper.counts[k]) for k in ("kills", "items", "doors", "keys", "exits", "cells", "damage")})
        s["distance"] = float(shaper.distance)
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
    game = make_game(args.scenario, 10_000 + args.seed + 7919 * start, p["skill"], p["features"] == "oracle",
                     p["until_death"])
    learner = Learner(p, game.get_available_buttons_size(), args.arm, args.seed)
    if saved is not None:
        learner.agent, learner.rng = saved["agent"], saved["rng"]
        with open(log_path) as f:
            kept = f.readlines()[:start]
        with open(log_path, "w") as f:
            f.writelines(kept)
        print(f"{args.out} resumed at episode {start}", flush=True)
    else:
        open(log_path, "w").close()
    shaper = drl.make_shaper(doom_rl_params(args.scenario, p["skill"], p["until_death"])) if args.scenario in MAPS else None
    t0 = time.time()
    with open(log_path, "a") as log:
        for ep in range(start, args.episodes):
            if args.checkpoint > 0 and ep > start and ep % args.checkpoint == 0:
                log.flush()
                with open(ckpt_path + ".tmp", "wb") as f:
                    pickle.dump({"episode": ep, "agent": learner.agent, "rng": learner.rng}, f)
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
    test_game = make_game(args.scenario, 20_000 + args.seed, p["skill"], p["features"] == "oracle", p["until_death"])
    rows = [episode(test_game, learner, scale, False, shaper=shaper) for _ in range(p["test"])]
    test_game.close()
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("game_reward", "kills", "steps", "seconds", "died", "spikes")}
    for k in ("items", "doors", "exits", "cells", "distance"):
        if k in rows[0]:
            summary[k] = float(np.mean([r[k] for r in rows]))
    summary["seconds_median"] = float(np.median([r["seconds"] for r in rows]))
    summary["seconds_max"] = float(np.max([r["seconds"] for r in rows]))
    np.save(os.path.join(args.out, "actor_W.npy"), learner.agent.actor.W)
    np.save(os.path.join(args.out, "critic_w.npy"), learner.agent.critic.w)
    with open(os.path.join(args.out, "test.json"), "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f)
    print(args.out, "TEST", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
