#!/usr/bin/env python3
"""Doom agents that learn while they play, and the common test and profile of any doom_rl agent:
the building blocks of the comparison pipeline (NNtesting/pipeline, doc/pipeline.md).

    online.py train   --out DIR --config JSON --method fa_td [--episodes 600] [--net-seed 1]
    online.py frozen  --out DIR --config JSON [--net-seed 1]           # the untrained network, as an agent
    online.py test    --agent DIR --out test.json [--episodes 30] [--workers 4]
    online.py profile --agent DIR --out profile.json [--steps 2000] [--method fa_td]

An agent is a folder with agent.json ({"kind", "params", "net_seed", ...}) and its weights: es.py's
best_theta.npy (kind "es"), online training's final.exr (kind "online"), or none (kind "frozen").
Every agent of one net_seed starts from the same random network (es.build), whatever its neuron
model or learning method, so a comparison starts from copies of one architecture.

Online methods learn on every game step (learning always on, no separate phase); the E-R state
and the rules' traces start fresh in each episode, the weights carry over. Signals:
- reward: the step's shaped reward minus its running mean (as doom_rl's own play);
- td: the TD error of a critic (Network.set_critic, TD(lambda)) one step later,
  delta = r + gamma V(next) - V(now). The critic reads the hidden layer when it does not learn,
  else the eye (on Gardens of Eris a critic reading a learning layer chased it into NaN).
A reward-driven rule (sign, trace, eligibility, eprop, perturbation) gives the signal to the chosen
action's readout (apply_reward_to) and, with hidden learning, to the hidden layers; an
error-driven rule (feedback alignment, surrogate gradients = truncated backpropagation through
time) gets an error per action: the signal for the chosen action, 0 for the others (apply_error).

Training is resumable: every `--checkpoint` episodes the network (full state, with the critic)
and the progress go to DIR; a restarted run continues from there, on the same game seeds.
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

# method -> (learning rule, signal, default rate). Rates are first guesses, not tuned for Doom
# (GoE G4: feedback alignment best near 1e-5; reward rules near 1e-3).
METHODS = {
    "sign": ("sign", "reward", 0.01),
    "trace": ("trace", "reward", 0.01),
    "eligibility_td": ("eligibility", "td", 1e-3),
    "eprop_td": ("eprop", "td", 1e-3),
    "perturbation_td": ("perturbation", "td", 1e-3),
    "fa_td": ("fa", "td", 1e-4),
    "surrogate_td": ("surrogate", "td", 1e-4),
}
ERROR_RULES = {"fa", "surrogate"}
SETTINGS = {
    "explore": 0.05,      # epsilon-greedy while training
    "reward_scale": 1.0,  # shaped reward x this is the signal
    "clip": 5.0,          # the signal is clipped to +-clip
    "baseline": 0.01,     # reward signal: rate of the running mean subtracted
    "gamma": 0.95,        # critic, per game step
    "lambda": 0.8,
    "critic_rate": 0.05,
    "hidden": False,      # the hidden layers learn too (default: readouts only)
}
TRAIN_SEED = 1000000   # training episode k plays game seed TRAIN_SEED + 1000 * seed + k
TEST_SEED = 4242       # test episode k plays game seed TEST_SEED + k (as doom.md's fresh games)


def rule(name):
    import exrelaxer as exr
    LR = exr.LearningRule
    return {"sign": LR.sign(), "trace": LR.traced(0.8, 0.0), "eligibility": LR.eligibility(0.9, 0.0),
            "eprop": LR.eprop(0.8, 0.5, 0.0), "perturbation": LR.perturbation(0.1, 0.8, 0.0),
            "fa": LR.feedback_alignment(0.8), "surrogate": LR.surrogate(8, 0.5)}[name]


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


# --- agents ---------------------------------------------------------------------------------------
def params(config, scenario):
    import es
    return es.params(config, scenario)


def load_agent(folder, fresh=True):
    """The agent's Player and its agent.json. fresh: the network's state at rest (a new game)."""
    import es
    import exrelaxer as exr
    with open(os.path.join(folder, "agent.json")) as f:
        info = json.load(f)
    p, net_seed = info["params"], info["net_seed"]
    if info["kind"] == "es":
        return es.build(p, net_seed, np.load(os.path.join(folder, info.get("theta", "best_theta.npy")))), info
    player = es.build(p, net_seed, None)
    if info["kind"] == "online":
        mode = exr.DeserializeMode.WeightsOnly if fresh else exr.DeserializeMode.FullState
        player.net = exr.Network.load(os.path.join(folder, "final.exr"), mode)
        # Played, not learning: no rule may act on the forward pass (node perturbation adds noise to
        # every sum). A Learner (profile) sets the method's rules again.
        for layer in player.layers + getattr(player, "rungs", []) + player.readouts:
            player.net.set_learning_rule(layer, exr.LearningRule.sign())
    return player, info


# --- the learner ----------------------------------------------------------------------------------
class Learner:
    """Turns a Player into an online learner by one method."""

    def __init__(self, player, method, rate=None, s=None):
        self.player, self.method = player, method
        self.s = dict(SETTINGS, **(s or {}))
        self.rule, self.signal, default = METHODS[method]
        self.rate = default if rate is None else rate
        n = player.net
        hidden = player.layers + getattr(player, "rungs", []) if self.s["hidden"] else []
        self.hidden = hidden
        for layer in hidden + player.readouts:
            n.unfreeze(layer)
            n.set_learning_rule(layer, rule(self.rule))
        if self.signal == "td":
            if self.s["hidden"]:
                n.set_critic(layers=[], inputs=["eye"], gamma=self.s["gamma"], lambda_=self.s["lambda"],
                             rate=self.s["critic_rate"])
            else:
                n.set_critic(layers=list(player.layers), inputs=[], gamma=self.s["gamma"],
                             lambda_=self.s["lambda"], rate=self.s["critic_rate"])
        self.baseline = 0.0
        self.updates = 0
        self.timing = None  # profile: seconds in learning calls

    def new_episode(self):
        n = self.player.net
        for layer in self.hidden + self.player.readouts:
            n.reset_traces(layer)
        if self.signal == "td":
            n.reset_critic()
        self.pending = None
        self.first = True

    def _learn(self, signal, action):
        n = self.player.net
        signal = float(np.clip(signal, -self.s["clip"], self.s["clip"]))
        if signal == 0.0 or not np.isfinite(signal):
            return
        if self.rule in ERROR_RULES:
            errors = np.zeros(len(self.player.readouts), dtype=np.float32)
            errors[action] = signal
            n.apply_error(errors, self.rate)
        else:
            for h in self.hidden:
                n.apply_reward_to(h, signal, self.rate)
            n.apply_reward_to(self.player.readouts[action], signal, self.rate)
        self.updates += 1

    def after_act(self):
        """After the network saw the new view (Player.act): the TD step for the previous move."""
        if self.signal != "td":
            return
        t0 = time.perf_counter()
        n = self.player.net
        if self.first:
            n.temporal_difference(0.0, False)  # the critic's first state
            self.first = False
        elif self.pending is not None:
            r, a = self.pending
            self._learn(n.temporal_difference(r, False), a)
            self.pending = None
        self._time(t0)

    def after_move(self, action, reward, terminal):
        t0 = time.perf_counter()
        r = reward * self.s["reward_scale"]
        if self.signal == "reward":
            self._learn(r - self.baseline, action)
            self.baseline += self.s["baseline"] * (r - self.baseline)
        elif terminal:
            self._learn(self.player.net.temporal_difference(r, True), action)
        else:
            self.pending = (r, action)
        self._time(t0)

    def _time(self, t0):
        if self.timing is not None:
            self.timing["learning"] = self.timing.get("learning", 0.0) + time.perf_counter() - t0


# --- one episode ----------------------------------------------------------------------------------
def episode(game, player, p, rng, explore=0.0, learner=None, timing=None, max_steps=0):
    """One episode (or max_steps of it); returns its stats. timing: a dict that collects seconds by part."""
    import experiment as e
    shaper = e.make_shaper(p)
    buttons = np.eye(len(e.ACTIONS), dtype=int).tolist()
    player.timing = timing
    if learner is not None:
        learner.timing = timing
        learner.new_episode()
    clock = time.perf_counter
    t = clock()
    game.new_episode()
    state = game.get_state()
    shaper.reset(state)
    if timing is not None:
        timing["game"] = timing.get("game", 0.0) + clock() - t
    total, steps = 0.0, 0
    spikes0, ticks0 = player.spikes, player.ticks_seen
    while not game.is_episode_finished():
        a = player.act(state, rng, explore, meter=True)
        if learner is not None:
            learner.after_act()
        t = clock()
        game.make_action(buttons[a], e.FRAME_SKIP)
        done = game.is_episode_finished()
        state = None if done else game.get_state()
        t1 = clock()
        r = shaper.step(state) if state is not None else shaper.end(game)
        if timing is not None:
            timing["game"] = timing.get("game", 0.0) + t1 - t
            timing["reward"] = timing.get("reward", 0.0) + clock() - t1
        stalled = state is not None and 0 < p.get("stall_steps", 0) <= shaper.since_new
        if learner is not None:
            learner.after_move(a, r, done or stalled)
        total += r
        steps += 1
        if stalled or (max_steps and steps >= max_steps):
            break
    ticks = max(player.ticks_seen - ticks0, 1)
    return {"reward": total, "steps": steps, "kills": shaper.counts["kills"], "deaths": shaper.counts["deaths"],
            "damage": shaper.counts["damage"], "items": shaper.counts["items"], "distance": shaper.distance,
            "spikes_per_step": (player.spikes - spikes0) / ticks * p["ticks"],
            "active_fraction": (player.spikes - spikes0) / ticks / max(player.neurons, 1)}


# --- commands -------------------------------------------------------------------------------------
def cmd_frozen(args):
    os.makedirs(args.out, exist_ok=True)
    p = params(json.loads(args.config), args.scenario)
    write_json(os.path.join(args.out, "agent.json"), {"kind": "frozen", "params": p, "net_seed": args.net_seed})


def cmd_train(args):
    import es
    import experiment as e
    import exrelaxer as exr
    exr.set_threads(1)
    os.makedirs(args.out, exist_ok=True)
    p = params(json.loads(args.config), args.scenario)
    settings = json.loads(args.settings)
    rate = args.rate if args.rate > 0 else None
    progress_path = os.path.join(args.out, "progress.json")
    checkpoint = os.path.join(args.out, "checkpoint.exr")
    player = es.build(p, args.net_seed, None)
    learner = Learner(player, args.method, rate, settings)
    done, rng_state = 0, None
    if os.path.exists(progress_path) and os.path.exists(checkpoint):
        with open(progress_path) as f:
            progress = json.load(f)
        done, rng_state, learner.baseline = progress["episodes"], progress["rng"], progress["baseline"]
        player.net = exr.Network.load(checkpoint, exr.DeserializeMode.FullState)
        print(f"resuming {args.method} at episode {done}", flush=True)
    log = os.path.join(args.out, "log.jsonl")
    if os.path.exists(log):  # episodes after the checkpoint are played again: drop their lines
        with open(log) as f:
            kept = [x for x in f if x.strip() and json.loads(x)["episode"] <= done]
        with open(log, "w") as f:
            f.writelines(kept)
    rng = np.random.default_rng(args.seed)
    if rng_state is not None:
        rng.bit_generator.state = rng_state
    info = {"kind": "online", "params": p, "net_seed": args.net_seed, "method": args.method,
            "rate": learner.rate, "settings": learner.s, "episodes": args.episodes}
    game = e.make_game(p, 1)
    try:
        while done < args.episodes:
            t = time.time()
            game.set_seed(TRAIN_SEED + 1000 * args.seed + done)
            stats = episode(game, player, p, rng, learner.s["explore"], learner)
            done += 1
            stats.update(episode=done, seconds=time.time() - t, updates=learner.updates)
            with open(log, "a") as f:
                f.write(json.dumps(stats) + "\n")
            print(f"episode {done}: reward {stats['reward']:.2f} kills {stats['kills']} "
                  f"steps {stats['steps']} {stats['seconds']:.1f} s", flush=True)
            if done % args.checkpoint == 0 or done == args.episodes:
                player.net.save(checkpoint + ".tmp")
                os.replace(checkpoint + ".tmp", checkpoint)
                write_json(progress_path, {"episodes": done, "rng": rng.bit_generator.state,
                                           "baseline": learner.baseline})
    finally:
        game.close()
    player.net.save(os.path.join(args.out, "final.exr"))
    write_json(os.path.join(args.out, "agent.json"), info)


_worker = {}


def _init(agent):
    import experiment as e
    import exrelaxer as exr
    exr.set_threads(1)
    with open(os.path.join(agent, "agent.json")) as f:
        p = json.load(f)["params"]
    _worker.update(agent=agent, p=p, game=e.make_game(p, 1))


def _test(k):
    player, _ = load_agent(_worker["agent"])  # a fresh network every episode
    game = _worker["game"]
    game.set_seed(TEST_SEED + k)
    return episode(game, player, _worker["p"], np.random.default_rng(k))


def cmd_test(args):
    t = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.agent,)) as pool:
            games = pool.map(_test, range(args.episodes))
    else:
        _init(args.agent)
        games = [_test(k) for k in range(args.episodes)]
        _worker["game"].close()
    keys = games[0].keys()
    out = {"episodes": args.episodes, "seconds": time.time() - t,
           "mean": {k: float(np.mean([g[k] for g in games])) for k in keys},
           "se": {k: float(np.std([g[k] for g in games], ddof=1) / np.sqrt(len(games))) if len(games) > 1 else 0.0
                  for k in keys},
           "games": games}
    write_json(args.out, out)
    print(f"test: reward {out['mean']['reward']:.2f} +- {out['se']['reward']:.2f}, kills {out['mean']['kills']:.1f}")


def synapses(player):
    """Weights of the hidden layers and readouts (the cochlea has none)."""
    layers = player.layers + getattr(player, "rungs", []) + player.readouts + \
        ([player.reservoir] if player.reservoir is not None else [])
    return int(sum(len(player.net.weights(l, i)) for l in layers for i in range(player.net.layer_size(l))))


def cmd_profile(args):
    """Where a game step's time goes for this agent: the game engine, the reward, the senses, the
    network's ticks and (with --method) online learning; plus the network's size."""
    import experiment as e
    import exrelaxer as exr
    exr.set_threads(1)
    player, info = load_agent(args.agent)
    p = info["params"]
    method = args.method or (info.get("method") if info["kind"] == "online" else "")
    learner = Learner(player, method, info.get("rate"), info.get("settings")) if method else None
    game = e.make_game(p, 1)
    game.set_seed(TEST_SEED)
    timing, steps, eps = {}, 0, 0
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    while steps < args.steps:
        stats = episode(game, player, p, rng, 0.0, learner, timing, args.steps - steps)
        steps += stats["steps"]
        eps += 1
    total = time.perf_counter() - t
    game.close()
    timing["other"] = max(0.0, total - sum(timing.values()))
    ticks = steps * p["ticks"]
    out = {"steps": steps, "episodes": eps, "seconds": total, "ms_per_step": 1000 * total / steps,
           "ms_per_step_by_part": {k: 1000 * v / steps for k, v in timing.items()},
           "share": {k: v / total for k, v in timing.items()},
           "us_per_tick": 1e6 * timing.get("network", 0.0) / ticks,
           "steps_per_second": steps / total, "ticks_per_step": p["ticks"],
           "neurons": int(player.neurons), "synapses": synapses(player), "inputs": int(player.eye_size),
           "method": method or None, "model": p["model"]}
    write_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("ms_per_step", "ms_per_step_by_part", "us_per_tick", "synapses")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "frozen"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--config", default="{}", help="doom_rl parameters as JSON (over es.py's defaults)")
        s.add_argument("--scenario", default="defend_the_center")
        s.add_argument("--net-seed", type=int, default=0)
        if name == "train":
            s.add_argument("--method", required=True, choices=sorted(METHODS))
            s.add_argument("--rate", type=float, default=0.0, help="learning rate (0: the method's default)")
            s.add_argument("--settings", default="{}", help="SETTINGS overrides as JSON")
            s.add_argument("--episodes", type=int, default=600)
            s.add_argument("--checkpoint", type=int, default=10, help="episodes between checkpoints")
            s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("test")
    s.add_argument("--agent", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--episodes", type=int, default=30)
    s.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("profile")
    s.add_argument("--agent", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--method", default="", help="profile with this online method learning (default: the agent's)")
    args = ap.parse_args()
    {"train": cmd_train, "frozen": cmd_frozen, "test": cmd_test, "profile": cmd_profile}[args.cmd](args)


if __name__ == "__main__":
    main()
