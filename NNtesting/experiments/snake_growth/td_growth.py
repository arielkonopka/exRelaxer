"""Snake with growth driven by a TD critic's error (research log §69).

The network and its learning are snake_growth's (experiment.py): state -> eye -> er (E-R, learns by
feedback alignment) -> three readouts that learn from their own error. Added here:

    critic       a linear TD(lambda) critic V(er outputs) (exrelaxer.actor_critic.Critic). It reads
                 only the E-R layer, like the readouts, so its error says how well that layer
                 represents what matters; it does not drive the actor.
    td trigger   ErrorPlateau: the critic's error (mean delta^2 / variance of the TD target) has
                 stopped falling while the critic is still learning -> grow `increment` neurons;
                 probation: growth that did not lower the error is undone.
    candidates   ShadowPool: `candidates` muted neurons learn to follow delta (Cascade-Correlation,
                 online); growth unmutes the best-correlated ones instead of adding random neurons.
                 candidate_rate=0: the pool does not learn and growth picks at random (the control).
    significance Significance: the least significant active neuron (by how much its readers and the
                 critic use it) is muted, restored if the critic's error rises; muted neurons join
                 the pool. memory: the layer's size limit with muted neurons; over it the least
                 correlated muted ones are deleted.
    saturation   SynapticScaling: a neuron whose weights press against `weight_limit` is scaled down
                 the first time and split the second time.

    python NNtesting/experiments/snake_growth/td_growth.py --start 1 --trigger td --candidates 8
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import exrelaxer as exr
from exrelaxer.actor_critic import Critic
from exrelaxer.development import ErrorPlateau, ShadowPool, Significance, SynapticScaling

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402

snake = E.snake

DEFAULTS = dict(E.DEFAULTS)
DEFAULTS.update({
    "recovery": 0.5,           # the library default (snake_growth keeps 0.9 for its old results)
    "max_depth": 1,            # width only (the user's choice, 2026-11-03)
    "trigger": "td",           # "td": ErrorPlateau; "saturation": the ActivityMonitor (snake_growth);
                               # "timer": every `timer` ticks (the control for "td");
                               # "none": no width growth (fixed, or only significance pruning)
    "timer": 8000,
    "gamma": 0.97, "lam": 0.9, "critic_rate": 0.03,
    "td_window": 2000,         # ticks per error window
    "td_patience": 3,          # windows without progress before growth
    "td_progress": 0.02,       # relative fall of the error that counts as progress
    "td_effort": 0.25,         # the critic must still move >= this x its first window's effort
    "td_probation": 3,         # windows before a growth is judged (undone if the error did not fall)
    "judge": "score",          # growth undone if, `undo` evaluations later, "score" did not rise
                               # (snake_growth's LIFO rule) or the critic's "error" did not fall
    "candidates": 0,           # shadow pool size; 0: growth adds random neurons (WidthGrowth)
    "candidate_rate": 0.02,    # 0: the pool does not learn, growth picks candidates at random
    "candidate_leak": 0.1,
    "significance": False,     # mute the least significant neuron (probation, then pool)
    "sig_after": 20000, "sig_every": 2000, "sig_fraction": 0.2, "sig_tolerance": 0.05,
    "sig_minimum": 0,          # active neurons kept; 0: the layer's minimum size
    "memory": 0,               # size limit including muted neurons; 0: none
    "saturation": False,       # synaptic scaling, then a split, on saturated weights
    "weight_limit": 10.0, "sat_share": 0.1, "sat_target": 0.5, "sat_repeat": 20000,
})


class TDAgent(E.Agent):
    def __init__(self, p):
        super().__init__(p)
        self.critic = Critic(self.net.layer_size(self.er), p["gamma"], p["lam"], p["critic_rate"])
        self.plateau = ErrorPlateau(window=p["td_window"], patience=p["td_patience"],
                                    min_progress=p["td_progress"], min_effort=p["td_effort"],
                                    probation=p["td_probation"], judge=p["judge"] == "error")
        self.pool = None
        if p["candidates"] > 0:
            self.pool = ShadowPool(size=p["candidates"], rate=p["candidate_rate"], leak=p["candidate_leak"],
                                   train=p["candidate_rate"] > 0)
            self._resize(lambda: self.pool.fill(self.net, self.er, source=self.eye))
        self.significance = Significance(after=p["sig_after"], every=p["sig_every"], fraction=p["sig_fraction"],
                                         tolerance=p["sig_tolerance"],
                                         minimum=p["sig_minimum"] or self.net.minimum_size(self.er)) \
            if p["significance"] else None
        self.saturation = SynapticScaling(limit=p["weight_limit"], share=p["sat_share"], target=p["sat_target"],
                                          repeat_window=p["sat_repeat"],
                                          max_size=None) if p["saturation"] else None
        self.reward = 0.0
        self.verdict = None
        self.growth = None          # (growth orders added, indices frozen by the growth)
        self.td_events, self.sig_events, self.deleted = [], [], 0
        self.windows_seen = 0
        self.errors = []            # critic error per window (unexplained share of the TD target)

    # --- bookkeeping: the critic's weights follow the layer's neurons ---------------------
    def _resize(self, change):
        """Runs `change` (a structural call) and realigns the critic to the layer by growth order."""
        order = [int(o) for o in self.net.growth_order(self.er)]
        before = list(zip(order, range(len(order))))
        result = change()
        after = [int(o) for o in self.net.growth_order(self.er)]
        w, e = self.critic.w, self.critic.e
        nw, ne = np.zeros(len(after)), np.zeros(len(after))
        # base neurons (order 0) keep their relative order; grown ones are matched by order
        base_old = [i for o, i in before if o == 0]
        grown_old = {o: i for o, i in before if o}
        k = 0
        for j, o in enumerate(after):
            if o == 0:
                if k < len(base_old) and base_old[k] < len(w):
                    nw[j], ne[j] = w[base_old[k]], e[base_old[k]]
                k += 1
            elif o in grown_old and grown_old[o] < len(w):
                nw[j], ne[j] = w[grown_old[o]], e[grown_old[o]]
        self.critic.w, self.critic.e = nw, ne
        if len(after) != len(order):
            self.monitor = self.new_monitor()
        return result

    def muted(self):
        return set(int(i) for i in self.net.muted_neurons(self.er))

    def active(self):
        m = self.muted()
        return [i for i in range(self.net.layer_size(self.er)) if i not in m]

    # --- playing ------------------------------------------------------------------------
    def choose(self, state):
        self.net.set_inputs("state", np.asarray(state, np.float32))
        self.net.step()
        self.monitor.observe(self.net, self.er)
        x = np.asarray(self.net.layer_output(self.er), dtype=np.float64).ravel()
        self._td(x, terminal=False)
        if self.significance is not None:
            self.significance.observe(self.net, self.er, self.readouts, extra=np.abs(self.critic.w * x))
        self.y = y = self.net.outputs()
        best = snake.STRAIGHT
        for a in (snake.LEFT, snake.RIGHT):
            if y[a] > y[best]:
                best = a
        return best

    def _td(self, x, terminal):
        w0 = self.critic.w.copy()
        previous = self.critic.previous
        delta, _ = self.critic.step(x, self.reward, terminal)
        if previous is None:
            return
        effort = float(np.abs(self.critic.w - w0).mean()) if w0.size else 0.0
        verdict = self.plateau.update(delta, delta + previous, effort, self.ticks)  # target = r + gamma V(s')
        if verdict is not None:
            self.verdict = verdict
        if self.pool is not None:
            self.pool.observe(self.net, self.er, delta)

    def end_game(self, reward):
        """The game ended with `reward` (no next state)."""
        self.reward = reward
        self._td(np.zeros(self.net.layer_size(self.er)), terminal=True)
        if self.pool is not None:
            self.pool._y_prev = None
        self.reward = 0.0

    def learn(self, action, reward):
        super().learn(action, reward)
        self.reward = reward

    # --- development ----------------------------------------------------------------------
    def develop(self):
        if not self.p["grow"]:
            return
        p, net, er = self.p, self.net, self.er
        size = net.layer_size(er)
        verdict, self.verdict = self.verdict, None
        if p["trigger"] == "td" and verdict == "grow":
            self._grow()
        elif verdict == "undo" and self.growth is not None:
            self._undo()
        elif verdict == "keep":
            self.growth = None
        if p["trigger"] == "saturation" and len(self.active()) < p["max_width"] and self.monitor.saturated:
            self._grow()
        if p["trigger"] == "timer" and self.ticks % p["timer"] == 0:
            self._grow()
        if len(self.plateau.history) > self.windows_seen:
            self.windows_seen = len(self.plateau.history)
            self.errors.append(self.plateau.last_error)
            if self.significance is not None:
                r = self.significance.judge(net, er, self.plateau.last_error, self.ticks)
                if r is not None and r[0] == "pruned" and self.pool is not None:
                    self.pool.revive(net, er, [r[1]])
        if self.significance is not None and not self.plateau.judging:
            pool = set(self.pool.indices(net, er)) if self.pool is not None else set()
            active = [i for i in self.active() if i not in pool]
            self.significance.prune(net, er, self.ticks, active, self.plateau.last_error)
        if self.saturation is not None:
            skip = self.muted()
            new = self._resize(lambda: self.saturation.check(net, er, self.ticks, skip))
            if new:
                order = net.growth_order(er)
                start = self.p["grown_resting"]
                for i in new:
                    net.set_resting_threshold(er, i, float(net.state_probe(er)["resting_threshold"][i]) * start)
        removed = self._resize(lambda: self.pruning.update(net, er, self.monitor, self.ticks))
        if removed and self.pool is not None:
            pass  # Pruning skips muted neurons, so the pool is untouched
        if p["memory"] > 0 and self.pool is not None:
            while net.layer_size(er) > p["memory"] and len(self.pool.members) > 0:
                corr = self.pool.correlations(net, er)
                if not corr:
                    break
                worst = min(corr, key=lambda i: (abs(corr[i]), -i))
                o = int(net.growth_order(er)[worst])
                self._resize(lambda: net.prune_neurons(er, [worst]))
                self.pool.forget(net, er, [o])
                self.deleted += 1
        if net.layer_size(er) != size:
            self.monitor = self.new_monitor()

    def _grow(self):
        p, net, er = self.p, self.net, self.er
        room = p["max_width"] - len(self.active())
        count = min(p["increment"], room)
        if count <= 0:
            return
        frozen_before = set(int(i) for i in net.frozen_neurons(er))
        if self.pool is not None:
            pool_idx = set(self.pool.indices(net, er))
            chosen = self.pool.promote(net, er, count, self.ticks)
            if p["freeze_old"]:
                for i in self.active():
                    if i not in chosen and i not in pool_idx:
                        net.freeze_neurons(er, i, 1)
            order = net.growth_order(er)
            added = [int(order[i]) for i in chosen]
            for i in chosen:
                if p["grown_resting"] != 1.0:
                    rest = float(net.state_probe(er)["resting_threshold"][i])
                    net.set_resting_threshold(er, i, rest * p["grown_resting"])
            self._resize(lambda: self.pool.fill(net, er))
        else:
            before = net.layer_size(er)
            self._resize(lambda: self.width.grow(net, er, self.monitor, self.ticks))
            order = net.growth_order(er)
            added = [int(order[i]) for i in range(before, net.layer_size(er))]
        newly_frozen = set(int(i) for i in net.frozen_neurons(er)) - frozen_before
        order = net.growth_order(er)
        self.growth = (added, [int(order[i]) if order[i] else ("base", i) for i in newly_frozen])
        self.grown_at = (self.evaluation, self.last_score)
        self.td_events.append((self.ticks, "grow", len(self.active())))

    def evaluated(self, score):
        """After each evaluation window: growth judged by the score (judge="score")."""
        if self.p["judge"] == "score" and self.growth is not None and self.p["undo"] > 0:
            at, before = self.grown_at
            if self.evaluation - at >= self.p["undo"]:
                if before is not None and score <= before:
                    self._undo()
                    self.plateau.backoff(kept=False)
                else:
                    self.growth = None
                    self.plateau.backoff(kept=True)
        self.evaluation += 1
        self.last_score = score

    def _undo(self):
        net, er = self.net, self.er
        added, frozen = self.growth
        self.growth = None
        order = [int(o) for o in net.growth_order(er)]
        idx = [i for i, o in enumerate(order) if o in added]
        if idx:
            self._resize(lambda: net.prune_neurons(er, idx))
        order = [int(o) for o in net.growth_order(er)]
        for key in frozen:
            if isinstance(key, tuple):
                i = key[1]
            else:
                i = order.index(key) if key in order else None
            if i is not None and i < len(order):
                net.freeze_neurons(er, i, 1, False)
        self.td_events.append((self.ticks, "undo", len(self.active())))


def play(agent, p, seed, games, learning):
    policy = snake.Rng(seed ^ 0x5EED)
    results = []
    for i in range(games):
        g = snake.Game(p["width"], p["height"], (seed * 1000003 + i) & snake.MASK)
        last = snake.MOVED
        while not g.over:
            action = agent.choose(g.state())
            if learning and p["explore"] > 0 and policy.uniform() < p["explore"]:
                action = policy.below(3)
            before = g.apple_distance()
            last = g.step(action)
            agent.ticks += 1
            if learning:
                if last == snake.ATE:
                    r = p["eat"]
                elif last == snake.DIED:
                    r = p["die"]
                else:
                    r = p["approach"] if g.apple_distance() < before else -p["approach"]
                agent.learn(action, r)
                if g.over:
                    agent.end_game(r)
                agent.develop()
        if not learning:
            agent.critic.reset()
            agent.reward = 0.0
        results.append((g.score, g.steps, last == snake.DIED))
    return np.array(results, dtype=np.float64)


def develop(p, seed, report=None):
    say = report or (lambda text: None)
    exr.reseed(seed)
    agent = TDAgent(p)
    baseline = play(agent, p, 50_000 + seed, p["eval_games"], learning=False)[:, 0].mean()
    curve, sizes = [], []
    agent.last_score = baseline
    for k in range(p["games"] // p["eval_games"]):
        games = play(agent, p, seed * 7919 + k, p["eval_games"], learning=True)
        score = games[:, 0].mean()
        curve.append(score)
        agent.evaluated(score)
        active = len(agent.active())
        sizes.append((active, agent.net.layer_size(agent.er)))
        say(f"evaluation {k}: {score:.2f} apples/game, active {active}, layer {agent.net.layer_size(agent.er)}, "
            f"error {agent.plateau.last_error if agent.plateau.last_error is not None else float('nan'):.3f}")
    test = play(agent, p, 90_000 + seed, 100, learning=False)
    pool_events = agent.pool.events if agent.pool is not None else []
    return agent, {
        "baseline": float(baseline),
        "final": float(np.mean(curve[-3:])),
        "last300": float(np.mean(curve[-6:])),
        "test": float(test[:, 0].mean()),
        "active": len(agent.active()),
        "layer": agent.net.layer_size(agent.er),
        "grows": sum(1 for e in agent.td_events if e[1] == "grow"),
        "undos": sum(1 for e in agent.td_events if e[1] == "undo"),
        "mutes": sum(1 for e in agent.significance.events if e[2] == "mute") if agent.significance else 0,
        "restores": sum(1 for e in agent.significance.events if e[2] == "restore") if agent.significance else 0,
        "scales": sum(1 for e in agent.saturation.events if e[2] == "scale") if agent.saturation else 0,
        "splits": sum(1 for e in agent.saturation.events if e[2] == "split") if agent.saturation else 0,
        "deleted": agent.deleted,
        "promoted_corr": [round(c, 3) for e in pool_events for c in e[2]],
        "ticks": agent.ticks,
        "curve": [round(float(c), 2) for c in curve],
        "sizes": sizes,
        "errors": [round(float(e), 4) for e in agent.errors],
        "events": agent.td_events,
    }


def parse_value(v, like):
    if isinstance(like, bool):
        return v.lower() in ("1", "true", "yes")
    return type(like)(v)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="append one JSON line per seed")
    ap.add_argument("--arm", default="", help="a name stored with each line")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--set", action="append", default=[], help="key=value")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    exr.set_threads(1)
    p = dict(DEFAULTS)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in p:
            raise SystemExit(f"unknown setting {k}")
        p[k] = parse_value(v, p[k])
    lo, hi = (int(s) for s in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
    for seed in range(lo, hi + 1):
        t0 = time.time()
        _, m = develop(p, seed, report=None if args.quiet else print)
        row = {"arm": args.arm, "seed": seed, "settings": {k: p[k] for k in p if p[k] != DEFAULTS[k]},
               "seconds": round(time.time() - t0, 1), **m}
        print(args.arm, seed, {k: m[k] for k in ("final", "test", "active", "layer", "grows", "undos", "mutes", "splits")},
              flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
