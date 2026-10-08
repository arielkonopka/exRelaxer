#!/usr/bin/env python3
"""The grown Gardens of Eris network (grow.py, exported by agent.py) trained further by different
learning techniques, every one from the same start, on the same stream of worlds, and tested on
the same fresh worlds as G3 (results/goe/columns/fresh.py).

    python3 NNtesting/experiments/goe_rl/online.py sweep --agent AGENT --out DIR      # rates on validation worlds
    python3 NNtesting/experiments/goe_rl/online.py final --agent AGENT --out DIR      # best rates, 3 seeds, test
    python3 NNtesting/experiments/goe_rl/online.py es --agent AGENT --out DIR         # continued ES, the baseline
    python3 NNtesting/experiments/goe_rl/online.py one --agent AGENT --technique eligibility --rate 1e-3

Online techniques learn on every game step while playing (learning always on, no separate phase):
the network keeps its weights from world to world, its E-R state starts fresh in each world.
What learns: the readouts and the newest column's E-R layer (h<k>), as when it grew; the older
columns and every convolution stay frozen. ES (the way the network was made) evolves the
readouts, the newest column's kernels and its E-R layer, as grow.py did.

Learning signals (one per game step):
- reward: the game's reward of the step (x reward_scale), right after the move, so the network's
  activity is still the step's own;
- td: the TD error of a critic (Network.set_critic over the columns that do not learn and the body,
  TD(lambda))
  after the next view: delta = r + gamma V(next) - V(now). The rules' traces carry the step's
  activity to that moment.
A reward-driven rule (sign, trace, eligibility, eprop, perturbation) gives the signal to the
chosen action's readout only (apply_reward_to) and to the hidden layer. An error-driven rule
(feedback alignment, eprop_error, surrogate) gets an error per action: the signal for the chosen
action, 0 for the others (Network.apply_error: the readouts their own error, h<k> a random
projection or, for surrogate, the gradient through the readouts and time).

Exploration while training: epsilon-greedy (epsilon 0.05). Tests: greedy, every world from a
fresh network with the learned weights, learning off ("frozen test"), and learning on inside
each test world ("online test").
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

# the G3 fresh worlds (results/goe/columns/fresh.py): 30 mazes and 10 rooms of each skill, and more of
# both: a maze score is chaotic (0.3 % noise on the weights moves a 20-maze mean from 30 to 59), so
# 200 mazes (standard error about 4) and 20 rooms of each skill
SKILLS = ["explore", "collect", "doors", "avatar", "mines", "maze"]
G3 = [("maze", 4242 + i) for i in range(30)] + \
     [(k, 4300 + 100 * j + i) for j, k in enumerate(SKILLS[:-1]) for i in range(10)]
TEST = G3 + [("maze", 5000 + i) for i in range(170)] + \
       [(k, 4310 + 100 * j + i) for j, k in enumerate(SKILLS[:-1]) for i in range(10)]
# validation worlds for choosing rates: never trained or tested on
VALID = [("maze", 610000 + i) for i in range(40)] + [(k, 620000 + 100 * j + i) for j, k in enumerate(SKILLS[:-1])
                                                    for i in range(10)]
TRAIN_SEED = 7000000  # training worlds: drawn from rng(TRAIN_SEED + seed), seeds in [1, 2^30)

# technique -> (rule for the readouts and h<k>, signal, curiosity)
TECHNIQUES = {
    "frozen": (None, None, False),
    "jitter": (None, None, False),  # control: no learning, random noise of `rate` x each group's RMS
    "sign": ("sign", "reward", False),
    "sign_td": ("sign", "td", False),
    "trace": ("trace", "td", False),
    "eligibility": ("eligibility", "td", False),
    "eligibility_raw": ("eligibility", "reward", False),
    "eprop": ("eprop", "td", False),
    "perturbation": ("perturbation", "td", False),
    "fa": ("fa", "td", False),
    "eprop_error": ("eprop_error", "td", False),
    "surrogate": ("surrogate", "td", False),
    "eligibility_curious": ("eligibility", "td", True),
}
ERROR_RULES = {"fa", "eprop_error", "surrogate"}
# "<technique>_ro": only the readouts learn (the newest column stays frozen, so the critic's features
# and the readouts' inputs keep their scale)
SETTINGS = {
    "epsilon": 0.05,        # exploration while training
    "reward_scale": 0.1,    # the game's reward x this is the learning signal (a death: -5)
    "clip": 5.0,            # the signal is clipped to +-clip
    "gamma": 0.95,          # critic, per game step
    "lambda": 0.8,
    "critic_rate": 0.05,
    "curiosity_scale": 1.0,  # intrinsic reward (x reward_scale) per step: the forward model's error
    "train_worlds": 300,    # worlds in a training stream
    "maze_share": 0.25,     # of the stream: random mazes; the rest the five skill rooms in equal shares
}


def rule(name):
    import exrelaxer as exr
    LR = exr.LearningRule
    return {"sign": LR.sign(), "trace": LR.traced(0.8, 0.0), "eligibility": LR.eligibility(0.9, 0.0),
            "eprop": LR.eprop(0.8, 0.5, 0.0), "perturbation": LR.perturbation(0.1, 0.8, 0.0),
            "fa": LR.feedback_alignment(0.8), "eprop_error": LR.eprop(0.8, 0.5, 0.0),
            "surrogate": LR.surrogate(8, 0.5)}[name]


def train_stream(seed, n, maze_share):
    rng = np.random.default_rng(TRAIN_SEED + seed)
    rooms = SKILLS[:-1]
    return [("maze" if rng.random() < maze_share else str(rng.choice(rooms)), int(rng.integers(1, 1 << 30)))
            for _ in range(n)]


# --- the learner ---------------------------------------------------------------------------------
class Learner:
    """A columns.Net that learns while it plays, by one technique."""

    def __init__(self, net, technique, rate, s=SETTINGS, seed=0):
        self.net, self.technique, self.rate, self.s = net, technique, rate, s
        readouts_only = technique.endswith("_ro")
        self.rule, self.signal, self.curious = TECHNIQUES[technique.removesuffix("_ro")]
        self.rng = np.random.default_rng(10007 + seed)
        n = net.net
        self.hidden = None if readouts_only else net.hidden[-1]
        self.learning = ([] if readouts_only else [net.hidden[-1]]) + net.readouts if self.rule else []
        for layer in self.learning:
            n.unfreeze(layer)
            n.set_learning_rule(layer, rule(self.rule))
        if self.signal == "td":
            # the critic reads the columns that do not learn: one that reads a learning column chases its
            # features and diverged (TD errors oscillating and growing to NaN within ~100 worlds)
            fixed = [h for h in net.hidden if h not in self.learning]
            n.set_critic(layers=fixed, inputs=["body"], gamma=s["gamma"], lambda_=s["lambda"],
                         rate=s["critic_rate"])
        if self.curious:
            n.set_curiosity(predict_inputs=["eye"], from_layers=list(net.hidden), from_inputs=["body"])
        self.updates = 0

    def new_world(self):
        n = self.net.net
        for layer in self.learning:
            n.reset_traces(layer)
        if self.signal == "td":
            n.reset_critic()
        if self.curious:
            n.reset_curiosity()

    def _learn(self, signal, action):
        n, rate = self.net.net, self.rate
        signal = float(np.clip(signal, -self.s["clip"], self.s["clip"]))
        if signal == 0.0:
            return
        if self.rule in ERROR_RULES:
            errors = np.zeros(len(self.net.readouts), dtype=np.float32)
            errors[action] = signal
            n.apply_error(errors, rate)
        else:
            if self.hidden is not None:
                n.apply_reward_to(self.hidden, signal, rate)
            n.apply_reward_to(self.net.readouts[action], signal, rate)
        self.updates += 1

    def play(self, game, p, world, learn=True, epsilon=0.0):
        """One world, learning on every step when `learn`; (reward, events, steps)."""
        import columns as C
        net, n = self.net, self.net.net
        index = [C.goe.ACTIONS.index(a) for a in C.ACTIONS]
        kind, seed = world
        C.prepare(game, kind, seed)
        net.reset(seed)
        self.new_world()
        limit = p["maze_moves"] if kind == "maze" else p["room_moves"]
        total, steps, pending = 0.0, 0, None  # pending: (reward, action) waiting for the next view (td)
        while not game.is_episode_finished() and steps < limit:
            a = net.act(game.get_state())
            curiosity = 0.0
            if self.curious:  # once per game step, after its ticks: how surprising this view was
                curiosity = n.curiosity_reward() * self.s["curiosity_scale"]
            if learn and self.signal == "td" and steps == 0:
                n.temporal_difference(0.0, False)  # the critic's first state
            if learn and pending is not None:
                r, prev = pending
                delta = n.temporal_difference(r + curiosity, False)
                self._learn(delta, prev)
            if epsilon > 0 and self.rng.random() < epsilon:
                a = int(self.rng.integers(len(C.ACTIONS)))
            r = game.make_action(index[a])
            total += r
            steps += 1
            if learn and self.rule:
                if self.signal == "reward":
                    self._learn(r * self.s["reward_scale"], a)
                else:
                    pending = (r * self.s["reward_scale"], a)
        if learn and pending is not None:  # the world's end: nothing after it
            delta = n.temporal_difference(pending[0], True)
            self._learn(delta, pending[1])
        return total, dict(game.episode_events), steps


def weights_of(net):
    """Every column's weights and the readouts, as agent.py saves them."""
    w = {f"{name}_{i}": v for name, vectors in net.frozen().items() for i, v in enumerate(vectors)}
    w["readouts"] = np.stack([net._vectors(r)[0] for r in net.readouts])
    return w


def build(agent_dir, weights=None):
    """A fresh columns.Net: the agent's own weights, or `weights` (weights_of)."""
    import agent as A
    net = A.load(agent_dir)
    if weights is not None:
        frozen = A._by_layer(_Npz(weights))
        net.load(frozen)
        for r, row in zip(net.readouts, weights["readouts"]):
            net._set_vectors(r, [row])
    return net


class _Npz(dict):
    @property
    def files(self):
        return list(self.keys())


# --- jobs (one process each) ---------------------------------------------------------------------
def _setup():
    import exrelaxer as exr
    exr.set_threads(1)


_GAME = []


def _game(p):
    """The process's one game (the goe package allows one per process)."""
    import columns as C
    if not _GAME:
        _GAME.append(C.make_game(p))
    return _GAME[0]


def _params(agent_dir):
    import columns as C
    with open(os.path.join(agent_dir, "agent.json")) as f:
        return C.defaults() | json.load(f)["params"]


def evaluate(agent_dir, weights, worlds, technique=None, rate=0.0, online=False):
    """Every world from a fresh network with `weights`; with `online` the technique keeps learning
    inside each world (it starts again from `weights` in the next)."""
    import columns as C
    p = _params(agent_dir)
    game = _game(p)
    rewards, events = [], []
    for i, world in enumerate(worlds):
        net = build(agent_dir, weights)
        if online:
            learner = Learner(net, technique, rate, seed=1000 + i)
            r, ev, _ = learner.play(game, p, world, learn=True, epsilon=0.0)
        else:
            r, ev = C.play(game, net, p, [world])
            r = float(r[0])
        rewards.append(float(r))
        events.append(ev)
    return rewards, events


def job_train(task):
    """(agent, technique, rate, seed, settings, evaluation) -> training log and test results."""
    _setup()
    import columns as C
    agent_dir, technique, rate, seed, s, tests = task
    p = _params(agent_dir)
    game = _game(p)
    net = build(agent_dir)
    learner = Learner(net, technique, rate, s, seed)
    start, curve, steps = time.time(), [], 0
    for world in train_stream(seed, s["train_worlds"], s["maze_share"]):
        r, ev, n = learner.play(game, p, world, learn=True, epsilon=s["epsilon"])
        steps += n
        curve.append({"kind": world[0], "reward": r, "death": ev.get("death", 0.0)})
    w = weights_of(net)
    if technique == "jitter":  # the noise floor: the start's readouts and newest column, randomly moved
        rng = np.random.default_rng(TRAIN_SEED + 99 + seed)
        newest = f"h{net.columns}_"
        w = {k: v + rng.standard_normal(v.shape) * rate * np.sqrt(np.mean(v ** 2))
             if k == "readouts" or k.startswith(newest) else v for k, v in w.items()}
    start_w = weights_of(build(agent_dir))
    drift = {k: float(np.linalg.norm(w[k] - start_w[k]) / (np.linalg.norm(start_w[k]) + 1e-12))
             for k in ("readouts", f"h{net.columns}_0")}
    out = {"technique": technique, "rate": rate, "seed": seed, "train_steps": steps, "updates": learner.updates,
           "train_seconds": time.time() - start, "curve": curve, "drift": drift}
    for name, worlds, online in tests:
        rewards, events = evaluate(agent_dir, w, worlds, technique, rate, online)
        out[name] = {"rewards": rewards, "events": events, "worlds": worlds}
    return out, w


def summary(rewards, worlds):
    by = {}
    for (kind, _), r in zip(worlds, rewards):
        by.setdefault("maze" if kind == "maze" else "rooms", []).append(r)
        by.setdefault(kind, []).append(r)
    return {k: float(np.mean(v)) for k, v in by.items()}


def _pool(workers):
    return mp.get_context("spawn").Pool(workers)


def _write(path, line):
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")


def _brief(out):
    keys = [k for k in ("valid", "test", "test_online") if k in out]
    parts = []
    for k in keys:
        s = summary(out[k]["rewards"], out[k]["worlds"])
        parts.append(f"{k} maze {s.get('maze', np.nan):.1f} rooms {s.get('rooms', np.nan):.2f}")
    return f"{out['technique']:20s} rate {out['rate']:<8g} seed {out['seed']}: " + ", ".join(parts) + \
        f" (drift readouts {out['drift']['readouts']:.3f}, {out['train_seconds']:.0f} s)"


def strip(out):
    """A log line: per-world rewards and mean events, without the world lists."""
    line = {k: v for k, v in out.items() if not isinstance(v, dict) or k == "drift"}
    for k in ("valid", "test", "test_online"):
        if k in out:
            ev = out[k]["events"]
            line[k] = {"summary": summary(out[k]["rewards"], out[k]["worlds"]), "rewards": out[k]["rewards"],
                       "events": {e: float(np.mean([x.get(e, 0.0) for x in ev])) for e in ev[0]}}
    return line


def cmd_sweep(args):
    techniques = args.techniques.split(",")
    rates = [float(r) for r in args.rates.split(",")]
    tasks = [(args.agent, t, r, 0, SETTINGS, [("valid", VALID, False)]) for t in techniques for r in rates
             if t not in ("frozen", "jitter")]
    if "frozen" in techniques:
        tasks.insert(0, (args.agent, "frozen", 0.0, 0, SETTINGS | {"train_worlds": 0}, [("valid", VALID, False)]))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "sweep.jsonl")
    if os.path.exists(path):  # resume: skip what is done
        with open(path) as f:
            done = {(l["technique"], l["rate"]) for l in map(json.loads, f)}
        tasks = [t for t in tasks if (t[1], t[2]) not in done]
    with _pool(args.workers) as pool:
        for out, _ in pool.imap_unordered(job_train, tasks):
            _write(path, strip(out))
            print(_brief(out), flush=True)


def best_rates(sweep_path):
    """technique -> the rate with the best mean validation reward (mazes and rooms weighed by
    their share of the test: each world counts the same)."""
    best = {}
    with open(sweep_path) as f:
        for line in map(json.loads, f):
            score = float(np.mean(line["valid"]["rewards"]))
            t = line["technique"]
            if t not in best or score > best[t][1]:
                best[t] = (line["rate"], score)
    return {t: r for t, (r, _) in best.items()}


def cmd_final(args):
    rates = best_rates(args.sweep) | {"jitter": args.jitter}
    if args.techniques:
        rates = {t: rates.get(t, 0.0) for t in args.techniques.split(",")}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "final.jsonl")
    tests = [("test", TEST, False)] + ([("test_online", TEST, True)] if args.online else [])
    tasks = []
    for t, r in rates.items():
        if t == "jitter":
            tasks += [(args.agent, t, args.jitter, seed, SETTINGS | {"train_worlds": 0}, [("test", TEST, False)])
                      for seed in range(1, args.seeds + 1)]
        elif t == "frozen":
            tasks.append((args.agent, t, 0.0, 0, SETTINGS | {"train_worlds": 0}, [("test", TEST, False)]))
        else:
            tasks += [(args.agent, t, r, seed, SETTINGS, tests) for seed in range(1, args.seeds + 1)]
    if os.path.exists(path):  # resume: skip what is done
        with open(path) as f:
            done = {(l["technique"], l["seed"]) for l in map(json.loads, f) if "seed" in l}
        tasks = [t for t in tasks if (t[1], t[3]) not in done]
    with _pool(args.workers) as pool:
        for out, w in pool.imap_unordered(job_train, tasks):
            _write(path, strip(out))
            np.savez(os.path.join(args.out, f"{out['technique']}_s{out['seed']}.npz"), **w)
            print(_brief(out), flush=True)


def cmd_one(args):
    out, _ = job_train((args.agent, args.technique, args.rate, args.seed, SETTINGS | {"train_worlds": args.worlds},
                        [("valid", VALID[::max(1, len(VALID) // args.valid)], False)]))
    print(_brief(out))
    print("training curve:", [round(c["reward"], 1) for c in out["curve"]])


# --- continued ES (grow.py's evolution on the same training mix) ---------------------------------
_es = {}


def _es_init(agent_dir):
    _setup()
    import columns as C
    import agent as A
    p = _params(agent_dir)
    net = A.load(agent_dir)
    _es.update(p=p, game=_game(p), agent=agent_dir, frozen=net.frozen(), columns=net.columns,
               net_seed=json.load(open(os.path.join(agent_dir, "agent.json")))["net_seed"])


def _es_play(task):
    import columns as C
    theta, worlds = task
    rewards = []
    for world in worlds:
        net = C.Net(_es["p"], _es["columns"], _es["frozen"], theta, _es["net_seed"])
        r, _ = C.play(_es["game"], net, _es["p"], [world])
        rewards.append(float(r[0]))
    return np.array(rewards)


def cmd_es(args):
    import grow as G
    _setup()
    net = build(args.agent)
    theta, scale = net.get_trainable()
    start_theta = theta.copy()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    rng = np.random.default_rng(TRAIN_SEED + args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "es.jsonl")
    steps = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_es_init, initargs=(args.agent,)) as pool:
        for gen in range(1, args.generations + 1):
            t0 = time.time()
            sigma, lr = args.sigma * scale, args.lr * scale
            eps = rng.standard_normal((args.pairs, len(theta)))
            worlds = [("maze" if rng.random() < SETTINGS["maze_share"] else str(rng.choice(SKILLS[:-1])),
                       int(rng.integers(1, 1 << 30))) for _ in range(args.episodes)]
            per_world = np.array(pool.map(_es_play, [(theta + sigma * e, worlds) for e in eps] +
                                          [(theta - sigma * e, worlds) for e in eps]))
            steps += sum((750 if k == "maze" else 150) for k, _ in worlds) * 2 * args.pairs  # at most
            ranks = G.centred_ranks(sum(G.centred_ranks(per_world[:, j]) for j in range(per_world.shape[1])))
            grad = (ranks[:args.pairs] - ranks[args.pairs:]) @ eps / (2 * args.pairs * sigma)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            theta = theta + lr * (m / (1 - 0.9 ** gen)) / (np.sqrt(v / (1 - 0.999 ** gen)) + 1e-8)
            line = {"generation": gen, "population_mean": float(per_world.mean()), "steps_at_most": steps,
                    "seconds": time.time() - t0}
            if gen % args.test_every == 0 or gen == args.generations:
                net.set_trainable(theta)
                w = weights_of(net)
                chunks = [VALID[i::args.workers] for i in range(args.workers)]
                res = pool.map(_es_eval, [(w, c) for c in chunks])
                rewards = np.empty(len(VALID))
                for i, r in enumerate(res):
                    rewards[i::args.workers] = r
                line["valid"] = summary(rewards.tolist(), VALID) | {"mean": float(rewards.mean())}
                np.save(os.path.join(args.out, f"theta_g{gen}.npy"), theta)
            _write(path, line)
            print(json.dumps(line), flush=True)
    net.set_trainable(theta)
    w = weights_of(net)
    np.savez(os.path.join(args.out, "es_final.npz"), **w)
    drift = float(np.linalg.norm(theta - start_theta) / np.linalg.norm(start_theta))
    print("drift", drift)


def _es_eval(task):
    w, worlds = task
    return np.array(evaluate(_es["agent"], w, worlds)[0])


def cmd_test(args):
    """The final test of saved weights (an ES run's es_final.npz): frozen."""
    w = dict(np.load(args.weights))
    chunks = [TEST[i::args.workers] for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers, initializer=_es_init, initargs=(args.agent,)) as pool:
        res = pool.map(_es_eval, [(w, c) for c in chunks])
    rewards = np.empty(len(TEST))
    for i, r in enumerate(res):
        rewards[i::args.workers] = r
    line = {"technique": args.name, "weights": args.weights, "test": {"summary": summary(rewards.tolist(), TEST),
                                                                       "rewards": rewards.tolist()}}
    _write(os.path.join(args.out, "final.jsonl"), line)
    print(json.dumps(line["test"]["summary"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sweep", "final", "es", "one", "test"])
    ap.add_argument("--agent", required=True, help="an agent folder (agent.py export)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--techniques", default=",".join(TECHNIQUES))
    ap.add_argument("--rates", default="1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--sweep", help="final: the sweep.jsonl whose best rates to use")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jitter", type=float, default=0.01, help="final: the jitter control's relative noise")
    ap.add_argument("--online", action="store_true", help="final: also test with learning on in each world")
    ap.add_argument("--technique", default="eligibility")
    ap.add_argument("--rate", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--valid", type=int, default=4)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--test-every", type=int, default=10)
    ap.add_argument("--weights", help="test: an .npz of weights_of()")
    ap.add_argument("--name", default="es")
    args = ap.parse_args()
    {"sweep": cmd_sweep, "final": cmd_final, "es": cmd_es, "one": cmd_one, "test": cmd_test}[args.command](args)


if __name__ == "__main__":
    main()
