#!/usr/bin/env python3
"""Teaches the column network of columns.py skill by skill, and grows it when it has learned.

    python3 NNtesting/experiments/goe_rl/grow.py --out results/goe-grow/run1 --workers 4 --generations 600
    python3 NNtesting/experiments/goe_rl/grow.py --out ... --config '{"curriculum": "maze", "max_columns": 1}'
    python3 NNtesting/experiments/goe_rl/grow.py --check      # growing keeps play exact; rooms build

Evolution is OpenAI-ES as in es.py (antithetic pairs, centred ranks, Adam, steps
relative to each group's weight RMS), on what the network trains now: the readouts,
the newest column's kernels and its E-R layer. Older columns are frozen. With
`fitness` per_world (the default) candidates are ranked in each world and the
ranks summed, so a maze world (rewards in the tens, a death -50) does not drown
the rooms (a few points each).

Stages. `curriculum` lists skills (columns.SKILLS: explore, collect, doors, avatar,
mines, maze). Stage s trains on the rooms of the first s + 1 skills, in equal
shares, plus a `maze_share` of random mazes (every candidate plays the same
worlds). The last stage keeps going, and can still grow, up to max_columns.

When to move on. Every `check_every` generations the current weights play a fixed
test set (test_per_kind worlds of each kind in the stage's mix, never trained on),
paired with the network the stage started from, on the same worlds:
- competent: the paired gain's bootstrap lower bound (5th percentile of the mean)
  is above 0, and the mean gain is at least `min_gain`, on `confirm` checks in a row;
- plateau: the test mean has not beaten its best of the stage for `patience` checks.
When both hold (and the stage has run `min_stage` generations), the next stage
starts with a new column (grow_columns true; false: the same network, as a
control). Its readout lines start at zero, so it starts playing as the old one did,
and the next stage's reference is that play on its own test set. A stage that never
becomes competent never moves on: the network grows only when it has learned.

Files in --out: config.json, log.jsonl (a line per generation, plus "check" and
"grow" lines), state.npz and frozen.npz (resume), best_stage<s>.exr / best.exr, and
before_stage<s>/ (the run just before stage s began: a no-growth control resumes from it).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TEST_SEED = 900000
_worker = {}

TRAIN = {
    "curriculum": ("explore,collect,doors,avatar,mines,maze", "skills in teaching order (columns.SKILLS)"),
    "maze_share": (0.25, "share of random-maze worlds in every stage's training mix"),
    "grow_columns": (True, "a new column with every new stage (false: one network, the control)"),
    "fitness": ("per_world", "per_world: candidates ranked in each world, the ranks summed (every world counts "
                             "the same); sum: ranked by the summed reward"),
    "max_columns": (8, "at most this many columns"),
    "check_every": (10, "generations between test checks"),
    "test_per_kind": (6, "fixed test worlds per kind in the stage's mix"),
    "min_stage": (30, "generations before a stage may end"),
    "patience": (3, "checks without a new best test mean: a plateau"),
    "confirm": (2, "competent checks in a row before growing"),
    "min_gain": (0.5, "smallest mean paired gain over the stage's start that counts"),
}


def params(config):
    import columns as C
    p = C.defaults() | {k: v[0] for k, v in TRAIN.items()}
    unknown = set(config) - set(p) - {"description"}
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    p.update(config)
    p.pop("description", None)
    skills = p["curriculum"].split(",")
    bad = [s for s in skills if s not in C.SKILLS]
    if bad or not skills:
        raise ValueError(f"curriculum: unknown skills {bad}; skills are {C.SKILLS}")
    return p


def stage_kinds(p, stage):
    skills = p["curriculum"].split(",")
    kinds = skills[:stage + 1]
    if "maze" not in kinds and p["maze_share"] > 0:
        kinds = kinds + ["maze"]
    return kinds


def test_worlds(p, stage):
    return [(kind, TEST_SEED + 1000 * i + j) for i, kind in enumerate(stage_kinds(p, stage))
            for j in range(p["test_per_kind"])]


def train_worlds(p, stage, rng, episodes):
    kinds = stage_kinds(p, stage)
    rooms = [k for k in kinds if k != "maze"]
    out = []
    for _ in range(episodes):
        maze = "maze" in kinds and (not rooms or rng.random() < p["maze_share"])
        out.append(("maze" if maze else str(rng.choice(rooms)), int(rng.integers(1, 1 << 30))))
    return out


# --- workers ------------------------------------------------------------------------------------
def init_worker(p, columns, frozen, net_seed):
    import exrelaxer as exr
    import columns as C
    exr.set_threads(1)
    _worker.update(p=p, columns=columns, frozen=frozen, net_seed=net_seed, game=C.make_game(p))


def play(task):
    """(theta or None for the random player, worlds) -> (per-world rewards, events, spikes per step by part)."""
    import columns as C
    theta, worlds = task
    w = _worker
    if theta is None:
        rewards, events = C.play(w["game"], C.RandomPlayer(), w["p"], worlds)
        return rewards, events, {}
    # a fresh network (fresh E-R and habituation state) for every world, so a world's score does not
    # depend on which worlds the same process played before it
    rewards, events, spikes, ticks = [], {}, {}, 0
    for world in worlds:
        net = C.Net(w["p"], w["columns"], w["frozen"], theta, w["net_seed"])
        r, e = C.play(w["game"], net, w["p"], [world], meter=True)
        rewards.append(r[0])
        for k, x in e.items():
            events[k] = events.get(k, 0.0) + x
        for k, x in net.spikes.items():
            spikes[k] = spikes.get(k, 0) + x
        ticks += net.ticks_seen
    return np.array(rewards), events, {k: x / max(ticks, 1) * w["p"]["ticks"] for k, x in spikes.items()}


def centred_ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def bootstrap_low(d, rng, n=2000, q=5):
    """The q-th percentile of the bootstrapped mean of d."""
    if len(d) < 2:
        return -np.inf
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.percentile(means, q))


# --- the check ----------------------------------------------------------------------------------
def check(args):
    """Growing keeps play exact, frozen columns stay frozen, every room builds and plays."""
    import exrelaxer as exr
    import columns as C
    p = params(json.loads(args.config) if args.config.strip().startswith("{") else {})
    exr.set_threads(1)
    game = C.make_game(p)
    worlds = [(k, 4242 + i) for i, k in enumerate(C.SKILLS)]
    net1 = C.Net(p, 1, net_seed=0)
    theta1, _ = net1.get_trainable()
    theta1 = theta1 + np.random.default_rng(5).standard_normal(len(theta1)) * 0.3 * np.abs(theta1).mean()
    trained = C.Net(p, 1, None, theta1, 0)
    frozen = trained.frozen()
    grown = C.Net(p, 2, frozen, None, 0)
    theta2 = carry_readouts(trained, grown)
    grown.set_trainable(theta2)
    for name, vectors in frozen.items():
        layer = {f"c{k + 1}": c for k, c in enumerate(grown.convs)} | {f"h{k + 1}": h for k, h in enumerate(grown.hidden)}
        for a, b in zip(vectors, grown._vectors(layer[name])):
            assert np.array_equal(a, b), f"{name} changed when growing"
    a_rewards, a_events = C.play(game, C.Net(p, 1, None, theta1, 0), p, worlds)
    b_rewards, b_events = C.play(game, C.Net(p, 2, frozen, theta2, 0), p, worlds)
    print("one column :", np.round(a_rewards, 2), {k: v for k, v in a_events.items() if v})
    print("grown to 2 :", np.round(b_rewards, 2), {k: v for k, v in b_events.items() if v})
    assert np.allclose(a_rewards, b_rewards) and a_events == b_events, "the grown network plays differently"
    # the check can see a change: the same grown network with non-zero new readout lines
    theta3 = theta2.copy()
    n_read = len(C.ACTIONS) * 2 * p["width"]
    theta3[:n_read] += np.random.default_rng(1).standard_normal(n_read) * 0.5
    c_rewards, _ = C.play(game, C.Net(p, 2, frozen, theta3, 0), p, worlds)
    print("perturbed  :", np.round(c_rewards, 2))
    print("check passed: growing keeps play exact; every room plays")


def carry_readouts(old, new):
    """The new network's trainable vector: the old readout lines kept, the new column's lines at
    zero, the new column's own weights as the new network made them."""
    theta_new, _ = new.get_trainable()
    width, columns = new.p["width"], old.columns
    for i, r in enumerate(old.readouts):  # the readouts come first, one vector of width x columns each
        at = i * width * new.columns
        theta_new[at:at + width * columns] = old._vectors(r)[0]
        theta_new[at + width * columns:at + width * new.columns] = 0.0
    return theta_new


# --- evolution ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--config", default="{}", help="columns.py and curriculum parameters, as JSON or a file of it")
    ap.add_argument("--net-seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--episodes", type=int, default=4, help="worlds per candidate per generation")
    ap.add_argument("--generations", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="check growing and the rooms, then exit")
    args = ap.parse_args()
    if args.check:
        return check(args)
    if not args.out:
        ap.error("--out is required")
    import columns as C
    if os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = json.loads(args.config)
    p = params(config)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    state_path, frozen_path = os.path.join(args.out, "state.npz"), os.path.join(args.out, "frozen.npz")

    def log(line):
        with open(log_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "params": p}, f, indent=1)

    last_stage = len(p["curriculum"].split(",")) - 1
    frozen = {}
    if os.path.exists(state_path):
        s = np.load(state_path, allow_pickle=True)
        st = json.loads(str(s["meta"]))
        theta, m, v, scale = s["theta"], s["m"], s["v"], s["scale"]
        if os.path.exists(frozen_path):
            fz = np.load(frozen_path)
            for key in fz.files:
                name, i = key.rsplit("_", 1)
                frozen.setdefault(name, {})[int(i)] = fz[key]
            frozen = {k: [d[i] for i in sorted(d)] for k, d in frozen.items()}
        print(f"resuming at generation {st['gen']}, stage {st['stage']}, {st['columns']} columns", flush=True)
    else:
        st = {"gen": 0, "stage": 0, "columns": 1, "stage_start": 0, "t": 0, "best_test": -1e18,
              "since_best": 0, "competent_run": 0, "ref": None, "random_ref": None, "best_ever": -1e18}
        theta, scale = C.Net(p, 1, None, None, args.net_seed).get_trainable()
        m, v = np.zeros_like(theta), np.zeros_like(theta)
    rng = np.random.default_rng(args.seed + st["gen"])
    boot = np.random.default_rng(args.seed + 7)

    def save():
        np.savez(state_path, theta=theta, m=m, v=v, scale=scale, meta=json.dumps(st))
        np.savez(frozen_path, **{f"{k}_{i}": w for k, vs in frozen.items() for i, w in enumerate(vs)})

    while st["gen"] < args.generations:
        sigma, lr = args.sigma * scale, args.lr * scale
        tests = test_worlds(p, st["stage"])
        with mp.get_context("spawn").Pool(args.workers, initializer=init_worker,
                                          initargs=(p, st["columns"], frozen, args.net_seed)) as pool:

            def run_test(th):
                chunks = [tests[i::args.workers] for i in range(args.workers)]
                out = pool.map(play, [(th, c) for c in chunks if c])
                rewards = np.empty(len(tests))
                for i, (r, _, _) in enumerate(out):
                    rewards[i::args.workers] = r
                events = {}
                for _, e, _ in out:
                    for k, x in e.items():
                        events[k] = events.get(k, 0.0) + x / len(tests)
                return rewards, events

            if st["ref"] is None:  # the stage's reference: how it plays as it starts, and a random player
                ref, ref_events = run_test(theta)
                rnd, rnd_events = run_test(None)
                st["ref"], st["random_ref"] = ref.tolist(), rnd.tolist()
                st["best_test"] = float(ref.mean())
                log({"kind": "stage", "generation": st["gen"], "stage": st["stage"], "columns": st["columns"],
                     "mix": stage_kinds(p, st["stage"]), "start_test": float(ref.mean()),
                     "random_test": float(rnd.mean()), "start_events": ref_events, "random_events": rnd_events})
                print(f"stage {st['stage']} ({', '.join(stage_kinds(p, st['stage']))}), {st['columns']} columns: "
                      f"test {ref.mean():.2f} at the start, random player {rnd.mean():.2f}", flush=True)
                save()
            grew = False
            while st["gen"] < args.generations and not grew:
                start = time.time()
                eps = rng.standard_normal((args.pairs, len(theta)))
                worlds = train_worlds(p, st["stage"], rng, args.episodes)
                tasks = [(theta + sigma * e, worlds) for e in eps] + [(theta - sigma * e, worlds) for e in eps]
                results = pool.map(play, tasks)
                per_world = np.array([r for r, _, _ in results])  # candidates x worlds
                rewards = per_world.sum(axis=1)
                if p["fitness"] == "per_world":  # each world counts the same: a maze death does not drown a room
                    ranks = centred_ranks(sum(centred_ranks(per_world[:, j]) for j in range(per_world.shape[1])))
                else:
                    ranks = centred_ranks(rewards)
                grad = (ranks[:args.pairs] - ranks[args.pairs:]) @ eps / (2 * args.pairs * sigma)
                st["gen"] += 1
                st["t"] += 1
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad ** 2
                theta = theta + lr * (m / (1 - 0.9 ** st["t"])) / (np.sqrt(v / (1 - 0.999 ** st["t"])) + 1e-8)
                spikes = {k: float(np.mean([s[k] for _, _, s in results])) for k in results[0][2]}
                log({"generation": st["gen"], "stage": st["stage"], "columns": st["columns"],
                     "population_mean": float(rewards.mean()) / len(worlds),
                     "population_best": float(rewards.max()) / len(worlds),
                     "worlds": [k for k, _ in worlds], "spikes_per_step": spikes, "seconds": time.time() - start})
                print(f"gen {st['gen']} (stage {st['stage']}, {st['columns']} col): population "
                      f"{rewards.mean() / len(worlds):.2f} best {rewards.max() / len(worlds):.2f}, "
                      f"spikes {', '.join(f'{k} {x:.0f}' for k, x in spikes.items())}, {time.time() - start:.0f} s",
                      flush=True)
                if st["gen"] % p["check_every"] == 0:
                    test, events = run_test(theta)
                    gain = test - np.array(st["ref"])
                    low = bootstrap_low(gain, boot)
                    competent = bool(low > 0 and gain.mean() >= p["min_gain"])
                    st["competent_run"] = st["competent_run"] + 1 if competent else 0
                    if test.mean() > st["best_test"]:
                        st["best_test"], st["since_best"] = float(test.mean()), 0
                        best = C.Net(p, st["columns"], frozen, theta, args.net_seed)
                        best.net.save(os.path.join(args.out, f"best_stage{st['stage']}.exr"))
                        if test.mean() > st["best_ever"] and st["stage"] == last_stage:
                            st["best_ever"] = float(test.mean())
                            best.net.save(os.path.join(args.out, "best.exr"))
                            np.save(os.path.join(args.out, "best_theta.npy"), theta)
                    else:
                        st["since_best"] += 1
                    plateau = st["since_best"] >= p["patience"]
                    by_kind = {}
                    for (kind, _), g in zip(tests, gain):
                        by_kind.setdefault(kind, []).append(g)
                    log({"kind": "check", "generation": st["gen"], "stage": st["stage"], "columns": st["columns"],
                         "test": float(test.mean()), "gain": float(gain.mean()), "gain_low": low,
                         "gain_by_kind": {k: float(np.mean(x)) for k, x in by_kind.items()},
                         "competent": competent, "competent_run": st["competent_run"], "plateau": plateau,
                         "events": events})
                    print(f"  check: test {test.mean():.2f} (start {np.mean(st['ref']):.2f}, random "
                          f"{np.mean(st['random_ref']):.2f}), gain {gain.mean():+.2f} (5% {low:+.2f}), "
                          f"{'competent' if competent else 'not yet'}, {'plateau' if plateau else 'improving'}",
                          flush=True)
                    ready = (st["competent_run"] >= p["confirm"] and plateau and st["t"] >= p["min_stage"])
                    more_stages = st["stage"] < last_stage
                    can_grow = p["grow_columns"] and st["columns"] < p["max_columns"]
                    if ready and (more_stages or can_grow):
                        # the run as it stands before moving on: a control can resume from it
                        # (copy both files into a new --out as state.npz and frozen.npz)
                        save()
                        snap = os.path.join(args.out, f"before_stage{st['stage'] + 1}")
                        os.makedirs(snap, exist_ok=True)
                        for path in (state_path, frozen_path):
                            with open(path, "rb") as src, open(os.path.join(snap, os.path.basename(path)), "wb") as dst:
                                dst.write(src.read())
                        if can_grow:
                            old = C.Net(p, st["columns"], frozen, theta, args.net_seed)
                            frozen = old.frozen()
                            new = C.Net(p, st["columns"] + 1, frozen, None, args.net_seed)
                            theta = carry_readouts(old, new)
                            scale = new.get_trainable()[1]  # the new column's own scale; readouts by their RMS
                            st["columns"] += 1
                        st["stage"] = min(st["stage"] + 1, last_stage)
                        m, v = np.zeros_like(theta), np.zeros_like(theta)
                        st.update(t=0, ref=None, random_ref=None, since_best=0, competent_run=0, best_test=-1e18)
                        log({"kind": "grow", "generation": st["gen"], "stage": st["stage"], "columns": st["columns"]})
                        print(f"gen {st['gen']}: stage {st['stage']}, {st['columns']} columns", flush=True)
                        grew = True
                save()


if __name__ == "__main__":
    main()
