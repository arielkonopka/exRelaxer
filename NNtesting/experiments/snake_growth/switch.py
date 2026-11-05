"""Does the plasticity potential protect an old skill? (research log §72)

Task A is Snake; task B is the same Snake with the 23 sensors in a fixed random order (a
"permuted Snake", as permuted MNIST in continual learning). Learning never stops:

    A  `games_a` games, learning      -> score on A without learning (before)
    B  `games_b` games, learning      -> score on A without learning (after), and on B
    A  `games_back` games, learning   -> how fast A comes back

retention = after / before. Arms as plasticity.py (no depth growth: max_depth 1).

    python NNtesting/experiments/snake_growth/switch.py OUT_DIR [--seeds 20]
    python NNtesting/experiments/snake_growth/switch.py OUT_DIR --summary
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment as sg
from plasticity import ARMS, paired

TEST_GAMES = 100


class SwitchAgent(sg.Agent):
    perm = None

    def choose(self, state):
        state = np.asarray(state, np.float32)
        return super().choose(state if self.perm is None else state[self.perm])


def window(agent, p, seed, games, scores):
    """Plays `games` learning games in evaluation windows; appends each window's mean."""
    for k in range(games // p["eval_games"]):
        g = sg.play(agent, p, seed + k, p["eval_games"], learning=True)
        scores.append(float(g[:, 0].mean()))
        agent.evaluated(scores[-1])


def test(agent, p, seed, perm):
    agent.perm = perm
    return float(sg.play(agent, p, seed, TEST_GAMES, learning=False)[:, 0].mean())


def job(args):
    out, name, seed, games_a, games_b, games_back = args
    p = dict(sg.DEFAULTS, **ARMS[name], max_depth=1)
    import exrelaxer as exr
    exr.reseed(seed)
    agent = SwitchAgent(p)
    perm = np.random.default_rng(1000 + seed).permutation(sg.snake.STATE_SIZE)
    curve_a, curve_b, curve_back = [], [], []
    agent.last_score = 0.0
    window(agent, p, seed * 7919, games_a, curve_a)
    before = test(agent, p, 50_000 + seed, None)
    size_a = agent.sizes()
    P_a = agent.net.plasticity_potential(agent.newest).tolist() if p["plasticity"] else []
    agent.perm = perm
    window(agent, p, seed * 7919 + 100_000, games_b, curve_b)
    after = test(agent, p, 50_000 + seed, None)
    on_b = test(agent, p, 60_000 + seed, perm)
    P_b = agent.net.plasticity_potential(agent.newest).tolist() if p["plasticity"] else []
    agent.perm = None
    window(agent, p, seed * 7919 + 200_000, games_back, curve_back)
    m = dict(arm=name, seed=seed, before=before, after=after, on_b=on_b,
             b_final=float(np.mean(curve_b[-3:])), back_first=curve_back[0],
             back_final=float(np.mean(curve_back[-3:])), size_a=size_a, size=agent.sizes(),
             P_a=P_a, P_b=P_b, curve_a=curve_a, curve_b=curve_b, curve_back=curve_back)
    (Path(out) / f"{name}_s{seed}.json").write_text(json.dumps(m))
    return name, seed, before, after


def summary(out, against="fixed16"):
    runs = {}
    for f in sorted(Path(out).glob("*.json")):
        m = json.loads(f.read_text())
        runs.setdefault(m["arm"], {})[m["seed"]] = m
    base = runs.get(against, {})
    print(f"| arm | seeds | A before | A after B | retention | B (learned) | A back, first window | "
          f"A back, final | Δ after vs {against} | mean P after A | mean P after B | neurons |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ARMS:
        ms = runs.get(name)
        if not ms:
            continue
        v = list(ms.values())
        mean = lambda k: np.mean([m[k] for m in v])
        ret = np.mean([m["after"] / max(m["before"], 1e-9) for m in v])
        d = paired(ms, base, lambda m: m["after"]) if name != against and base else None
        dcol = f"{d[0]:+.2f} ± {d[1]:.2f}" if d else ""
        pa = np.mean([np.mean(m["P_a"]) for m in v]) if v[0]["P_a"] else float("nan")
        pb = np.mean([np.mean(m["P_b"]) for m in v]) if v[0]["P_b"] else float("nan")
        print(f"| {name} | {len(v)} | {mean('before'):.2f} | {mean('after'):.2f} | {ret:.2f} | {mean('on_b'):.2f} | "
              f"{mean('back_first'):.2f} | {mean('back_final'):.2f} | {dcol} | {pa:.2f} | {pb:.2f} | "
              f"{np.mean([sum(m['size']) for m in v]):.1f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--games-a", type=int, default=3000)
    parser.add_argument("--games-b", type=int, default=1500)
    parser.add_argument("--games-back", type=int, default=500)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--against", default="fixed16")
    args = parser.parse_args()
    if args.summary:
        summary(args.out, args.against)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        todo = [(args.out, n, s, args.games_a, args.games_b, args.games_back)
                for s in range(1, args.seeds + 1) for n in args.arms if not (Path(args.out) / f"{n}_s{s}.json").exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            for name, seed, before, after in pool.map(job, todo):
                print(f"{name} s{seed}: A {before:.2f} -> {after:.2f}", flush=True)
