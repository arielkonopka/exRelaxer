#!/usr/bin/env python3
"""G3 fresh-world test: exported agents (goe_rl/agent.py) and a random player on worlds no run
trained or tested on: 30 mazes (seeds 4242-4271, as G2) and 10 rooms of each skill.

    python3 results/goe/columns/fresh.py OUT.json AGENT_DIR [AGENT_DIR ...]

Also the untrained one-column network both runs start from. Per agent and kind: mean reward (the run's weights), mean G2 reward (explore and avatar
weighed 0, as G2 had them) and each event's mean count. Every world starts a fresh network.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "NNtesting", "experiments", "goe_rl"))

import agent as A  # noqa: E402
import columns as C  # noqa: E402

WORLDS = [("maze", 4242 + i) for i in range(30)] + \
         [(k, 4300 + 100 * j + i) for j, k in enumerate(C.SKILLS[:-1]) for i in range(10)]


def main():
    import exrelaxer as exr
    exr.set_threads(1)
    out, agents = sys.argv[1], sys.argv[2:]
    p = C.defaults() | json.load(open(os.path.join(agents[0], "agent.json")))["params"]
    game = C.make_game(p)
    players = {os.path.basename(os.path.normpath(a)): (lambda a=a: A.load(a)) for a in agents}
    players["untrained"] = lambda: C.Net(p, 1, net_seed=0)  # where both runs started
    players["random"] = C.RandomPlayer
    g2 = {k: p["w_" + k] for k in C.EVENTS} | {"explore": 0.0, "avatar": 0.0}
    result = {}
    for name, make in players.items():
        per = {}
        for kind, seed in WORLDS:
            r, ev = C.play(game, make(), p, [(kind, seed)])
            d = per.setdefault(kind, {"reward": [], "g2_reward": [], "events": []})
            d["reward"].append(float(r[0]))
            d["g2_reward"].append(float(sum(g2[k] * v for k, v in ev.items())))
            d["events"].append(ev)
        result[name] = {kind: {"worlds": len(d["reward"]), "reward": float(np.mean(d["reward"])),
                               "reward_sd": float(np.std(d["reward"])), "g2_reward": float(np.mean(d["g2_reward"])),
                               "rewards": d["reward"],
                               "events": {k: float(np.mean([e[k] for e in d["events"]])) for k in C.EVENTS}}
                        for kind, d in per.items()}
        print(name, {k: round(v["reward"], 2) for k, v in result[name].items()}, flush=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
