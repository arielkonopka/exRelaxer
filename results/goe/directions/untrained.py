"""G5: untrained networks (net seeds 0-2) and a random player, 4 against 8 directions, on 30 fresh mazes.

    python3 results/goe/directions/untrained.py results/goe/directions/untrained.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                "NNtesting", "experiments", "goe_rl"))
import exrelaxer as exr  # noqa: E402
import experiment as e  # noqa: E402

exr.set_threads(1)
seeds = [4242 + i for i in range(30)]
out = {}
game = None
for d in (4, 8):
    for name, cfg in (("er_rec", {"recurrent": True}), ("er", {})):
        p = dict({k: v[0] for k, v in e.PARAMS.items()}, radius=6, directions=d, **cfg)
        game = game or e.make_game(p)
        rows = []
        for net_seed in range(3):
            exr.reseed(net_seed)
            pl = e.Player(p)
            r = e.play(game, pl, p, seeds, meter=True)
            r["spikes"] = pl.spikes / max(pl.ticks_seen, 1) * p["ticks"]
            rows.append(r)
        out[f"{name}/{d}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | \
                             {"reward_by_seed": [r["reward"] for r in rows]}
        print(name, d, {k: round(v, 2) for k, v in out[f"{name}/{d}"].items() if not isinstance(v, list)}, flush=True)
    p = dict({k: v[0] for k, v in e.PARAMS.items()}, radius=6, directions=d)
    rows = [e.play(game, e.RandomPlayer(s, p), p, seeds) for s in range(3)]
    out[f"random/{d}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("random", d, {k: round(v, 2) for k, v in out[f"random/{d}"].items()}, flush=True)
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
