"""G5: the best and final weights of the 4- and 8-direction es.py runs (er_rec) on the 30 fresh mazes.

    python3 results/goe/directions/evolved.py RUNS_DIR results/goe/directions/evolved.json

RUNS_DIR holds er_rec_d4 and er_rec_d8 (es.py --out folders).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                "NNtesting", "experiments", "goe_rl"))
import numpy as np  # noqa: E402
import exrelaxer as exr  # noqa: E402
import es  # noqa: E402
import experiment as e  # noqa: E402

exr.set_threads(1)
runs, out_path = sys.argv[1], sys.argv[2]
seeds = [4242 + i for i in range(30)]
out = {}
game = None
for d in (4, 8):
    run = os.path.join(runs, f"er_rec_d{d}")
    with open(os.path.join(run, "config.json")) as f:
        p = json.load(f)["params"]
    game = game or e.make_game(p)
    best = np.load(os.path.join(run, "best_theta.npy"))
    final = np.load(os.path.join(run, "state.npz"))["theta"]
    log = [json.loads(line) for line in open(os.path.join(run, "log.jsonl"))]
    # consistency: the best weights on the validation worlds give the logged best
    v = e.play(game, es.build(p, 0, best), p, [es.VALIDATION_SEED + i for i in range(6)])["reward"]
    print(d, "validation check", round(v, 2), "logged best", round(max(l["validation_reward"] for l in log), 2),
          flush=True)
    for name, theta in (("best", best), ("final", final)):
        # each world from a fresh network (paired per world), then the 30 in a row from one network as
        # untrained.py and G2 played them (the E-R state carries over from world to world)
        rewards = [e.play(game, es.build(p, 0, theta), p, [s])["reward"] for s in seeds]
        pl = es.build(p, 0, theta)
        r = e.play(game, pl, p, seeds, meter=True)
        r["spikes"] = pl.spikes / max(pl.ticks_seen, 1) * p["ticks"]
        r["reward_by_world"] = rewards
        out[f"er_rec/{d}/{name}"] = r
        print(d, name, {k: round(x, 2) for k, x in r.items() if not isinstance(x, list)}, flush=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
