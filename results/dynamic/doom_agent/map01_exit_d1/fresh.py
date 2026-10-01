"""D13: 30 fresh MAP01 games (seed 4242) for one set of weights; prints and saves a JSON line."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/claude/exRelaxer/NNtesting/experiments/doom_rl")
import es, experiment as e, rewards
O, name, theta_path, out = sys.argv[1:5]
p = json.load(open(f"{O}/map01_exit/config.json"))["params"]
closest, last = [], []
_reset = rewards.Shaper.reset
def reset(self, state):  # record each game's closest walking distance to the exit
    if getattr(self, "closest", None) is not None:
        closest.append(self.closest)
    last[:] = [self]
    return _reset(self, state)
rewards.Shaper.reset = reset
es.init_worker(p, 0)
th = None if theta_path == "none" else np.load(theta_path)
player = es.build(p, 0, th); g = es._worker["game"]; g.set_seed(4242)
s, parts = e.play(g, player, p, 30, 0.0, 0.0, np.random.default_rng(0), meter=True)
closest.append(last[0].closest)
r = {k: round(s[k], 2) for k in ("reward", "kills", "deaths", "exits", "cells", "items", "doors", "steps")}
r["minutes"] = round(s["steps"] * e.FRAME_SKIP / 35 / 60, 2)
r["spikes"] = round(player.spikes / max(player.ticks_seen, 1) * p["ticks"])
r["closest_mean"] = round(float(np.mean(closest)), 0) if closest else None
r["parts"] = {k: round(float(v), 2) for k, v in parts.items()} if isinstance(parts, dict) else None
print(name, r, flush=True)
json.dump({name: r}, open(out, "w"))
