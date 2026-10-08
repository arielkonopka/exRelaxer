"""D14: 30 fresh games (seed 4242, until death or 30 minutes) for one agent."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/claude/exRelaxer/NNtesting/experiments/doom_rl")
import es, experiment as e
run, weights, name, out = sys.argv[1:5]
p = json.load(open(f"{run}/config.json"))["params"]
if weights.endswith(".npz"):
    s = np.load(weights); p["depth"] = int(s["depth"]); th = s["theta"]
else:
    th = np.load(weights)
    bj = f"{run}/best.json"
    try:
        p["depth"] = int(json.load(open(bj))["depth"])
    except Exception:
        pass
es.init_worker(p, 0)
player = es.build(p, 0, th); g = es._worker["game"]; g.set_seed(4242)
s, _ = e.play(g, player, p, 30, 0.0, 0.0, np.random.default_rng(0), meter=True)
r = {k: round(s[k], 2) for k in ("reward", "kills", "deaths")}
r["survive_s"] = round(s["steps"] * e.FRAME_SKIP / 35, 1)
r["spikes"] = round(player.spikes / max(player.ticks_seen, 1) * p["ticks"])
r["depth"] = p["depth"]; r["thresholds"] = bool(p.get("readout_thresholds", False))
print(name, r, flush=True)
json.dump({name: r}, open(out, "w"))
