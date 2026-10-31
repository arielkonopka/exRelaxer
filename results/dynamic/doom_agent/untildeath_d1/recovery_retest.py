"""D15: replay a saved Doom agent on D12's 30 fresh games (seed 4242) at a given recovery.
    python3 results/dynamic/doom_agent/untildeath_d1/recovery_retest.py AGENT_DIR THETA RECOVERY GAMES OUT.json"""
import json, sys, time, numpy as np
sys.path.insert(0, "NNtesting/experiments/doom_rl")
import es, experiment as e
d, theta, rec, games, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
p = json.load(open(d + "/config.json"))["params"]
if rec != "none":
    p["recovery"] = float(rec)
es.init_worker(p, 0)
th = None if theta == "untrained" else np.load(d + "/" + theta)
player = es.build(p, 0, th); g = es._worker["game"]; g.set_seed(4242)
t = time.time()
s, _ = e.play(g, player, p, games, 0.0, 0.0, np.random.default_rng(0), meter=True)
r = {k: round(s[k], 2) for k in ("reward", "kills", "deaths")}
r["survive_s"] = round(s["steps"] * 4 / 35, 1)
r["spikes"] = round(player.spikes / max(player.ticks_seen, 1) * p["ticks"])
r["wall_s"] = round(time.time() - t)
r.update(agent=d, theta=theta, recovery=rec, games=games)
print(r, flush=True)
json.dump(r, open(out, "w"))
es.close_game()
