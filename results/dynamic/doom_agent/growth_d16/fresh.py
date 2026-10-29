"""D16: an es.py run's best weights on 30 fresh games (seed 4242): 1-minute games and games until death (30-min cap).
    python3 fresh.py RUN_DIR [RUN_DIR ...] > out.jsonl"""
import json, sys, numpy as np
sys.path.insert(0, "NNtesting/experiments/doom_rl")
import es, experiment as e
for d in sys.argv[1:]:
    p = json.load(open(d + "/config.json"))["params"]
    b = json.load(open(d + "/best.json"))
    p.update(width=b["width"], young_from=b.get("young_from"))
    for name, tics in (("1min", 2100), ("until_death", 63000)):
        q = dict(p, episode_tics=tics)
        es._worker.pop("game", None)
        es.init_worker(q, 0)
        for theta in ("best", "untrained"):
            player = es.build(q, 0, np.load(d + "/best_theta.npy") if theta == "best" else None)
            g = es._worker["game"]; g.set_seed(4242)
            s, _ = e.play(g, player, q, 30, 0.0, 0.0, np.random.default_rng(0), meter=True)
            r = {"run": d.rstrip("/").split("/")[-1], "games": name, "theta": theta, "width": q["width"],
                 "reward": round(s["reward"], 2), "kills": round(s["kills"], 2), "deaths": round(s["deaths"], 2),
                 "survive_s": round(s["steps"] * 4 / 35, 1), "spikes": round(player.spikes / max(player.ticks_seen, 1) * q["ticks"])}
            print(json.dumps(r), flush=True)
        es._worker["game"].close()
