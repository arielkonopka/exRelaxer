"""Best and final weights of each run, the untrained network and a random player on 30 fresh worlds."""
import json, sys, numpy as np
sys.path.insert(0, "NNtesting/experiments/goe_rl")
import exrelaxer as exr, experiment as e, es
exr.set_threads(1)
S = sys.argv[1]
seeds = [4242 + i for i in range(30)]
out = {}
for run, cfg in (("er_rec", {"recurrent": True}), ("er_nohab", {"habituation": False})):
    p = es.params(cfg) if hasattr(es, "params") else None
    p = dict({k: v[0] for k, v in e.PARAMS.items()}, evolve="all", **cfg)
    p["radius"] = 6
    g = e.make_game(p)
    # consistency: best weights on the validation worlds give the logged best
    best = np.load(f"{S}/{run}/best_theta.npy"); final = np.load(f"{S}/{run}/state.npz")["theta"]
    log = [json.loads(l) for l in open(f"{S}/{run}/log.jsonl")]
    v = e.play(g, es.build(p, 0, best), p, [777777 + i for i in range(6)])["reward"]
    print(run, "validation check", round(v, 2), "logged best", round(max(l["validation_reward"] for l in log), 2), flush=True)
    for name, theta in (("untrained", None), ("best", best), ("final", final)):
        pl = es.build(p, 0, theta)
        r = e.play(g, pl, p, seeds, meter=True)
        r["spikes"] = pl.spikes / max(pl.ticks_seen, 1) * p["ticks"]
        out[f"{run}/{name}"] = r
        print(run, name, {k: round(x, 2) for k, x in r.items()}, flush=True)
    g.close()
p = dict({k: v[0] for k, v in e.PARAMS.items()}, radius=6)
g = e.make_game(p)
out["random"] = e.play(g, e.RandomPlayer(0), p, seeds)
print("random", {k: round(x, 2) for k, x in out["random"].items()})
json.dump(out, open(f"{S}/fresh.json", "w"), indent=1)
