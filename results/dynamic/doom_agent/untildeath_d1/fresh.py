import json,sys,numpy as np
sys.path.insert(0,".")
import es, experiment as e
O=sys.argv[1]
cfg=json.load(open(f"{O}/untildeath/config.json")); p=cfg["params"]
es.init_worker(p,0)
st_=np.load(f"{O}/untildeath/state.npz")
out={}
for name,th in [("final",st_["theta"]),("best",np.load(f"{O}/untildeath/best_theta.npy")),("start",np.load(f"{O}/winner_theta.npy")),("untrained",None)]:
    player=es.build(p,0,th); g=es._worker["game"]; g.set_seed(4242)
    s,_=e.play(g,player,p,30,0.0,0.0,np.random.default_rng(0),meter=True)
    out[name]={k:round(s[k],2) for k in ("reward","kills","deaths")}
    out[name]["survive_s"]=round(s["steps"]*4/35,1)
    out[name]["spikes"]=round(player.spikes/max(player.ticks_seen,1)*p["ticks"])
    print(name,out[name],flush=True)
json.dump(out,open(f"{O}/untildeath/fresh30.json","w"))
