#!/usr/bin/env python3
"""Saved Gardens of Eris agents of grow.py: export a run, load it back, play it.

    python3 NNtesting/experiments/goe_rl/agent.py export RUN_DIR OUT_DIR   # the run's current weights
    python3 NNtesting/experiments/goe_rl/agent.py check OUT_DIR            # reloads play exactly as exported

An exported agent is a folder:
    agent.json   columns.py parameters, columns, stage, generation, net seed, what each file holds
    weights.npz  every column's weights by layer (c<k>: kernels, h<k>: neuron weight vectors,
                 as "<layer>_<index>") and "readouts" (14 x width * columns)
    net.exr      the same network in the library's format (exr.Network.load), with the E-R state
                 as it was built (fresh), for experiments that use the library directly

load(folder) rebuilds a columns.Net with every weight in place: its layers (net.convs,
net.hidden, net.readouts) are the handles to train with other rules (they are built frozen,
since ES set them; unfreeze what you train). load_exr(folder) wraps the .exr file the same
way, by layer name.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import columns as C  # noqa: E402


def export(run_dir, out_dir):
    """The run's current network (state.npz, frozen.npz, config.json) as an agent folder."""
    config, p, meta, net = _read_run(run_dir)
    net_seed = config["args"]["net_seed"]
    os.makedirs(out_dir, exist_ok=True)
    weights = {f"{name}_{i}": w for name, vectors in net.frozen().items() for i, w in enumerate(vectors)}
    weights["readouts"] = np.stack([net._vectors(r)[0] for r in net.readouts])
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    net.net.save(os.path.join(out_dir, "net.exr"))
    info = {"params": p, "columns": meta["columns"], "stage": meta["stage"], "generation": meta["gen"],
            "net_seed": net_seed, "curriculum": config["params"].get("curriculum"),
            "grow_columns": config["params"].get("grow_columns"), "source": os.path.abspath(run_dir),
            "actions": C.ACTIONS, "channels": C.CHANNELS, "body": ["energy", "ammo", "avatars"] + C.SECTIONS,
            "files": {"weights.npz": "c<k>_<i> kernels, h<k>_<i> neuron weights, readouts (actions x inputs)",
                      "net.exr": "the library network (exr.Network.load)"}}
    with open(os.path.join(out_dir, "agent.json"), "w") as f:
        json.dump(info, f, indent=1)
    return out_dir


def load(folder):
    """The agent as a columns.Net, every weight from weights.npz."""
    with open(os.path.join(folder, "agent.json")) as f:
        info = json.load(f)
    p = C.defaults() | info["params"]
    w = np.load(os.path.join(folder, "weights.npz"))
    net = C.Net(p, info["columns"], _by_layer(w), None, info["net_seed"])
    for r, row in zip(net.readouts, w["readouts"]):
        net._set_vectors(r, [row])
    return net


class ExrAgent(C.Net):
    """A columns.Net whose network is the saved .exr file, found by layer name."""

    def __init__(self, folder):
        import exrelaxer as exr
        with open(os.path.join(folder, "agent.json")) as f:
            info = json.load(f)
        self.p, self.columns = C.defaults() | info["params"], info["columns"]
        self.net = exr.Network.load(os.path.join(folder, "net.exr"))
        find = self.net.find_layer
        self.eyes = ([find("retina")] if self.p["retina"] else []) + [find("raw")]
        self.convs = [find(f"c{k}") for k in range(1, self.columns + 1)]
        self.pools = [find(f"p{k}") for k in range(1, self.columns + 1)]
        self.hidden = [find(f"h{k}") for k in range(1, self.columns + 1)]
        self.readouts = [find(a.lower()) for a in C.ACTIONS]
        self.metered = {"retina": self.eyes[:-1], "conv": self.convs, "hidden": self.hidden}
        self.spikes = {k: 0 for k in self.metered}
        self.ticks_seen = 0
        self.rng = np.random.default_rng(0)


def load_exr(folder):
    return ExrAgent(folder)


def check(folder, worlds=None):
    """The exported agent, rebuilt from weights.npz and from net.exr, plays the same games as the
    run's own network (each world from a fresh network)."""
    import exrelaxer as exr
    exr.set_threads(1)
    with open(os.path.join(folder, "agent.json")) as f:
        info = json.load(f)
    worlds = worlds or [(k, 777000 + i) for i, k in enumerate(C.SKILLS)]
    p = C.defaults() | info["params"]
    game = C.make_game(p)
    out = {}
    run = info["source"]
    makers = [("weights.npz", lambda: load(folder)), ("net.exr", lambda: load_exr(folder))]
    if os.path.exists(os.path.join(run, "state.npz")):  # the run itself, while its folder exists
        makers.insert(0, ("run", lambda: _from_run(run)))
    for name, make in makers:
        rewards = []
        for world in worlds:
            r, _ = C.play(game, make(), p, [world])
            rewards.append(float(r[0]))
        out[name] = rewards
        print(f"{name:12s}", np.round(rewards, 2))
    same = all(r == out["net.exr"] for r in out.values())
    print("identical play" if same else "PLAY DIFFERS")
    return same


def _by_layer(npz):
    """{layer: [vector, ...]} from "<layer>_<index>" keys (readouts left out)."""
    out = {}
    for key in npz.files:
        if key != "readouts":
            name, i = key.rsplit("_", 1)
            out.setdefault(name, {})[int(i)] = npz[key]
    return {k: [d[i] for i in sorted(d)] for k, d in out.items()}


def _read_run(run_dir):
    """(config, parameters, state meta, the run's network) from a grow.py output folder."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    p = C.defaults() | {k: v for k, v in config["params"].items() if k in C.PARAMS}
    s = np.load(os.path.join(run_dir, "state.npz"))
    meta = json.loads(str(s["meta"]))
    frozen = _by_layer(np.load(os.path.join(run_dir, "frozen.npz")))
    return config, p, meta, C.Net(p, meta["columns"], frozen, s["theta"], config["args"]["net_seed"])


def _from_run(run_dir):
    return _read_run(run_dir)[3]

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "export":
        print(export(sys.argv[2], sys.argv[3]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    else:
        print(__doc__)
        sys.exit(2)
