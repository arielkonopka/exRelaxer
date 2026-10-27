"""A critical period for grown neurons (research log §39): newborn neurons start hyperexcitable
(a low E-R resting threshold) and more plastic (a larger learning gain), then mature over
`mature_ticks` into neurons like the rest (exrelaxer.development.Maturation), as adult-born
hippocampal granule cells do. Compared with §38's grown neurons that rest lower for good, and with
the current init; every run also records the E-R layer's firing-pattern diversity per evaluation
window (PatternDiversity), and two arms grow on falling diversity instead of saturation.

    python NNtesting/experiments/snake_growth/maturation.py OUT_DIR                  # every arm
    python NNtesting/experiments/snake_growth/maturation.py OUT_DIR --summary        # the tables

Setup: §37's best one-cell start (ER(1), recovery 0.5, growth on 25% saturated ticks, 4 neurons
per growth), 1,500 games. Each run writes OUT_DIR/<arm>_s<seed>.json (as copy_init.py).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment as sg  # this folder's experiment.py

START = {"start": 1, "recovery": 0.5, "saturated_share": 0.25, "increment": 4}


def mature(ticks, resting=0.5, gain=1.0, end_resting=1.0, end_gain=1.0, **more):
    return dict(mature_ticks=ticks, mature_resting=resting, mature_gain=gain, mature_end_resting=end_resting,
                mature_end_gain=end_gain, **more)


ARMS = {
    "random": {},                                                   # the current init
    "rest05": {"grown_resting": 0.5},                               # §38: rests at x 0.5 for good
    "gain3": mature(1, 1.0, 3.0, 1.0, 3.0),                         # learns 3x faster for good
    "rest05_gain3": mature(1, 0.5, 3.0, 0.5, 3.0),                  # both, for good
    # Maturation: the threshold rises from x 0.5 to the layer's
    "mat_rest_2k": mature(2000),
    "mat_rest_10k": mature(10000),
    "mat_rest_50k": mature(50000),
    # ... the learning gain falls from 3x to 1x
    "mat_gain_10k": mature(10000, 1.0, 3.0),
    "mat_gain_50k": mature(50000, 1.0, 3.0),
    # ... both (the hippocampal picture)
    "mat_both_2k": mature(2000, 0.5, 3.0),
    "mat_both_10k": mature(10000, 0.5, 3.0),
    "mat_both_50k": mature(50000, 0.5, 3.0),
    # hyperexcitable at birth, settling at §38's x 0.5
    "mat_rest025_to05_10k": mature(10000, 0.25, 3.0, 0.5),
    # the critical period closes: matured neurons freeze, and growth no longer freezes the others
    "mat_both_10k_close": mature(10000, 0.5, 3.0, mature_freeze=True, freeze_old=False),
    "mat_both_50k_close": mature(50000, 0.5, 3.0, mature_freeze=True, freeze_old=False),
    "random_nofreeze": {"freeze_old": False},                       # control for the two above
    "nofreeze_rest05_gain3": mature(1, 0.5, 3.0, 0.5, 3.0, freeze_old=False),   # young for good, nothing frozen
    "mat_both_100k": mature(100000, 0.5, 3.0),
    "mat_both_100k_close": mature(100000, 0.5, 3.0, mature_freeze=True, freeze_old=False),
    "mat_both_20k_close": mature(20000, 0.5, 3.0, mature_freeze=True, freeze_old=False),
    # Growth when the layer's firing patterns lose diversity (entropy per neuron over 1,000 ticks
    # below the value), instead of or as well as on saturation
    "entropy_085": {"trigger": "entropy", "entropy_min": 0.85},
    "entropy_090": {"trigger": "entropy", "entropy_min": 0.9},
    "entropy_095": {"trigger": "entropy", "entropy_min": 0.95},
    "either_090": {"trigger": "either", "entropy_min": 0.9},
    "either_090_rest05": {"trigger": "either", "entropy_min": 0.9, "grown_resting": 0.5},
}


def job(args):
    out, name, seed, extra = args
    p = dict(sg.DEFAULTS, **START, **ARMS[name], **extra)
    log = []
    agent, m = sg.develop(p, seed, report=log.append)
    m = {k: (v if k == "diversity" else [float(x) for x in v] if isinstance(v, list) else float(v))
         for k, v in m.items()}
    m.update(arm=name, seed=seed, params=p, sizes=agent.sizes(), log=log,
             matured=agent.maturation.matured if agent.maturation else 0)
    (Path(out) / f"{name}_s{seed}.json").write_text(json.dumps(m))
    return name, seed, m["final"]


def load(out):
    runs = {}
    for f in sorted(Path(out).glob("*.json")):
        m = json.loads(f.read_text())
        runs.setdefault(m["arm"], {})[m["seed"]] = m
    return runs


def paired(a, b, key):
    seeds = sorted(set(a) & set(b))
    d = np.array([key(a[s]) - key(b[s]) for s in seeds])
    return d.mean(), (d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0)


def diversity(m, field, last=3):
    return float(np.mean([d[field] for d in m["diversity"][-last:]]))


def summary(out):
    runs = load(out)
    base = runs.get("random", {})
    final = lambda m: m["final"]
    area = lambda m: float(np.mean(m["curve"]))
    print("| arm | seeds | final | Δ final vs random | Δ all vs random | neurons | widths | "
          "entropy (bits) | entropy / neuron | distinct | redundancy | active |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ARMS:
        ms = runs.get(name)
        if not ms:
            continue
        v = list(ms.values())
        df, sf = paired(ms, base, final) if name != "random" and base else (0, 0)
        da, sa = paired(ms, base, area) if name != "random" and base else (0, 0)
        delta = (f"{df:+.2f} ± {sf:.2f} | {da:+.2f} ± {sa:.2f}" if name != "random" else " | ")
        cols = [np.mean([diversity(m, k) for m in v]) for k in ("entropy",)]
        per = np.mean([diversity(m, "entropy") / m["diversity"][-1]["size"] for m in v])
        print(f"| {name} | {len(v)} | {np.mean([final(m) for m in v]):.2f} | {delta} | "
              f"{np.mean([m['size'] for m in v]):.1f} | {np.mean([m['width_events'] for m in v]):.1f} | "
              f"{cols[0]:.2f} | {per:.2f} | {np.mean([diversity(m, 'distinct') for m in v]):.0f} | "
              f"{np.mean([diversity(m, 'redundancy') for m in v]):.3f} | {np.mean([diversity(m, 'active') for m in v]):.2f} |")
    correlations({k: v for k, v in runs.items() if ARMS[k].get("trigger", "saturation") == "saturation"})


def correlations(runs):
    """Does diversity predict the score? Across every run (saturation-triggered arms only: the
    entropy-triggered ones fail by construction), and within a layer size."""
    rows = [(m["curve"][k], d["entropy"], d["entropy"] / max(d["size"], 1), d["distinct"], d["redundancy"], d["size"])
            for arm in runs.values() for m in arm.values() for k, d in enumerate(m["diversity"])]
    if len(rows) < 3:
        return
    a = np.array(rows, dtype=np.float64)
    names = ("entropy", "entropy / neuron", "distinct", "redundancy", "size")
    print("\nPer evaluation window, every run: correlation of the score with")
    print("| measure | r (all windows) | r (last 3 windows, per run) | partial r given size |")
    print("|---|---|---|---|")
    last = np.array([(np.mean(m["curve"][-3:]), diversity(m, "entropy"),
                      np.mean([d["entropy"] / max(d["size"], 1) for d in m["diversity"][-3:]]),
                      diversity(m, "distinct"), diversity(m, "redundancy"), m["diversity"][-1]["size"])
                     for arm in runs.values() for m in arm.values()])
    for j, name in enumerate(names, start=1):
        r_all = np.corrcoef(a[:, 0], a[:, j])[0, 1]
        r_last = np.corrcoef(last[:, 0], last[:, j])[0, 1]
        if name == "size":
            pr = float("nan")
        else:  # residuals of score and measure after a linear fit on log size
            x = np.column_stack([np.ones(len(a)), np.log(a[:, 5])])
            res = lambda y: y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            pr = np.corrcoef(res(a[:, 0]), res(a[:, j]))[0, 1]
        print(f"| {name} | {r_all:+.2f} | {r_last:+.2f} | {pr:+.2f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in range(1, 31)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=None, help="games per run (default: the experiment's)")
    parser.add_argument("--set", action="append", default=[], help="KEY=VALUE overrides for every arm")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.out)
    else:
        extra = {"games": args.games} if args.games else {}
        for kv in args.set:
            k, v = kv.split("=", 1)
            extra[k] = type(sg.DEFAULTS[k])(v) if not isinstance(sg.DEFAULTS[k], bool) else v.lower() in ("1", "true")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        done = {(f.stem.rsplit("_s", 1)[0], int(f.stem.rsplit("_s", 1)[1])) for f in Path(args.out).glob("*.json")}
        jobs = [(args.out, n, int(s), extra) for s in args.seeds.split(",") for n in args.arms.split(",")
                if (n, int(s)) not in done]
        with ProcessPoolExecutor(args.workers) as pool:
            for name, seed, final in pool.map(job, jobs):
                print(name, seed, round(final, 2), flush=True)
        summary(args.out)
