"""Plasticity potential instead of maturation (research log §72): every neuron of the adaptive E-R
layer has a P in [0, 1] that fades while its local error (its feedback-alignment share of the
readout error) stays below epsilon and rises with the error otherwise; its weight updates are
scaled by P (LearningRule.with_plasticity, doc/learning.md#plasticity-potential). Nothing is
frozen at growth: P decides who keeps changing. Compared with growth as it is (older neurons
frozen), §39's best maturation arm (nothing frozen, grown neurons young for good) and fixed
ER(16), with and without P.

    python NNtesting/experiments/snake_growth/plasticity.py OUT_DIR [--seeds 20] [--games 3000]
    python NNtesting/experiments/snake_growth/plasticity.py OUT_DIR --summary

Setup: §37's one-cell start (ER(1), recovery 0.5, growth on 25% saturated ticks, 4 neurons per
growth). Each run writes OUT_DIR/<arm>_s<seed>.json.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiment as sg  # this folder's experiment.py

START = {"start": 1, "recovery": 0.5, "saturated_share": 0.25, "increment": 4}
FIXED = {"start": 16, "recovery": 0.5, "grow": False}
YOUNG = dict(mature_ticks=1, mature_resting=0.5, mature_gain=3.0, mature_end_resting=0.5, mature_end_gain=3.0)


def P(alpha, epsilon, **more):
    return dict(plasticity=True, p_alpha=alpha, p_epsilon=epsilon, **more)


ARMS = {
    "fixed16": FIXED,
    "fixed16_P_a01_e01": dict(FIXED, **P(0.01, 0.1)),
    "fixed16_P_a001_e001": dict(FIXED, **P(0.001, 0.01)),
    "grow": START,                                                   # older neurons frozen at growth
    "grow_nofreeze": dict(START, freeze_old=False),
    "young_nofreeze": dict(START, freeze_old=False, **YOUNG),        # §39's best one-cell start
    "P_a001_e001": dict(START, freeze_old=False, **P(0.001, 0.01)),
    "P_a01_e001": dict(START, freeze_old=False, **P(0.01, 0.01)),
    "P_a001_e01": dict(START, freeze_old=False, **P(0.001, 0.1)),
    "P_a01_e01": dict(START, freeze_old=False, **P(0.01, 0.1)),
    "P_a01_e01_readouts": dict(START, freeze_old=False, **P(0.01, 0.1, p_readouts=True)),
    # participation: the local error is |error| x min(1, |output|); silent neurons settle
    "fixed16_Pp_a01_e01": dict(FIXED, **P(0.01, 0.1, p_participation=True)),
    "fixed16_Pp_a001_e001": dict(FIXED, **P(0.001, 0.01, p_participation=True)),
    "fixed16_Pp_a01_e01_readouts": dict(FIXED, **P(0.01, 0.1, p_participation=True, p_readouts=True)),
    "Pp_a01_e01": dict(START, freeze_old=False, **P(0.01, 0.1, p_participation=True)),
    "Pp_a001_e001": dict(START, freeze_old=False, **P(0.001, 0.01, p_participation=True)),
    "Pp_a01_e01_readouts": dict(START, freeze_old=False, **P(0.01, 0.1, p_participation=True, p_readouts=True)),
    # weight feedback: the hidden error comes through the readouts' real weights
    "fixed16_wfb": dict(FIXED, weight_feedback=True),
    "fixed16_wfb_P_a01_e01": dict(FIXED, weight_feedback=True, **P(0.01, 0.1)),
    "fixed16_wfb_P_a001_e001": dict(FIXED, weight_feedback=True, **P(0.001, 0.01)),
    "wfb_nofreeze": dict(START, freeze_old=False, weight_feedback=True),
    "wfb_P_a01_e01": dict(START, freeze_old=False, weight_feedback=True, **P(0.01, 0.1)),
    "wfb_P_a001_e001": dict(START, freeze_old=False, weight_feedback=True, **P(0.001, 0.01)),
}


def job(args):
    out, name, seed, games = args
    p = dict(sg.DEFAULTS, **ARMS[name], games=games)
    log = []
    agent, m = sg.develop(p, seed, report=log.append)
    m = {k: (v if k == "diversity" else [float(x) for x in v] if isinstance(v, list) else float(v))
         for k, v in m.items()}
    m.update(arm=name, seed=seed, params=p, sizes=agent.sizes(), log=log)
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


def summary(out, against="fixed16"):
    runs = load(out)
    base = runs.get(against, {})
    final = lambda m: m["final"]
    area = lambda m: float(np.mean(m["curve"]))
    print(f"| arm | seeds | final | Δ final vs {against} | Δ all vs {against} | neurons | widths | "
          "mean P (last 3) | share P < 0.1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in ARMS:
        ms = runs.get(name)
        if not ms:
            continue
        v = list(ms.values())
        if name != against and base:
            df, sf = paired(ms, base, final)
            da, sa = paired(ms, base, area)
            delta = f"{df:+.2f} ± {sf:.2f} | {da:+.2f} ± {sa:.2f}"
        else:
            delta = " | "
        if "p_mean" in v[0]["diversity"][-1]:
            pm = np.mean([np.mean([d["p_mean"] for d in m["diversity"][-3:]]) for m in v])
            pl = np.mean([np.mean([d["p_low"] for d in m["diversity"][-3:]]) for m in v])
            pcol = f"{pm:.2f} | {pl:.2f}"
        else:
            pcol = "- | -"
        print(f"| {name} | {len(v)} | {np.mean([final(m) for m in v]):.2f} | {delta} | "
              f"{np.mean([m['size'] for m in v]):.1f} | {np.mean([m['width_events'] for m in v]):.1f} | {pcol} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--against", default="fixed16")
    args = parser.parse_args()
    if args.summary:
        summary(args.out, args.against)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        todo = [(args.out, name, seed, args.games) for seed in range(1, args.seeds + 1) for name in args.arms
                if not (Path(args.out) / f"{name}_s{seed}.json").exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            for name, seed, final in pool.map(job, todo):
                print(f"{name} s{seed}: {final:.2f}", flush=True)
