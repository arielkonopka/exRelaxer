#!/usr/bin/env python3
"""Reruns recorded trials from results/ on the current build and compares their metrics.

Every nntest trial line carries its experiment, parameters and seed, so a
trial can be rerun exactly: `run EXPERIMENT --set NAME=VALUE ... --seed SEED
--trials 1`. This tool samples a few parameter combinations per result file,
reruns them and says, per trial, whether every metric is the same as recorded
(`same`), close (`close`, all within --tol relative), or changed (`changed`,
with the metrics that moved most). Timing metrics (names with us, seconds,
time, ...) are not compared; metrics only one of the two runs has are listed
but not counted as a change.

    NNtesting/tools/reproduce.py results/rerun/raw/*.jsonl.gz --out repro.jsonl
    NNtesting/tools/reproduce.py results --per-file 2 --max-seconds 60
    NNtesting/tools/reproduce.py results --only changed@repro.jsonl --legacy --out legacy.jsonl

Limits: a parameter whose recorded value contains a comma cannot be passed
with --set (the harness splits on commas); it is left out, so the
experiment's current default applies, and the trial is marked `list_param`.
A parameter recorded with an empty value is left out too, and so is one whose
value is a path (an output file or folder, which the rerun must not overwrite). A recorded parameter
the experiment no longer has is left out (`gone`), and one it has gained since
runs at its default (`new`): a changed result with `new` may be that
parameter's default, not a change in the code.
Python experiments run with NNtesting/nntest.py (PYTHONPATH must reach the
exrelaxer package).
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_trials(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "trial" and "experiment" in d and "params" in d and not d.get("error"):
                yield d


def sample(trials, per_file):
    """Up to per_file trials with distinct parameters, spread over the file."""
    seen, distinct = set(), []
    for t in trials:
        key = (t["experiment"], json.dumps(t["params"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            distinct.append(t)
    if len(distinct) <= per_file:
        return distinct
    step = (len(distinct) - 1) / max(per_file - 1, 1)
    return [distinct[round(i * step)] for i in range(per_file)]


# --legacy: the value a parameter added later must take to give the behaviour of a run recorded
# before it existed: (value, recorded before this UTC time or None for always). Only applied when
# the recorded trial lacks the parameter.
LEGACY = {
    "growth": ("log", None),                    # linear became the default 2026-09-27 (PR #10)
    "normalize": ("false", None),               # added 2026-09-27 (PR #15) as off; on since PR #25
    "readout_normalize": ("false", None),
    "spontaneous_amplitude": ("0.01", None),    # 0.1 since 2026-09-29 (PR #25)
    "habituation": ("false", None),
    "pretrain": ("0", None),
    # no recovery parameter before 2026-10-26: the library default, 0.9 until PR #48
    "recovery": ("0.9", "2026-10-08T21:16:00Z"),
}


def known_experiments(nntest, nntest_py):
    cpp, py = set(), set()
    for cmd, out in ((nntest, cpp), (nntest_py, py)):
        try:
            text = subprocess.run(cmd + ["list"], capture_output=True, text=True, timeout=120).stdout
        except Exception:
            continue
        for line in text.splitlines():
            if line.strip():
                out.add(line.split()[0])
    return cpp, py


def describe_params(cmd, experiment, cache):
    """Names of the parameters `experiment` declares now (from `describe`)."""
    if experiment in cache:
        return cache[experiment]
    text = subprocess.run(cmd + ["describe", experiment], capture_output=True, text=True, timeout=120).stdout
    names, inside = {}, False
    for line in text.splitlines():
        if line.startswith("parameters:"):
            inside = True
        elif inside and line.startswith("  ") and line.strip():
            parts = line.split()
            names[parts[0]] = parts[1] if len(parts) > 1 else ""
        elif inside:
            inside = False
    cache[experiment] = names
    return names


def rerun(cmd, trial, declared, threads, timeout, legacy=False):
    notes = []
    args = cmd + ["run", trial["experiment"], "--trials", "1", "--seed", str(trial["seed"])]
    if threads:
        args += ["--threads", str(threads)]
    for name, value in sorted(trial["params"].items()):
        value = str(value)
        if declared and name not in declared:
            notes.append(f"gone:{name}")
            continue
        if "/" in value:  # an output path (dump, trace): the rerun must not write into results/
            notes.append(f"path_param:{name}")
            continue
        if "," in value or value == "":
            if value and value != declared.get(name):  # an empty value: assumed to be the (empty) default
                notes.append(f"list_param:{name}")
            continue
        args += ["--set", f"{name}={value}"]
    recorded_at = trial.get("env", {}).get("time", "")
    for n, v in declared.items():
        if n in trial["params"]:
            continue
        if legacy and n in LEGACY and (LEGACY[n][1] is None or recorded_at < LEGACY[n][1]):
            args += ["--set", f"{n}={LEGACY[n][0]}"]
            notes.append(f"legacy:{n}={LEGACY[n][0]}")
        else:
            notes.append(f"new:{n}={v}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.jsonl"
        try:
            proc = subprocess.run(args + ["--out", str(out)], capture_output=True, text=True,
                                  timeout=timeout, cwd=ROOT)
        except subprocess.TimeoutExpired:
            return None, notes + ["timeout"]
        new = list(read_trials(out)) if out.exists() else []
        if not new:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            return None, notes + ["failed: " + " | ".join(tail)]
        return new[0], notes


TIMING = ("_us", "us_per", "seconds", "_ms", "per_second", "throughput", "time")


def is_timing(name):
    return any(t in name for t in TIMING)


def compare(old, new, tol):
    """Compares the metrics both runs have, timings excepted; returns (status, diffs, added, removed)."""
    a, b = old.get("metrics", {}), new.get("metrics", {})
    added = sorted(k for k in set(b) - set(a) if not is_timing(k))
    removed = sorted(k for k in set(a) - set(b) if not is_timing(k))
    diffs = []
    for k in sorted(set(a) & set(b)):
        if is_timing(k):
            continue
        x, y = a[k], b[k]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y:
                diffs.append((k, x, y, math.inf))
            continue
        if x == y or (math.isnan(x) and math.isnan(y)):
            continue
        scale = max(abs(x), abs(y), 1e-12)
        diffs.append((k, x, y, abs(x - y) / scale))
    if not diffs:
        return "same", [], added, removed
    if all(d[3] <= tol for d in diffs):
        return "close", diffs, added, removed
    return "changed", sorted(diffs, key=lambda d: -d[3]), added, removed


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="+", help="result files or folders (searched for *.jsonl[.gz])")
    p.add_argument("--per-file", type=int, default=2, help="parameter combinations per file (default 2)")
    p.add_argument("--max-seconds", type=float, default=120, help="skip trials that took longer (default 120)")
    p.add_argument("--tol", type=float, default=1e-6, help="relative difference still called close")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--nntest", default=str(ROOT / "build/NNtesting/nntest"))
    p.add_argument("--out", help="append one JSON line per rerun trial")
    p.add_argument("--legacy", action="store_true",
                   help="give parameters added after a run their old behaviour (LEGACY) instead of their default")
    p.add_argument("--only", metavar="STATUS@FILE",
                   help="rerun only the trials a previous --out FILE gave a status starting with STATUS")
    o = p.parse_args()

    files = []
    for path in map(Path, o.paths):
        files += sorted(path.rglob("*.jsonl*")) if path.is_dir() else [path]
    nntest = [o.nntest]
    nntest_py = [sys.executable, str(ROOT / "NNtesting/nntest.py")]
    cpp, py = known_experiments(nntest, nntest_py)
    cache = {}
    wanted = None
    if o.only:
        prefix, _, previous = o.only.partition("@")
        wanted = set()
        for line in open(previous):
            r = json.loads(line)
            if r["status"].startswith(prefix):
                wanted.add((r["file"], r["seed"], json.dumps(r["params"], sort_keys=True)))
        files = sorted({ROOT / f for f, _, _ in wanted})
    out = open(o.out, "a") if o.out else None
    counts = {}
    for f in files:
        trials = sample(read_trials(f), o.per_file)
        if wanted is not None:
            rel = str(f.relative_to(ROOT))
            trials = [t for t in read_trials(f)
                      if (rel, t["seed"], json.dumps(t["params"], sort_keys=True)) in wanted]
            trials = list({json.dumps([t["seed"], t["params"]], sort_keys=True): t for t in trials}.values())
        for t in trials:
            exp = t["experiment"]
            row = {"file": str(f.relative_to(ROOT) if f.is_absolute() and ROOT in f.parents else f),
                   "experiment": exp, "seed": t["seed"], "params": t["params"],
                   "recorded_commit": t.get("env", {}).get("git")}
            if exp in cpp:
                cmd = nntest
            elif exp in py:
                cmd = nntest_py
            else:
                row.update(status="unknown_experiment")
                cmd = None
            if cmd and t.get("seconds", 0) > o.max_seconds:
                row.update(status="skipped_slow", seconds=t.get("seconds"))
                cmd = None
            if cmd:
                new, notes = rerun(cmd, t, describe_params(cmd, exp, cache), o.threads,
                                   timeout=max(600, 20 * t.get("seconds", 1)), legacy=o.legacy)
                row["notes"] = notes
                if new is None:
                    row["status"] = "error"
                else:
                    status, diffs, added, removed = compare(t, new, o.tol)
                    row.update(metrics_added=added, metrics_removed=removed)
                    flags = sorted({n.split(":")[0] for n in notes if n.startswith(("list_param", "gone", "new"))})
                    if status != "same" and flags:
                        status += "+" + ",".join(flags)
                    row.update(status=status, seconds=new.get("seconds"),
                               diffs=[{"metric": k, "old": x, "new": y, "rel": (None if r == math.inf else r)}
                                      for k, x, y, r in diffs[:8]])
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            top = row.get("diffs", [{}])[0] if row.get("diffs") else {}
            print(f"{row['status']:<22} {exp:<20} {row['file']}"
                  + (f"  {top.get('metric')}: {top.get('old')} -> {top.get('new')}" if top else ""), flush=True)
            if out:
                out.write(json.dumps(row) + "\n")
                out.flush()
    print("totals:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
