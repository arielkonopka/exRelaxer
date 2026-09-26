#!/usr/bin/env python3
"""Summarize and compare nntest result files (JSON Lines, see NNtesting/README.md).

  compare.py RESULTS.jsonl...                  one table per experiment: a row per
                                               parameter combination, only the
                                               parameters that vary as columns
  compare.py --baseline BASE.jsonl NEW.jsonl   the same combinations in both:
                                               change of every metric, with a
                                               Welch t statistic

Options:
  --metric NAME   only this metric (repeatable)
  --t T           |t| at or above which a change is flagged (default 2)

Standard library only.
"""
import argparse
import json
import math
import sys
from collections import OrderedDict


def load(paths):
    """Summary records keyed by (experiment, params), the last one winning."""
    summaries = OrderedDict()
    runs = OrderedDict()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{number}: not JSON ({e})")
                if record.get("type") != "summary":
                    continue
                key = (record["experiment"], tuple(sorted(record["params"].items())))
                summaries[key] = record
                env = record.get("env", {})
                runs[env.get("run")] = env
    return summaries, runs


def describe_runs(runs, label):
    for env in runs.values():
        dirty = " (dirty)" if env.get("dirty") else ""
        native = " native" if env.get("native") else ""
        print(f"{label}run {env.get('run')}: git {env.get('git')}{dirty}, {env.get('cpu')}, "
              f"{env.get('threads')} threads, {env.get('build')}{native}, {env.get('compiler')}")


def fmt(value):
    if value is None:
        return "-"
    if value == 0 or 1e-3 <= abs(value) < 1e5:
        return f"{value:.4g}"
    return f"{value:.3e}"


def print_table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def varying_params(keys):
    values = {}
    for _, params in keys:
        for name, value in params:
            values.setdefault(name, set()).add(value)
    return [name for name, seen in values.items() if len(seen) > 1]


def selected_metrics(records, wanted):
    names = []
    for record in records:
        for name in record["metrics"]:
            if name not in names and (not wanted or name in wanted):
                names.append(name)
    return names


def summary_mode(args):
    summaries, runs = load(args.results)
    describe_runs(runs, "")
    by_experiment = OrderedDict()
    for key, record in summaries.items():
        by_experiment.setdefault(key[0], []).append((key, record))
    for experiment, items in by_experiment.items():
        print(f"\n{experiment}")
        keys = [k for k, _ in items]
        params = varying_params(keys)
        metrics = selected_metrics([r for _, r in items], args.metric)
        header = params + [f"{m} (mean +- se)" for m in metrics] + ["trials"]
        rows = []
        for (_, p), record in items:
            p = dict(p)
            row = [p[name] for name in params]
            for m in metrics:
                s = record["metrics"].get(m)
                row.append("-" if s is None else f"{fmt(s['mean'])} +- {fmt(s['stderr'])}")
            row.append(f"{record['trials'] - record['failed']}/{record['trials']}")
            rows.append(row)
        print_table(header, rows)


def welch_t(a, b):
    """t of mean(b) - mean(a) from two summaries (n, mean, sd)."""
    va = (a["sd"] or 0) ** 2 / a["n"] if a["n"] else 0
    vb = (b["sd"] or 0) ** 2 / b["n"] if b["n"] else 0
    diff = b["mean"] - a["mean"]
    if va + vb == 0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return diff / math.sqrt(va + vb)


def baseline_mode(args):
    base, base_runs = load([args.baseline])
    new, new_runs = load(args.results)
    describe_runs(base_runs, "baseline ")
    describe_runs(new_runs, "new      ")
    common = [k for k in new if k in base]
    if not common:
        sys.exit("no parameter combination appears in both")
    rows = []
    flagged = 0
    for key in common:
        experiment, params = key
        a, b = base[key], new[key]
        for m in selected_metrics([a, b], args.metric):
            sa, sb = a["metrics"].get(m), b["metrics"].get(m)
            if sa is None or sb is None or sa["mean"] is None or sb["mean"] is None:
                continue
            t = welch_t(sa, sb)
            change = (sb["mean"] - sa["mean"]) / abs(sa["mean"]) * 100 if sa["mean"] else math.nan
            flag = "*" if abs(t) >= args.t else ""
            flagged += bool(flag)
            varying = " ".join(f"{k}={v}" for k, v in params if k in varying_params(common))
            rows.append([experiment, varying, m, fmt(sa["mean"]), fmt(sb["mean"]),
                         "-" if math.isnan(change) else f"{change:+.1f}%", f"{t:+.2f}", flag])
    print()
    print_table(["experiment", "params", "metric", "baseline", "new", "change", "t", ""], rows)
    print(f"\n{flagged} change(s) with |t| >= {args.t} (*)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="nntest result files (JSON Lines)")
    parser.add_argument("--baseline", help="compare RESULTS against this file")
    parser.add_argument("--metric", action="append", help="only this metric (repeatable)")
    parser.add_argument("--t", type=float, default=2.0, help="|t| at or above which a change is flagged")
    args = parser.parse_args()
    if args.baseline:
        baseline_mode(args)
    else:
        summary_mode(args)


if __name__ == "__main__":
    main()
