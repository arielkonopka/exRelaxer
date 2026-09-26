"""nntest for Python: runs experiments written in Python, like the C++ nntest.

An experiment is a folder with an experiment.py (plus whatever else it needs:
helper modules, notes, configs), or, for a small one, a single .py file, in
NNtesting/experiments/. It registers itself with the decorator:

    import exrelaxer as exr
    from exrelaxer import harness as nnt

    @nnt.experiment(
        name="my_task",                                   # unique across C++ and Python
        description="what it measures, in one line",
        tags=["learning"],                                # "quick": also runs in ctest
        params={"hidden": (32, "hidden neurons"),         # name: (default, help); the default's
                "er": (False, "E-R in the hidden layer")},  # type (int, float, bool, str) parses --set
        trials=10,
        expect={"accuracy": (0.9, None)},                 # metric: (min, max), checked with defaults
    )
    def run(t):
        hidden, er = t.params["hidden"], t.params["er"]   # exr.reseed(t.seed) has been called
        ...
        t.record("accuracy", accuracy)

The runner seeds every trial (exr.reseed(seed), and t.rng is
numpy.random.default_rng(seed)), sweeps --set grids, prints per-metric
statistics, checks the expectations on default runs and appends JSON Lines
in the same format as the C++ nntest, so NNtesting/tools/compare.py reads
both.

    NNtesting/nntest.py list
    NNtesting/nntest.py describe mnist_gabor
    NNtesting/nntest.py run mnist_gabor --set mix=128,256 --trials 3 --out results.jsonl

(python -m exrelaxer.nntest, or the exr-nntest command, is the same runner;
they find NNtesting/experiments from the current directory.)
Exit status: 0 ok, 1 a check failed or a trial raised, 2 usage.
"""
import argparse
import contextlib
import fnmatch
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core, datasets

# --- Parameters ----------------------------------------------------------------


def to_text(value):
    """A parameter value as the text nntest records (bools as true/false, like C++)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_bool(text):
    lowered = text.strip().lower()
    if lowered in ("true", "on", "yes", "1"):
        return True
    if lowered in ("false", "off", "no", "0"):
        return False
    raise ValueError(f"'{text}' is not true or false")


@dataclass
class Param:
    name: str
    default: object
    help: str = ""

    def parse(self, text):
        """`text` (from --set) as the default's type."""
        kind = type(self.default)
        try:
            if kind is bool:
                return parse_bool(text)
            if kind is int:
                return int(text)
            if kind is float:
                return float(text)
        except ValueError as e:
            raise ValueError(f"parameter {self.name} = '{text}' is not {kind.__name__}: {e}") from None
        return text


class Params(dict):
    """Parameter values of one run, typed like their defaults. Unknown names raise KeyError."""

    def __missing__(self, key):
        raise KeyError(f"unknown parameter '{key}'")

    # The C++ getters, for experiments ported from there.
    def get_int(self, name):
        return int(self[name])

    def get_double(self, name):
        return float(self[name])

    def get_bool(self, name):
        value = self[name]
        return value if isinstance(value, bool) else parse_bool(str(value))

    def get_string(self, name):
        return to_text(self[name])


# --- Experiments -----------------------------------------------------------------


@dataclass
class Experiment:
    name: str
    description: str = ""
    tags: list = field(default_factory=list)
    params: list = field(default_factory=list)   # [Param]
    trials: int = 10
    expect: dict = field(default_factory=dict)    # metric -> (min or None, max or None)
    run: object = None
    source: Path = None                           # the experiment's file
    directory: Path = None                        # its folder (a folder experiment) or the file's folder


_registry = []
_loading = None   # (file, directory) of the experiment being imported
_loaded = set()   # experiment files already imported (discover is idempotent)


def experiments():
    """Every registered experiment, in registration order."""
    return list(_registry)


def experiment(name=None, description="", tags=(), params=None, trials=10, expect=None):
    """Decorator registering run(trial) as an experiment (see the module docs)."""
    def register(run):
        source, directory = _loading or (Path(run.__code__.co_filename), Path(run.__code__.co_filename).parent)
        exp_name = name or (directory.name if source.name == "experiment.py" else source.stem)
        if any(e.name == exp_name for e in _registry):
            raise ValueError(f"nntest: two experiments named '{exp_name}'")
        specs = []
        for key, value in (params or {}).items():
            default, help_text = value if isinstance(value, tuple) else (value, "")
            specs.append(Param(key, default, help_text))
        checks = {}
        for metric, bounds in (expect or {}).items():
            if isinstance(bounds, dict):
                bounds = (bounds.get("min"), bounds.get("max"))
            checks[metric] = tuple(bounds)
        _registry.append(Experiment(exp_name, description, list(tags), specs, trials, checks, run, source,
                                    directory))
        return run
    return register


def discover(root):
    """Imports every Python experiment under `root`: DIR/experiment.py folders and single *.py files."""
    global _loading
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no experiments directory {root}")
    for entry in sorted(root.iterdir()):
        if entry.name.startswith(("_", ".")):
            continue
        if entry.is_dir() and (entry / "experiment.py").exists():
            source, directory = entry / "experiment.py", entry
        elif entry.is_file() and entry.suffix == ".py":
            source, directory = entry, root
        else:
            continue
        if source.resolve() in _loaded:
            continue
        _loaded.add(source.resolve())
        module_name = f"nntest_experiment_{entry.stem}"
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        added = str(directory) not in sys.path
        if added:
            sys.path.insert(0, str(directory))  # a folder's own helper modules import by name
        _loading = (source, directory)
        try:
            spec.loader.exec_module(module)
        finally:
            _loading = None
            if added and entry.is_file():
                sys.path.remove(str(directory))


# --- Trials ------------------------------------------------------------------------


class Trial:
    """One seeded run of an experiment."""

    def __init__(self, index, seed, params, experiment_, log, data_dir=None):
        self.index = index
        self.seed = seed
        self.params = params
        self.experiment = experiment_
        self.rng = np.random.default_rng(seed)
        self.metrics = {}
        self._log = log
        self._data_dir = data_dir

    def record(self, metric, value):
        """Records one value of a metric; recorded again, the last value counts."""
        self.metrics[metric] = float(value)

    def log(self, *parts):
        """Diagnostics, shown with --verbose."""
        if self._log:
            print(*parts, file=self._log)

    @property
    def directory(self):
        """The experiment's folder (for its own files)."""
        return self.experiment.directory

    def dataset(self, name, split="train", **options):
        """datasets.load(name, split, ...) from the runner's data directory."""
        options.setdefault("data_dir", self._data_dir)
        return datasets.load(name, split, **options)

    @staticmethod
    def best_ms(repeats, iterations, f):
        """Milliseconds per call of f(), best of `repeats` runs of `iterations` calls."""
        best = math.inf
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                f()
            best = min(best, (time.perf_counter() - start) * 1000 / iterations)
        return best


def summarize(values):
    """n, mean, sd (n - 1), stderr, min, max, as the C++ runner."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "stderr": None, "min": None, "max": None}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "stderr": sd / math.sqrt(n) if n > 1 else 0.0,
            "min": min(values), "max": max(values)}


def json_number(value):
    return value if value is not None and math.isfinite(value) else None


# --- Environment -----------------------------------------------------------------


def cpu_model():
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def git_state(directory):
    """(12-character commit, dirty) of the repository containing `directory`."""
    def git(*args):
        return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, check=True).stdout
    try:
        return git("rev-parse", "--short=12", "HEAD").strip(), bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def capture_environment(directory):
    info = _core.build_info()
    commit, dirty = git_state(directory)
    now = time.gmtime()
    return {
        "run": time.strftime("%Y%m%dT%H%M%SZ", now) + f"-{random.getrandbits(16):04x}",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
        "host": socket.gethostname(),
        "cpu": cpu_model(),
        "compiler": f"{info['compiler']}, python {platform.python_version()}",
        "build": info["build"],
        "native": info["native"],
        "openmp": info["openmp"],
        "threads": _core.threads(),
        "git": commit,
        "dirty": dirty,
        "runner": "python",
    }


# --- Commands ----------------------------------------------------------------------


class UsageError(Exception):
    pass


def select(selectors):
    chosen = []
    for selector in selectors:
        matched = [e for e in _registry
                   if selector == "all"
                   or (selector.startswith("tag:") and selector[4:] in e.tags)
                   or (not selector.startswith("tag:") and fnmatch.fnmatchcase(e.name, selector))]
        if not matched:
            raise UsageError(f"no experiment matches '{selector}' (see list)")
        chosen += [e for e in matched if e not in chosen]
    return chosen


def grid(e, sets):
    """Every combination of the --set values for the parameters `e` declares, as text."""
    axes = [[to_text(p.default)] if p.name not in sets else sets[p.name] for p in e.params]
    return [dict(zip([p.name for p in e.params], combo)) for combo in itertools.product(*axes)]


def run_command(args):
    if args.threads:
        _core.set_threads(args.threads)
    chosen = select(args.selectors)
    sets = {}
    for setting in args.set or []:
        name, eq, values = setting.partition("=")
        if not eq or not name or not values:
            raise UsageError(f"--set needs NAME=VALUE[,VALUE...], got '{setting}'")
        sets[name] = values.split(",")
    for name in sets:
        if not any(p.name == name for e in chosen for p in e.params):
            raise UsageError(f"no selected experiment has a parameter '{name}'")

    env = capture_environment(chosen[0].directory)
    print(f"run {env['run']}  {env['cpu']}, {env['threads']} threads, {env['build']}"
          f"{' native' if env['native'] else ''}, git {env['git']}{' (dirty)' if env['dirty'] else ''}")
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    log = sys.stderr if args.verbose else None
    status = 0
    try:
        for e in chosen:
            trials = args.trials or e.trials
            for config in grid(e, sets):
                defaults = all(config[p.name] == to_text(p.default) for p in e.params)
                described = " ".join(f"{k}={v}" for k, v in config.items()) or "(no parameters)"
                print(f"\n{e.name}  [{described}]  {trials} trials")
                try:
                    params = Params({p.name: p.parse(config[p.name]) for p in e.params})
                except ValueError as err:
                    raise UsageError(str(err)) from None
                values, failed, seconds_total = {}, 0, 0.0
                for i in range(trials):
                    seed = (args.seed + i) & 0xFFFFFFFF
                    _core.reseed(seed)
                    trial = Trial(i, seed, params, e, log, args.data)
                    error = None
                    start = time.perf_counter()
                    try:
                        e.run(trial)
                    except Exception as err:  # noqa: BLE001 - a failing trial is reported, not fatal
                        error = f"{type(err).__name__}: {err}"
                        if args.verbose:
                            traceback.print_exc()
                    seconds = time.perf_counter() - start
                    seconds_total += seconds
                    if error:
                        failed += 1
                        print(f"  trial {i} (seed {seed}) failed: {error}")
                    else:
                        for metric, v in trial.metrics.items():
                            values.setdefault(metric, []).append(v)
                    if out:
                        out.write(json.dumps({
                            "type": "trial", "experiment": e.name, "params": config, "trial": i, "seed": seed,
                            "seconds": seconds, "metrics": {k: json_number(v) for k, v in trial.metrics.items()},
                            "error": error, "env": env}) + "\n")
                if failed:
                    status = 1

                summaries = {m: summarize(v) for m, v in sorted(values.items())}
                width = max([6, *map(len, summaries)])
                print(f"  {'metric':<{width}}{'mean':>12}{'stderr':>11}{'sd':>11}{'min':>11}{'max':>11}")
                for m, s in summaries.items():
                    print(f"  {m:<{width}}" + "".join(f"{s[k]:>{w}.5g}" for k, w in
                                                       (("mean", 12), ("stderr", 11), ("sd", 11), ("min", 11),
                                                        ("max", 11))))
                print(f"  {trials - failed} of {trials} trials ok, {seconds_total:.3g} s")

                checks = []
                if defaults:
                    for metric, (lo, hi) in e.expect.items():
                        s = summaries.get(metric)
                        passed = (s is not None and s["mean"] is not None
                                  and (lo is None or s["mean"] >= lo) and (hi is None or s["mean"] <= hi))
                        if not passed:
                            status = 1
                        bounds = (f" >= {lo:g}" if lo is not None else "") + (f" <= {hi:g}" if hi is not None else "")
                        print(f"  check {metric}{bounds}: {'pass' if passed else 'FAIL'}")
                        checks.append({"metric": metric, "min": lo, "max": hi, "pass": passed})
                if out:
                    out.write(json.dumps({
                        "type": "summary", "experiment": e.name, "params": config, "defaults": defaults,
                        "trials": trials, "failed": failed, "seconds": seconds_total,
                        "metrics": {m: {k: json_number(v) if k != "n" else v for k, v in s.items()}
                                    for m, s in summaries.items()},
                        "checks": checks, "env": env}) + "\n")
                    out.flush()
    finally:
        if out:
            out.close()
    return status


def list_command(args):
    width = max([4, *(len(e.name) for e in _registry)])
    for e in _registry:
        if args.tag and args.tag not in e.tags:
            continue
        kind = "folder" if e.source.name == "experiment.py" else "file"
        print(f"{e.name:<{width + 2}}{e.description}  [{','.join(e.tags)}]  ({kind})")
    return 0


def describe_command(args):
    matches = [e for e in _registry if e.name == args.name]
    if not matches:
        raise UsageError(f"no experiment named '{args.name}' (see list)")
    e = matches[0]
    print(f"{e.name}: {e.description}\nsource: {e.source}\ndefault trials: {e.trials}\nparameters:")
    for p in e.params:
        print(f"  {p.name:<18}{to_text(p.default):<10}{p.help}")
    if e.expect:
        print("checks (default parameters):")
        for metric, (lo, hi) in e.expect.items():
            print(f"  {metric}" + (f" >= {lo:g}" if lo is not None else "") + (f" <= {hi:g}" if hi is not None else ""))
    return 0


def default_experiments_dir():
    if os.environ.get("EXR_EXPERIMENTS"):
        return Path(os.environ["EXR_EXPERIMENTS"])
    nntesting = datasets.find_nntesting()
    return nntesting / "experiments" if nntesting else None


def main(argv=None, experiments_dir=None):
    parser = argparse.ArgumentParser(prog="nntest.py", description="Runs Python nntest experiments.",
                                     epilog="Selectors: NAME, a pattern with '*', tag:TAG, or all.")
    parser.add_argument("--experiments", type=Path,
                        help="experiments directory (default $EXR_EXPERIMENTS or NNtesting/experiments)")
    parser.add_argument("--data", help="datasets directory (default $EXR_DATA or NNtesting/data)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="experiments (optionally only those with a tag)")
    p_list.add_argument("--tag")
    p_describe = sub.add_parser("describe", help="an experiment's parameters, defaults and checks")
    p_describe.add_argument("name")
    p_run = sub.add_parser("run", help="run experiments")
    p_run.add_argument("selectors", nargs="+")
    p_run.add_argument("--trials", type=int, help="trials per parameter combination (default: the experiment's)")
    p_run.add_argument("--seed", type=int, default=0, help="trial i uses seed S + i (default 0)")
    p_run.add_argument("--set", action="append", metavar="NAME=V1,V2,...",
                       help="parameter values; several --set give every combination")
    p_run.add_argument("--threads", type=int, help="OpenMP threads (default: all)")
    p_run.add_argument("--out", help="append every trial and summary to OUT (JSON Lines)")
    p_run.add_argument("--verbose", action="store_true", help="show the experiments' diagnostics")
    args = parser.parse_args(argv)

    try:
        if getattr(args, "trials", None) is not None and args.trials < 1:
            raise UsageError("--trials must be at least 1")
        root = args.experiments or experiments_dir or default_experiments_dir()
        if root is None:
            raise UsageError("no experiments directory: pass --experiments or set EXR_EXPERIMENTS")
        discover(root)
        command = {"list": list_command, "describe": describe_command, "run": run_command}[args.command]
        return command(args)
    except (UsageError, FileNotFoundError) as err:
        print(f"nntest: {err}", file=sys.stderr)
        return 2

