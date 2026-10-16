#!/usr/bin/env python3
"""The comparison pipeline: neuron models x learning methods on one shared architecture, played in
a game (Doom now; Gardens of Eris later), resumable at every level, with the data to find
bottlenecks. See doc/pipeline.md.

    pipeline.py run     --suite doom [--run NAME] [--neurons er,lif] [--methods es_all,fa_td]
                        [--seeds 1] [--stages train,test,profile] [--slots 8] [--stop-after HOURS]
    pipeline.py status  --suite ... [--run NAME]
    pipeline.py plan    --suite ... [--run NAME]     # GitHub Actions: the neurons with work left, as JSON
    pipeline.py report  --suite ... [--run NAME]     # report.md, results.csv, results.json
    pipeline.py system  --suite ... [--run NAME]     # this machine, once per call

A cell is one neuron model x one learning method x one seed. Each cell runs its stages in order:
train (evolution or online learning), test (fresh games, learning off) and profile (where a game
step's time goes). Every stage writes <stage>.done.json when it finishes, so a run that is
stopped (Ctrl-C, a cancelled job, a reboot) and started again skips what is done; inside a stage,
es.py resumes from its state.npz and online.py from its last checkpoint.

Everything lives under the pipeline home ($EXR_PIPELINE_HOME, default ~/exr-pipeline), in
runs/<run>/ (run defaults to the suite's name). A cell's folder name ends in a hash of its
configuration, so editing the suite starts the changed cells afresh and keeps the old ones.

While stages run, a monitor samples the machine every few seconds into runs/<run>/monitor.jsonl:
CPU busy and iowait, memory available, load, and each running stage's memory (its process group);
each done marker carries its stage's wall time, CPU time (its processes and their children),
peak memory and the monitor's averages over the stage.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # the repository
DOOM = os.path.join(ROOT, "NNtesting", "experiments", "doom_rl")
STAGES = ("train", "test", "profile")


# --- small helpers --------------------------------------------------------------------------------
def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def git_sha():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def merged(*dicts):
    out = {}
    for d in dicts:
        out.update(d or {})
    return out


def csv_list(text, cast=str):
    return [cast(x) for x in text.split(",") if x.strip()] if text else []


# --- suites and cells -----------------------------------------------------------------------------
class Cell:
    def __init__(self, suite, neuron, method, seed, run_dir):
        self.neuron, self.method, self.seed = neuron, method, seed
        self.env = suite["env"]
        self.scenario = suite.get("scenario", "")
        self.params = merged(suite.get("architecture"), suite["neurons"][neuron])
        self.m = dict(suite["methods"][method])
        self.test = suite.get("test", {})
        self.profile = suite.get("profile", {})
        key = json.dumps({"env": self.env, "scenario": self.scenario, "params": self.params, "method": self.m,
                          "seed": seed, "test": self.test, "profile": self.profile}, sort_keys=True)
        self.hash = hashlib.sha1(key.encode()).hexdigest()[:8]
        self.name = f"{neuron}.{method}.s{seed}"
        self.dir = os.path.join(run_dir, "cells", f"{self.name}.{self.hash}")

    def marker(self, stage):
        return os.path.join(self.dir, f"{stage}.done.json")

    def done(self, stage):
        return os.path.exists(self.marker(stage))

    def next_stage(self, stages):
        for s in STAGES:
            if s in stages and not self.done(s):
                return s
            if not self.done(s):
                return None  # an earlier stage that this run does not do is missing
        return None


def load_suite(path):
    with open(path) as f:
        suite = json.load(f)
    if suite.get("env") not in ENVS:
        raise SystemExit(f"suite env must be one of {', '.join(ENVS)} (got {suite.get('env')!r})")
    return suite


def cells(suite, run_dir, neurons=(), methods=(), seeds=()):
    out = []
    for n in neurons or suite["neurons"]:
        if n not in suite["neurons"]:
            raise SystemExit(f"neuron {n!r} is not in the suite ({', '.join(suite['neurons'])})")
        for m in methods or suite["methods"]:
            if m not in suite["methods"]:
                raise SystemExit(f"method {m!r} is not in the suite ({', '.join(suite['methods'])})")
            if m in suite.get("skip", {}).get(n, []):
                continue  # combinations the suite leaves out
            for s in seeds or suite.get("seeds", [1]):
                out.append(Cell(suite, n, m, s, run_dir))
    return out


# --- the Doom adapter -----------------------------------------------------------------------------
class Doom:
    """Stage commands for doom_rl: es.py (kind es), online.py train (kind online), online.py frozen."""

    def __init__(self, slots):
        self.slots = slots

    def train(self, cell):
        out = os.path.join(cell.dir, "train")
        m, py = cell.m, sys.executable
        kind = m["kind"]
        if kind == "frozen":
            return [py, os.path.join(DOOM, "online.py"), "frozen", "--out", out, "--config", json.dumps(cell.params),
                    "--scenario", cell.scenario, "--net-seed", str(cell.seed)], 1
        if kind == "online":
            return [py, os.path.join(DOOM, "online.py"), "train", "--out", out, "--config", json.dumps(cell.params),
                    "--scenario", cell.scenario, "--net-seed", str(cell.seed), "--seed", str(cell.seed),
                    "--method", m["method"], "--rate", str(m.get("rate", 0.0)),
                    "--settings", json.dumps(m.get("settings", {})), "--episodes", str(m.get("episodes", 600)),
                    "--checkpoint", str(m.get("checkpoint", 10))], 1
        if kind == "es":
            pairs = m.get("pairs", 12)
            workers = min(self.slots, m.get("workers") or 2 * pairs + 1)
            config = dict(cell.params, evolve=m.get("evolve", "readout"))
            return [py, os.path.join(DOOM, "es.py"), "--out", out, "--scenario", cell.scenario,
                    "--config", json.dumps(config), "--net-seed", str(cell.seed), "--seed", str(cell.seed),
                    "--pairs", str(pairs), "--sigma", str(m.get("sigma", 0.1)), "--lr", str(m.get("lr", 0.03)),
                    "--episodes", str(m.get("episodes", 2)), "--validation", str(m.get("validation", 10)),
                    "--generations", str(m.get("generations", 200)), "--workers", str(workers)], workers
        raise SystemExit(f"{cell.name}: unknown method kind {kind!r}")

    def after_train(self, cell):
        """es.py leaves no agent.json: write it (online.py writes its own)."""
        out = os.path.join(cell.dir, "train")
        if cell.m["kind"] == "es":
            config = read_json(os.path.join(out, "config.json"))
            write_json(os.path.join(out, "agent.json"), {"kind": "es", "params": config["params"],
                                                         "net_seed": cell.seed, "theta": "best_theta.npy"})
        return self.train_summary(cell)

    def train_summary(self, cell):
        out = os.path.join(cell.dir, "train")
        lines = []
        try:
            with open(os.path.join(out, "log.jsonl")) as f:
                lines = [json.loads(x) for x in f if x.strip()]
        except OSError:
            pass
        if not lines:
            return {}
        if cell.m["kind"] == "es":
            m = cell.m
            per_gen = 2 * m.get("pairs", 12) * m.get("episodes", 2) + m.get("validation", 10)
            last = lines[-20:]
            return {"generations": len(lines), "episodes": len(lines) * per_gen,
                    "best_validation": lines[-1]["best_validation"],
                    "final_validation": sum(x["validation_reward"] for x in last) / len(last),
                    "first_validation": lines[0]["validation_reward"],
                    "seconds_per_generation": sum(x["seconds"] for x in lines) / len(lines)}
        last = lines[-50:]
        return {"episodes": len(lines), "steps": sum(x["steps"] for x in lines),
                "final_train_reward": sum(x["reward"] for x in last) / len(last),
                "first_train_reward": sum(x["reward"] for x in lines[:50]) / len(lines[:50]),
                "seconds_per_episode": sum(x["seconds"] for x in lines) / len(lines),
                "updates": lines[-1].get("updates", 0)}

    def progress(self, cell):
        log = os.path.join(cell.dir, "train", "log.jsonl")
        try:
            with open(log) as f:
                n = sum(1 for x in f if x.strip())
        except OSError:
            return ""
        total = cell.m.get("generations", 200) if cell.m["kind"] == "es" else cell.m.get("episodes", 600)
        return f"{n}/{total} {'gens' if cell.m['kind'] == 'es' else 'episodes'}"

    def test_cmd(self, cell):
        workers = min(self.slots, cell.test.get("workers", 4))
        return [sys.executable, os.path.join(DOOM, "online.py"), "test", "--agent", os.path.join(cell.dir, "train"),
                "--out", os.path.join(cell.dir, "test.json"), "--episodes", str(cell.test.get("episodes", 30)),
                "--workers", str(workers)], workers

    def profile_cmd(self, cell):
        return [sys.executable, os.path.join(DOOM, "online.py"), "profile", "--agent",
                os.path.join(cell.dir, "train"), "--out", os.path.join(cell.dir, "profile.json"),
                "--steps", str(cell.profile.get("steps", 2000))], 1

    def command(self, cell, stage):
        return {"train": self.train, "test": self.test_cmd, "profile": self.profile_cmd}[stage](cell)

    def summary(self, cell, stage):
        if stage == "train":
            return self.after_train(cell)
        return read_json(os.path.join(cell.dir, f"{stage}.json"), {})


ENVS = {"doom": Doom}  # "goe" next: the same stages over NNtesting/experiments/goe_rl


# --- the machine monitor --------------------------------------------------------------------------
def _cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle, iowait = v[3], v[4] if len(v) > 4 else 0
    return sum(v), idle, iowait


def _meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


def _group_rss(pgid):
    """Resident memory of every process in a process group, bytes."""
    total = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.getpgid(int(pid)) != pgid:
                continue
            with open(f"/proc/{pid}/statm") as f:
                total += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    return total


class Monitor(threading.Thread):
    """Samples the machine every `interval` seconds while the run lasts."""

    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self.path, self.interval = path, interval
        self.running = {}  # task label -> pgid
        self.samples = []  # (time, cpu busy, iowait, mem available)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.linux = os.path.exists("/proc/stat")

    def run(self):
        if not self.linux:
            return
        prev = _cpu_times()
        while not self.stop_event.wait(self.interval):
            now = _cpu_times()
            dt = max(now[0] - prev[0], 1)
            busy = 1.0 - (now[1] - prev[1] + now[2] - prev[2]) / dt
            iowait = (now[2] - prev[2]) / dt
            prev = now
            mem = _meminfo()
            with self.lock:
                tasks = dict(self.running)
            rec = {"t": time.time(), "cpu_busy": round(busy, 4), "iowait": round(iowait, 4),
                   "mem_available_mb": round(mem.get("MemAvailable", 0) / 2 ** 20),
                   "mem_total_mb": round(mem.get("MemTotal", 0) / 2 ** 20),
                   "load1": os.getloadavg()[0],
                   "tasks": {k: round(_group_rss(g) / 2 ** 20) for k, g in tasks.items()}}
            with self.lock:
                self.samples.append((rec["t"], busy, iowait, rec["mem_available_mb"], rec["tasks"]))
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def summary(self, start, end, label):
        with self.lock:
            s = [x for x in self.samples if start <= x[0] <= end]
        if not s:
            return {}
        rss = [x[4].get(label, 0) for x in s]
        return {"samples": len(s), "machine_cpu_busy_mean": sum(x[1] for x in s) / len(s),
                "machine_cpu_busy_max": max(x[1] for x in s), "machine_iowait_mean": sum(x[2] for x in s) / len(s),
                "machine_mem_available_min_mb": min(x[3] for x in s), "stage_rss_max_mb": max(rss),
                "stage_rss_mean_mb": sum(rss) / len(rss)}


# --- running stages -------------------------------------------------------------------------------
class Runner:
    def __init__(self, run_dir, env, monitor, slots):
        self.run_dir, self.env, self.monitor, self.slots = run_dir, env, monitor, slots
        self.used = 0
        self.procs = {}  # label -> Popen
        self.active = set()  # cells with a stage running (from start() until its thread ends)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.stopping = False
        self.failures = []
        self.sha = git_sha()

    def start(self, cell, stage):
        cmd, need = self.env.command(cell, stage)
        need = max(1, min(need, self.slots))
        os.makedirs(cell.dir, exist_ok=True)
        write_json(os.path.join(cell.dir, "cell.json"), {"neuron": cell.neuron, "method": cell.method,
                                                         "seed": cell.seed, "env": cell.env,
                                                         "scenario": cell.scenario, "params": cell.params,
                                                         "method_config": cell.m, "test": cell.test,
                                                         "profile": cell.profile, "hash": cell.hash})
        with self.cond:
            self.used += need
            self.active.add(cell.name)
        threading.Thread(target=self._run, args=(cell, stage, cmd, need), daemon=True).start()

    def _run(self, cell, stage, cmd, need):
        label = f"{cell.name}:{stage}"
        log_path = os.path.join(cell.dir, f"{stage}.log")
        start = time.time()
        code, ru = -1, None
        try:
            with open(log_path, "a") as log:
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.sha[:12]} {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                        start_new_session=True)  # its own process group: stop and measure it whole
                with self.cond:
                    self.procs[label] = proc
                write_json(os.path.join(self.run_dir, "running", label.replace(":", "@")), {"pgid": proc.pid})
                with self.monitor.lock:
                    self.monitor.running[label] = proc.pid
                _, status, ru = os.wait4(proc.pid, 0)
                proc.returncode = code = os.waitstatus_to_exitcode(status)
        finally:
            end = time.time()
            with self.monitor.lock:
                self.monitor.running.pop(label, None)
            try:
                os.remove(os.path.join(self.run_dir, "running", label.replace(":", "@")))
            except OSError:
                pass
            ok = code == 0 and not self.stopping
            summary = {}
            if ok:
                try:
                    summary = self.env.summary(cell, stage)
                except Exception as e:  # a stage that ran but left nothing readable has failed
                    ok, summary = False, {"error": repr(e)}
            usage = {"wall_seconds": end - start, "slots": need}
            if ru is not None:
                usage.update(user_seconds=ru.ru_utime, system_seconds=ru.ru_stime,
                             max_rss_mb=ru.ru_maxrss / 1024,  # the largest single process, KiB on Linux
                             cpu_utilisation=(ru.ru_utime + ru.ru_stime) / max(end - start, 1e-9) / need,
                             major_faults=ru.ru_majflt, voluntary_switches=ru.ru_nvcsw,
                             involuntary_switches=ru.ru_nivcsw)
            usage.update(self.monitor.summary(start, end, label))
            record = {"cell": cell.name, "stage": stage, "neuron": cell.neuron, "method": cell.method,
                      "seed": cell.seed, "exit": code, "git": self.sha, "started": start, "finished": end,
                      "command": cmd, "usage": usage, "result": summary}
            if ok:
                write_json(cell.marker(stage), record)
                failed = os.path.join(cell.dir, f"{stage}.failed.json")
                if os.path.exists(failed):
                    os.remove(failed)
                print(f"done  {label} in {end - start:.0f} s", flush=True)
            elif not self.stopping:
                try:
                    with open(log_path) as f:
                        record["log_tail"] = f.readlines()[-30:]
                except OSError:
                    pass
                write_json(os.path.join(cell.dir, f"{stage}.failed.json"), record)
                self.failures.append(label)
                print(f"FAIL  {label} (exit {code}), see {log_path}", flush=True)
            with self.cond:
                self.used -= need
                self.procs.pop(label, None)
                self.active.discard(cell.name)
                self.cond.notify_all()

    def stop(self):
        """Stops every running stage (their process groups); what they saved so far stays."""
        self.stopping = True
        with self.cond:
            procs = list(self.procs.values())
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass


def stop_leftovers(run_dir):
    """Stages of an earlier run of this run folder that outlived it (a runner that killed the
    pipeline but not its process groups) would write into the same cells: stop them first."""
    folder = os.path.join(run_dir, "running")
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):
        pgid = (read_json(os.path.join(folder, name)) or {}).get("pgid")
        try:
            with open(f"/proc/{pgid}/cmdline", "rb") as f:
                ours = ROOT.encode() in f.read()  # not a recycled process id
            if ours:
                print(f"stopping a leftover stage {name.replace('@', ':')} (process group {pgid})", flush=True)
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(30):
                    time.sleep(1)
                    os.killpg(pgid, 0)
                os.killpg(pgid, signal.SIGKILL)
        except (OSError, TypeError):
            pass
        os.remove(os.path.join(folder, name))


def run(args, suite, run_dir):
    stages = csv_list(args.stages) or list(STAGES)
    todo = cells(suite, run_dir, csv_list(args.neurons), csv_list(args.methods), csv_list(args.seeds, int))
    slots = args.slots or suite.get("slots") or os.cpu_count() or 1
    env = ENVS[suite["env"]](slots)
    os.makedirs(run_dir, exist_ok=True)
    stop_leftovers(run_dir)
    record_system(run_dir, suite)
    monitor = Monitor(os.path.join(run_dir, "monitor.jsonl"), args.interval)
    monitor.start()
    runner = Runner(run_dir, env, monitor, slots)
    deadline = time.time() + args.stop_after * 3600 if args.stop_after > 0 else None

    def on_signal(signum, frame):
        print(f"signal {signum}: stopping the running stages; run again to continue", flush=True)
        runner.stop()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    tried = set()  # (cell, stage) that failed in this run: not again until the next run
    print(f"run {os.path.basename(run_dir)}: {len(todo)} cells, stages {','.join(stages)}, {slots} slots", flush=True)
    while not runner.stopping:
        launched = False
        with runner.cond:
            busy = set(runner.active)
        tried |= {tuple(f.split(":")) for f in runner.failures}
        pending = [(c, c.next_stage(stages)) for c in todo if c.name not in busy]
        pending = [(c, s) for c, s in pending if s is not None and (c.name, s) not in tried]
        if not pending and not busy:
            break
        if deadline is None or time.time() < deadline:
            for c, s in pending:
                _, need = env.command(c, s)
                need = max(1, min(need, slots))
                with runner.cond:
                    free = slots - runner.used
                if need <= free or (runner.used == 0):
                    print(f"start {c.name}:{s} ({need} slots)", flush=True)
                    runner.start(c, s)
                    launched = True
        elif not busy:
            print("stop-after reached", flush=True)
            break
        if not launched:
            with runner.cond:
                runner.cond.wait(timeout=5)
    while True:  # stopping: wait for the stages to end
        with runner.cond:
            if not runner.active:
                break
            runner.cond.wait(timeout=5)
    monitor.stop_event.set()
    if runner.stopping:
        return 130
    report(suite, run_dir)
    left = [c.name for c in todo if c.next_stage(stages) is not None]
    if runner.failures:
        print(f"{len(runner.failures)} stages failed: {', '.join(runner.failures)}", flush=True)
        return 1
    return 0 if not left or deadline else 1


# --- status, plan, system, report -----------------------------------------------------------------
def status(suite, run_dir):
    env = ENVS[suite["env"]](1)
    rows = []
    for c in cells(suite, run_dir):
        marks = []
        for s in STAGES:
            if c.done(s):
                marks.append(f"{s}:done")
            elif os.path.exists(os.path.join(c.dir, f"{s}.failed.json")):
                marks.append(f"{s}:FAILED")
            else:
                marks.append(f"{s}:-")
        rows.append(f"{c.name:40s} {' '.join(marks):45s} {env.progress(c)}")
    print("\n".join(rows))


def plan(suite, run_dir, stages, neurons=(), methods=(), seeds=()):
    """The neurons that still have work, for a GitHub Actions matrix."""
    left = sorted({c.neuron for c in cells(suite, run_dir, neurons, methods, seeds) if c.next_stage(stages) is not None},
                  key=list(suite["neurons"]).index)
    print(json.dumps(left))


def record_system(run_dir, suite):
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "python": sys.version.split()[0], "cpus": os.cpu_count(),
            "git": git_sha(), "suite": suite.get("name")}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu_model"] = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "")
        info["mem_total_mb"] = round(_meminfo()["MemTotal"] / 2 ** 20)
    except OSError:
        pass
    try:
        info["affinity_cpus"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    for mod in ("numpy", "vizdoom", "exrelaxer"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", getattr(m, "__file__", "?"))
        except Exception as e:  # recorded, not fatal: the stages report their own errors
            info[mod] = f"missing ({e.__class__.__name__})"
    info["disk_free_gb"] = round(shutil.disk_usage(run_dir).free / 2 ** 30, 1)
    with open(os.path.join(run_dir, "system.jsonl"), "a") as f:
        f.write(json.dumps(info) + "\n")
    return info


def _fmt(x, digits=2):
    return "" if x is None else f"{x:.{digits}f}"


def report(suite, run_dir):
    """Collects every done marker into report/: results.json, results.csv and report.md."""
    out_dir = os.path.join(run_dir, "report")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for c in cells(suite, run_dir):
        row = {"cell": c.name, "neuron": c.neuron, "method": c.method, "seed": c.seed}
        for s in STAGES:
            row[s] = read_json(c.marker(s))
        row["failed"] = [s for s in STAGES if os.path.exists(os.path.join(c.dir, f"{s}.failed.json"))]
        rows.append(row)
    write_json(os.path.join(out_dir, "results.json"), rows)

    flat = []
    for r in rows:
        t, tr, pr = (r["test"] or {}).get("result", {}), (r["train"] or {}), (r["profile"] or {}).get("result", {})
        mean, se = t.get("mean", {}), t.get("se", {})
        parts = pr.get("ms_per_step_by_part", {})
        flat.append({"neuron": r["neuron"], "method": r["method"], "seed": r["seed"],
                     "test_reward": mean.get("reward"), "test_reward_se": se.get("reward"),
                     "test_kills": mean.get("kills"), "test_steps": mean.get("steps"),
                     "test_deaths": mean.get("deaths"), "active_fraction": mean.get("active_fraction"),
                     "spikes_per_step": mean.get("spikes_per_step"),
                     "train_episodes": tr.get("result", {}).get("episodes"),
                     "train_wall_s": tr.get("usage", {}).get("wall_seconds"),
                     "train_cpu_util": tr.get("usage", {}).get("cpu_utilisation"),
                     "train_max_rss_mb": tr.get("usage", {}).get("max_rss_mb"),
                     "ms_per_step": pr.get("ms_per_step"), "us_per_tick": pr.get("us_per_tick"),
                     "synapses": pr.get("synapses"),
                     **{f"ms_{k}": v for k, v in parts.items()}})
    keys = []
    for f_ in flat:
        keys += [k for k in f_ if k not in keys]
    with open(os.path.join(out_dir, "results.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for f_ in flat:
            f.write(",".join("" if f_.get(k) is None else str(f_.get(k)) for k in keys) + "\n")

    neurons = [n for n in suite["neurons"] if any(x["neuron"] == n for x in flat)]
    methods = [m for m in suite["methods"] if any(x["method"] == m for x in flat)]

    def cell_mean(n, m, key):
        v = [x[key] for x in flat if x["neuron"] == n and x["method"] == m and x.get(key) is not None]
        return sum(v) / len(v) if v else None, len(v)

    def table(key, title, digits=2, se_key=None):
        lines = [f"### {title}", "", "| neuron | " + " | ".join(methods) + " |",
                 "|---|" + "---|" * len(methods)]
        for n in neurons:
            cells_ = []
            for m in methods:
                v, k = cell_mean(n, m, key)
                if v is None:
                    cells_.append("")
                    continue
                text = _fmt(v, digits)
                if se_key and k == 1:
                    se, _ = cell_mean(n, m, se_key)
                    text += f" ± {_fmt(se, digits)}"
                elif k > 1:
                    text += f" ({k} seeds)"
                cells_.append(text)
            lines.append(f"| {n} | " + " | ".join(cells_) + " |")
        return lines + [""]

    md = [f"# {suite.get('name', 'Comparison')}: {os.path.basename(run_dir)}", "",
          f"Environment {suite['env']} / {suite.get('scenario', '')}; "
          f"{sum(1 for r in rows if r['test'])} of {len(rows)} cells tested, "
          f"generated {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    shas = sorted({(r[s] or {}).get("git", "")[:12] for r in rows for s in STAGES if r[s]})
    if len(shas) > 1:
        md += [f"Note: stages ran on {len(shas)} commits ({', '.join(shas)}).", ""]
    md += ["## Results", ""]
    md += table("test_reward", "Test reward (fresh games, learning off; mean ± standard error)", 2, "test_reward_se")
    md += table("test_kills", "Kills per game", 2)
    md += table("test_steps", "Steps survived per game", 0)
    md += table("active_fraction", "Active fraction of hidden neurons per tick", 3)
    md += ["## Cost", ""]
    md += table("train_wall_s", "Training wall time (s)", 0)
    md += table("train_cpu_util", "Training CPU use per slot (1.0 = every slot busy)", 2)
    md += table("ms_per_step", "Profiled ms per game step", 2)
    md += bottlenecks(flat, run_dir)
    fails = [f"{r['cell']}:{s}" for r in rows for s in r["failed"]]
    if fails:
        md += ["## Failed stages", "", *[f"- {f}" for f in fails], ""]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))
    return os.path.join(out_dir, "report.md")


EXPORT = ("cell.json", "train.done.json", "test.done.json", "profile.done.json", "train.failed.json",
          "test.failed.json", "profile.failed.json", "test.json", "profile.json", "train/log.jsonl",
          "train/agent.json", "train/config.json", "train/best_theta.npy", "train/best.json", "train/final.exr")


def export(suite, run_dir, dest):
    """The report, the machine's records and every cell's results and agents (not the resumable
    training state) copied to `dest`, e.g. for a GitHub Actions artifact."""
    os.makedirs(dest, exist_ok=True)
    for name in ("report", "system.jsonl", "monitor.jsonl"):
        src = os.path.join(run_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, name), dirs_exist_ok=True)
        elif os.path.exists(src):
            shutil.copy2(src, dest)
    for c in cells(suite, run_dir):
        for name in EXPORT:
            src = os.path.join(c.dir, name)
            if os.path.exists(src):
                to = os.path.join(dest, "cells", os.path.basename(c.dir), name)
                os.makedirs(os.path.dirname(to), exist_ok=True)
                shutil.copy2(src, to)


def bottlenecks(flat, run_dir):
    """Where the time goes: the profile's parts per neuron model, and the machine over the run."""
    md = ["## Bottlenecks", ""]
    prof = [x for x in flat if x.get("ms_per_step")]
    if prof:
        parts = []
        for x in prof:
            parts += [k for k in x if k.startswith("ms_") and k != "ms_per_step" and k not in parts]
        md += ["### Time per game step by part (ms, mean over profiled cells)", "",
               "| neuron | method | " + " | ".join(p[3:] for p in parts) + " | total | µs/tick | synapses |",
               "|---|---|" + "---|" * (len(parts) + 3)]
        seen = {}
        for x in prof:
            seen.setdefault((x["neuron"], x["method"]), []).append(x)
        for (n, m), xs in seen.items():
            avg = lambda k: sum(x.get(k) or 0 for x in xs) / len(xs)
            md.append(f"| {n} | {m} | " + " | ".join(_fmt(avg(p)) for p in parts) +
                      f" | {_fmt(avg('ms_per_step'))} | {_fmt(avg('us_per_tick'), 1)} | {int(avg('synapses'))} |")
        md.append("")
        tot = sum(x["ms_per_step"] for x in prof)
        shares = {p[3:]: sum(x.get(p) or 0 for x in prof) / tot for p in parts}
        top = max(shares, key=shares.get)
        md += [f"Largest share over all profiled cells: **{top}** ({shares[top]:.0%}); "
               + ", ".join(f"{k} {v:.0%}" for k, v in sorted(shares.items(), key=lambda kv: -kv[1])) + ".", ""]
    samples = []
    try:
        with open(os.path.join(run_dir, "monitor.jsonl")) as f:
            samples = [json.loads(x) for x in f if x.strip()]
    except OSError:
        pass
    if samples:
        busy = sum(s["cpu_busy"] for s in samples) / len(samples)
        io = sum(s["iowait"] for s in samples) / len(samples)
        mem = min(s["mem_available_mb"] for s in samples)
        rss = max((sum(s["tasks"].values()) for s in samples), default=0)
        md += ["### The machine while stages ran", "",
               f"- CPU busy {busy:.0%} on average (iowait {io:.1%}) over {len(samples)} samples",
               f"- memory available never below {mem} MB; all stages together at most {rss} MB", ""]
        if busy < 0.6:
            md.append("- CPUs often idle: more slots, or more ES pairs/workers, would use the machine better.")
        if io > 0.05:
            md.append("- Noticeable iowait: the disk (checkpoints, logs) slows the stages.")
        md.append("")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "status", "plan", "report", "system"))
    ap.add_argument("--suite", required=True, help="a suite: its name in suites/ (doom, smoke) or a JSON file")
    ap.add_argument("--run", default="", help="run name (default: the suite's name)")
    ap.add_argument("--home", default=os.environ.get("EXR_PIPELINE_HOME") or os.path.expanduser("~/exr-pipeline"))
    ap.add_argument("--neurons", default="", help="only these neuron models (comma-separated)")
    ap.add_argument("--methods", default="", help="only these methods")
    ap.add_argument("--seeds", default="", help="only these seeds")
    ap.add_argument("--stages", default="", help="only these stages (train,test,profile)")
    ap.add_argument("--slots", type=int, default=0, help="CPU slots for stages (default: the suite's, or all CPUs)")
    ap.add_argument("--stop-after", type=float, default=0.0, help="start no stage after this many hours")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between machine samples")
    ap.add_argument("--export", default="", help="report: also copy the results (not training state) here")
    args = ap.parse_args()
    path = next((p for p in (args.suite, os.path.join(HERE, args.suite),
                             os.path.join(HERE, "suites", args.suite + ".json")) if os.path.isfile(p)), None)
    if path is None:
        raise SystemExit(f"no suite {args.suite!r} (a JSON file, or a name in NNtesting/pipeline/suites)")
    suite = load_suite(path)
    run_dir = os.path.join(args.home, "runs", args.run or suite.get("name", "run"))
    os.makedirs(run_dir, exist_ok=True)
    if args.command == "run":
        sys.exit(run(args, suite, run_dir))
    if args.command == "status":
        status(suite, run_dir)
    elif args.command == "plan":
        plan(suite, run_dir, csv_list(args.stages) or list(STAGES), csv_list(args.neurons), csv_list(args.methods),
             csv_list(args.seeds, int))
    elif args.command == "report":
        print(report(suite, run_dir))
        if args.export:
            export(suite, run_dir, args.export)
    elif args.command == "system":
        print(json.dumps(record_system(run_dir, suite), indent=1))


if __name__ == "__main__":
    main()
