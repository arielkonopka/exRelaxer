# The comparison pipeline

Neuron models and learning methods compared on one architecture, playing a
game, on a machine of your own: a GitHub Actions workflow
(`.github/workflows/compare.yml`) for a self-hosted runner, around a runner
script that works the same by hand (`NNtesting/pipeline/pipeline.py`). Doom
first; Gardens of Eris is the next environment.

## What it compares

A **suite** (`NNtesting/pipeline/suites/*.json`) names an environment, a
shared architecture, the neuron models, the learning methods and the seeds.
A **cell** is one neuron model x one method x one seed. Every cell of one
seed starts from the same random network (`es.build` reseeds the library
with the seed), so the cells differ only in their neurons and in how they
learn.

`suites/doom.json`: `defend_the_center`, the colour screen pooled 8 x 8
(20 x 15 x 3 = 900 sensors) and the two-ear cochlea, one hidden layer of 128
with fading habituation and no feedback (the best E-R topology of
[doom.md](doom.md)), 8 action readouts, 6 ticks per game step.

| neuron models | |
|---|---|
| `er` | the original E-R neuron |
| `relu` | rectified linear |
| `tanh`, `sigmoid`, `softplus` | rate units ([neuron_models.md](neuron_models.md)) |
| `lif`, `alif` | leaky integrate-and-fire, adaptive |
| `izhikevich` | regular spiking |
| `resonator` | damped oscillator |

| methods | kind | what learns, from what |
|---|---|---|
| `frozen` | none | nothing: the untrained network, the floor |
| `es_readout` | evolution | readouts, by OpenAI-ES on game reward (es.py) |
| `es_all` | evolution | every weight, by ES |
| `sign` | online | readouts, sign rule on the step's reward minus its running mean |
| `eligibility_td` | online | readouts, eligibility traces on a TD critic's error |
| `eprop_td` | online | hidden + readouts, e-prop on the TD error |
| `fa_td` | online | hidden + readouts, feedback alignment (a random-feedback stand-in for backpropagation) |
| `surrogate_td` | online | hidden + readouts, surrogate gradients: truncated backpropagation through time (8 ticks) |
| `perturbation_td` | online | readouts, node perturbation on the TD error |

Online methods learn on every game step while playing (learning always on);
`NNtesting/experiments/doom_rl/online.py` has the details. Their rates are
first guesses (GoE G4's), not tuned for Doom; a rate is one key in the suite
(`"rate": 1e-5`), and the same method twice under two names compares rates.
`"skip": {"izhikevich": ["es_all"]}` leaves combinations out.

Evolution plays far more games than online learning (200 generations x 58
games against 600): the report shows each cell's training games next to its
score.

## Stages

Each cell runs three stages in order:

1. **train**: es.py, `online.py train`, or `online.py frozen`;
2. **test**: 30 fresh games (seeds 4242...), learning off, no exploration,
   each from a fresh network state, in parallel;
3. **profile**: 2000 game steps in one process, timing each part of a step:
   `game` (ViZDoom), `reward` (the shaper), `senses` (screen pooling and
   sound), `network` (the ticks), `learning` (the method's updates).

## Stopping and continuing

Everything lives on the machine, in the pipeline home (`$EXR_PIPELINE_HOME`,
default `~/exr-pipeline`; a repository variable of that name moves it):

```
~/exr-pipeline/
  venv/                         numpy, nanobind, ViZDoom
  build/<commit>-native<ON|OFF> the library and Python package of a commit (the 3 newest kept)
  runs/<run>/
    cells/<neuron>.<method>.s<seed>.<hash>/
      train/                    es.py's or online.py's folder: log.jsonl, state, the agent
      <stage>.done.json         the stage's result, wall time, CPU time, memory, commit
      <stage>.failed.json       the exit code and the log's tail; retried on the next run
      <stage>.log
    monitor.jsonl               the machine every 5 s
    system.jsonl                the machine at every start
    report/                     report.md, results.csv, results.json
```

A stopped run continues at three levels: the workflow skips neurons with
nothing left (re-run it, or "Re-run failed jobs"); a cell skips stages with a
done marker; inside training, es.py resumes from its last generation and
`online.py train` from its last checkpoint (every 10 games; the games after
it are played again, on the same seeds). Files are written whole (a
temporary file, then a rename), so a stop mid-write loses at most one step.

A cell's folder name ends in a hash of its configuration: editing the suite
starts the changed cells afresh and leaves the old ones in place. A new run
name starts everything afresh.

## Running it

On GitHub: **Actions > Compare neurons > Run workflow**. The inputs pick the
suite, the run name, a subset of neurons, methods or seeds, the CPU slots,
and a time limit (`stop_after`, in hours: no new stage starts after it; the
next run continues). Jobs:

- `setup`: the virtual environment and the build of this commit
  (`runner-setup.sh`, once per commit), the machine's description, and which
  neurons still have work;
- `neuron` (one per neuron model, one at a time): its cells, several at
  once on the runner's CPUs (an ES training takes as many slots as workers,
  up to all; online training, test and profile one or a few);
- `report` (also after failures or a cancel): `report.md` on the run's
  summary page, and an artifact `compare-<run>-<n>` with the report, the
  machine's records and every cell's results and agents (not the
  resumable training state).

The runner needs Linux with `build-essential cmake python3-venv python3-dev`
(and `libopenal1` for Doom's sound), the labels `self-hosted` and `Linux`,
and network access for pip on the first run.

By hand, on any Linux machine:

```bash
eval "$(NNtesting/pipeline/runner-setup.sh --native | sed 's/^/export /')"
$PYTHON NNtesting/pipeline/pipeline.py run --suite smoke             # every stage in a few minutes
$PYTHON NNtesting/pipeline/pipeline.py run --suite doom --neurons er,lif --methods frozen,es_all,fa_td
$PYTHON NNtesting/pipeline/pipeline.py status --suite doom
$PYTHON NNtesting/pipeline/pipeline.py report --suite doom
```

Ctrl-C stops the running stages cleanly; the same command continues.

## Bottleneck data

- **Per stage** (`<stage>.done.json`, `usage`): wall time, user and system
  CPU time of the stage and its children, CPU use per slot it held, the
  largest process's peak memory, page faults, context switches, and the
  monitor's averages over the stage (machine CPU busy and iowait, memory
  available, the stage's process group memory).
- **Per step** (`profile.json`): ms per game step by part, µs per network
  tick, neurons and synapses.
- **The machine** (`monitor.jsonl`, `system.jsonl`): CPU busy and iowait,
  load, memory available and each running stage's memory every 5 s; CPU
  model, cores, memory, disk, Python and library versions at every start.

`report.md` ends with a bottleneck section: the profile's parts per cell,
the largest share over all cells, and the machine over the run, with a note
when CPUs sat idle or the disk held stages up.

First smoke run (this container, 4 cores, tiny network of 16): a step took
~1.5 ms, of which ViZDoom 76 %, the senses 10 %, the network 9 %; pooling the
colour screen with `mean()` cost more than the network, so it now sums
integers row-wise (7x faster).
