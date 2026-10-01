# Doom: how to run, watch and retest

A practical guide to every Doom test in the repository: installing
ViZDoom, watching the best agent, retesting it, training new agents with
other parameters, and using an agent from C++. What the experiments found
is in [doom.md](doom.md); this page is only how to run them.

## What there is

| Tool | Language | What it does |
|------|----------|--------------|
| `doom` experiment | Python | imitation: a small network copies an oracle that aims at a monster (`basic`, `predict_position`) |
| `doom_rl` experiment | Python | reward only: screen + stereo sound → hidden layer → 8 action readouts, trained by a learning rule |
| `doom_rl/es.py` | Python | evolves the weights of a `doom_rl` network (how every good agent so far was made) |
| `doom_rl/search.py` | Python | long search over topology, E-R, habituation and learning settings |
| `doom_rl/retest.py` | Python | retests a search's best configs on fresh seeds |
| `doom_rl/watch.py` | Python | shows a saved agent playing: a video with subtitles and a replay page, or the live game window |
| `examples/doom_agent` | C++ | loads a saved agent (`best.exr`) and runs one game step |

The game itself runs only from Python: ViZDoom is installed as a Python
package, and all game loops are in `NNtesting/experiments/doom*/`. The
network is the same C++ library underneath, so an agent saved by Python
(`best.exr`) loads in C++ unchanged ([C++](#c)). No C++ experiment plays
Doom: the C++ `nntest` has no `doom` experiment.

## Setup

Use a virtual environment outside the repository's own folders (the
system Python on Debian refuses `pip install`):

```bash
python3 -m venv .venv
source .venv/bin/activate
PYTHON="$PWD/.venv/bin/python" ./build.sh --vizdoom
export PYTHONPATH="$PWD/build/EXrelaxer.py/package"
```

`--vizdoom` installs ViZDoom (the official release, from PyPI), nanobind,
pytest and numpy with `pip`, then builds and tests everything including
the Python package. The package stays in `build/`, so every command below
needs the `PYTHONPATH` line in each new shell (or `pip install
./EXrelaxer.py` once, to install it into the venv).

- **Sound**: `sudo apt-get install libopenal1`. Without it the game runs,
  but the audio buffer is silent and the agents hear nothing.
- **Replays**: `sudo apt-get install ffmpeg` (`watch.py` encodes the video
  with it).
- **Threads**: the Doom scripts call `exr.set_threads(1)` themselves; in
  your own code do the same (in C++: `OMP_NUM_THREADS=1`). With OpenMP
  one game step is about 20× slower.
- **Check**: `python -c "import vizdoom, exrelaxer; print(vizdoom.__version__)"`.

All commands run from the repository root.

## Python

### Watch the best agent

```bash
python NNtesting/experiments/doom_rl/watch.py                 # replay.mp4 + .srt + .html: 1 game, 3 minutes
python NNtesting/experiments/doom_rl/watch.py --minutes 0 --games 3 --out long/replay.html   # whole games
python NNtesting/experiments/doom_rl/watch.py --live          # ViZDoom's own window (needs a display)
python NNtesting/experiments/doom_rl/watch.py --live --speed 0.5
```

The replay is three files next to each other: `replay.mp4`, a video of
exactly the 160 × 120 gray screen the network saw (H.264, one frame per
game step, 8.75 a second); `replay.srt`, subtitles with each step's
action, health, ammo, kills, reward and spikes (VLC and mpv load them with
the video); and `replay.html`, a small page that plays the video with the
network beside it: its 8 readouts, which hidden E-R neurons fired at the
decision tick, and step-by-step seeking. Keep the three together. The
video takes about 3–4 KB per step (a 20-minute game: about 30 MB) and the
page about 100 bytes per step; the browser streams the video, so long
games do not fill its memory. `--video mkv` puts the subtitles inside the
video; `--video mpg` writes MPEG-2, which VLC and mpv play but browsers do
not. The sound it heard is not recorded. Needs `ffmpeg`
(`sudo apt-get install ffmpeg`).

| Option | Default | Meaning |
|--------|---------|---------|
| `--agent DIR` | `results/dynamic/doom_agent/untildeath_d1` | a folder written by `es.py` (needs `config.json` and the weights) |
| `--theta FILE` | `best_theta.npy`, else `final_theta.npy` | which weights in that folder |
| `--seed S` | 4242 | game seed; 4242 starts the fresh test games of [doom.md](doom.md) |
| `--games N` | 1 | games to play |
| `--minutes M` | 3 | stop each game after M game minutes (0: the agent's own limit) |
| `--out FILE` | `replay.html` | the replay page; the video and subtitles get the same name (`replay.mp4`, `replay.srt`) |
| `--video F` | `mp4` | `mp4` (H.264), `mkv` (H.264 with the subtitles inside) or `mpg` (MPEG-2: VLC and mpv, not browsers) |
| `--live` | off | show the game window instead of writing a replay |
| `--speed X` | 1 | with `--live`: 1 is real time |

### Retest a saved agent

A saved agent can be retested on other games: other seeds, more games,
other game lengths, another difficulty or scenario. Its network (width,
depth, pooling, sound, ticks) cannot change, because the weights only fit
that network. This is how the numbers in [doom.md](doom.md) were measured:

```python
# retest_agent.py: run from the repository root
import json, sys
import numpy as np
sys.path.insert(0, "NNtesting/experiments/doom_rl")
import es, experiment as e

agent = "results/dynamic/doom_agent/untildeath_d1"
p = json.load(open(agent + "/config.json"))["params"]
p.update(episode_tics=35 * 60 * 5, skill=3)        # what to change: 5-minute games, skill 3
es.init_worker(p, 0)                               # the game, headless, one thread
player = es.build(p, 0, np.load(agent + "/best_theta.npy"))
game = es._worker["game"]
game.set_seed(4242)                                # the fresh test games
stats, parts = e.play(game, player, p, 10, 0.0, 0.0, np.random.default_rng(0), meter=True)
print({k: round(v, 2) for k, v in stats.items()})  # means over the 10 games
print({k: round(v, 2) for k, v in parts.items()})  # the reward, part by part
```

`e.play(game, player, p, games, lr, explore, rng)` with `lr = 0` and
`explore = 0` only plays; the agent's E-R state carries on from one game
to the next, as in the fresh tests. Pass the untrained network with
`es.build(p, 0, None)` for the control. Things worth changing in `p`:

| Key | Meaning |
|-----|---------|
| `scenario` | `defend_the_center` (the agent's own), `health_gathering`, `deadly_corridor`, `my_way_home`, `map01` (a whole level), ... |
| `skill` | 1 (easiest) to 5 |
| `episode_tics` | game length, 35 tics a second (63000 = 30 minutes) |
| `w_*` | reward weights: they change the score only, not the play |

### Quick experiments through the harness

`doom_rl` and `doom` are ordinary `nntest` experiments: every parameter is
set with `--set`, several values run every combination, and each trial
also plays the same games with the untrained network (`*_control`).

```bash
NNtesting/nntest.py describe doom_rl                  # every parameter with its default and help
NNtesting/nntest.py run doom_rl --set scenario=defend_the_center \
    --set model=er,relu --set sound=true,false --trials 3 --out doom_rl.jsonl
NNtesting/nntest.py run doom --set scenario=predict_position --set model=er,relu --set window=0,1
NNtesting/tools/compare.py doom_rl.jsonl --metric reward --metric reward_control --metric kills
```

| Option | Meaning |
|--------|---------|
| `--set NAME=V1,V2` | parameter values; several `--set` give every combination |
| `--trials N` | trials per combination (default 3); trial `i` uses seed `S + i` |
| `--seed S` | first seed (default 0); training games use seed 1000 + trial seed, test games 5000 + trial seed |
| `--out FILE` | append trials and summaries as JSON Lines, for `compare.py` |
| `--verbose` | the experiment's diagnostics |

The main `doom_rl` parameters (the full list is in `describe`):

| Group | Parameters |
|-------|------------|
| game | `scenario` (default `map01`), `skill`, `episode_tics` (2100 = 1 minute) |
| senses | `pool` (4: 40 × 30 screen), `sound`, `bands` (cochlea bands per ear), `ticks` (network ticks per game step) |
| network | `model` (`relu`, `er`, `gate`, `clamp`), `width`, `depth`, `feedback` (`none`, `recurrent`, `topdown`, `both`), `feedback_width`, `reservoir`, `reservoir_recurrent`, `recurrent_scale` |
| feedback ladder | `feedback_first` (rung neurons per layer above h1; 0: none), `feedback_first_from` (`all`, `top`), `feedback_first_model` (`h1`, or `relu`, `er`, `gate`, `clamp` for rung layers of their own), `feedback_first_habituation` |
| E-R, habituation | `habituation`, `habituation_decay`, `habituation_fade_after`, `habituation_tolerance`, `normalize`, `spontaneous_amplitude` |
| learning | `train`, `test` (episodes), `rule` (`sign`, `trace`), `reward_mode`, `lr`, `explore`, `baseline`, `trace`, `learn_hidden`, `readout`, `readout_from` |
| reward | `w_hurt`, `w_death`, `w_kill`, `w_ammo`, `w_fire`, `w_armor`, `w_item`, `w_key`, `w_door`, `w_exit`, `w_idle`, `idle_steps`, `idle_distance` |

A trial with the defaults trains 100 one-minute games, so it takes
minutes. For a smoke test use `--set train=2 --set test=1 --set
episode_tics=350 --trials 1` (about a second). Note that learning rules
alone have not beaten the untrained network on Doom
([research log §20](research.md#20-dynamic-ladder-time-varying-input-and-doom)):
use `es.py` to get an agent that plays.

### Train a new agent with other parameters (es.py)

`es.py` evolves the weights: every candidate plays the same games, the
better half pulls the weights its way, and the current weights play 10
fixed validation games (seed 777777) every generation. Network parameters
go into `--config` as JSON, over the defaults of `es.py` (E-R with fading
habituation, `depth` 2, `width` 128, `feedback` `recurrent`, `ticks` 6,
`pool` 8, sound on, `evolve` `readout`) and the `doom_rl` defaults above.

```bash
# the best agent's recipe, from scratch: 2-minute games first
python NNtesting/experiments/doom_rl/es.py --out results/doom-es/try1 --workers 4 --generations 300 \
    --config '{"depth": 1, "feedback": "none", "evolve": "all", "readout": "last"}'

# then games until death, continuing from those weights (same network)
python NNtesting/experiments/doom_rl/es.py --out results/doom-es/try1_long --workers 4 --generations 300 \
    --sigma 0.05 --lr 0.02 --episodes 3 --init results/doom-es/try1/best_theta.npy \
    --config '{"depth": 1, "feedback": "none", "evolve": "all", "readout": "last", "episode_tics": 63000}'

# something else: a wider network, more ticks, faster habituation
python NNtesting/experiments/doom_rl/es.py --out results/doom-es/try2 --workers 4 --generations 100 \
    --config '{"depth": 1, "feedback": "none", "evolve": "all", "width": 256, "ticks": 8, "habituation_decay": 0.8}'

python NNtesting/experiments/doom_rl/watch.py --agent results/doom-es/try2   # watch the result

# growing layers: start with one, add one on top whenever the network learns, up to 3
python NNtesting/experiments/doom_rl/es.py --out results/doom-es/grow --workers 4 --generations 300 --grow-to 3 \
    --config '{"depth": 1, "feedback": "none", "evolve": "all"}'
```

With `--grow-to N` a layer is added on top when the mean validation reward
of the last `--grow-window` (20) generations beats the first 20 at the
current depth by `--grow-margin` (1.0). The readouts then read every
layer, and their weights from the new layer start at zero, so growing
does not change play ([D10](doom.md#d10-growing-layers-during-evolution-2026-09-28)).
The workers restart with the deeper network, and a resumed run continues
at its depth.

| Option | Default | Meaning |
|--------|---------|---------|
| `--out DIR` | required | results folder; a run with the same `--out` resumes |
| `--scenario` | `defend_the_center` | any scenario, or `map01` |
| `--config JSON` | `{}` | `doom_rl` parameters, plus `evolve`: `readout` (only the 8 readouts) or `all` (every weight) |
| `--generations` | 300 | generations to run in total |
| `--workers` | 4 | parallel game processes (up to your cores) |
| `--pairs` | 12 | antithetic pairs per generation (24 candidates) |
| `--episodes` | 2 | games per candidate |
| `--validation` | 10 | fixed validation games per generation |
| `--sigma`, `--lr` | 0.1, 0.03 | perturbation and Adam step, relative to the weights' RMS |
| `--net-seed` | 0 | seed of the random starting network |
| `--seed` | 1 | seed of the evolution |
| `--init FILE` | | start from `best_theta.npy` of the same network (with `evolve` `all`, readout-only weights also work) |
| `--grow-to N` | 0 | with `evolve` `all`: add hidden layers one at a time, up to N, once the network learns (`--grow-window`, `--grow-margin`) |

Output in `--out`:

| File | Content |
|------|---------|
| `config.json` | the command's arguments and every parameter: what `watch.py` rebuilds the agent from |
| `log.jsonl` | a line per generation: `validation_reward`, `validation_kills`, `population_mean`, `population_best`, `best_validation`, spikes, seconds |
| `best.exr` | the network with the best validation weights, in the library's format (loads in Python and C++) |
| `best_theta.npy`, `best.json` | those weights, and their generation and validation reward |
| `state.npz` | the current weights (`theta`) and optimizer state, for resuming |

Progress while it runs:

```bash
tail -f results/doom-es/try1/log.jsonl
python -c "import json; r=[json.loads(l) for l in open('results/doom-es/try1/log.jsonl')]; \
print([(x['generation'], round(x['validation_reward'], 1)) for x in r[-10:]])"
```

Time: with 4 workers one generation took about 8 s on 2-minute games and
about 100 s once games lasted until death (the agent survives about 21
minutes), so 300 generations are an hour to a day.

### Feedback ladder

The ladder brings every hidden layer above h1 back to h1 as input, one
network tick late. A signal that climbs to h<k> returns to h1 after a
loop through k layers, so the rungs keep the past over different spans:
memory to explore with.

```
              ┌──────────── r3 ◄─────────────────┐
              │ ┌────────── r2 ◄──────┐          │
              ▼ ▼                     │          │
eye, ears ──► h1 ───────────────────► h2 ──────► h3 ──► readouts

r<k>: the rung of h<k>; h1 reads it one tick late
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `feedback_first` | 0 | neurons per rung; 0: no ladder |
| `feedback_first_from` | `all` | `all`: every layer above h1 has a rung; `top`: only the top layer |
| `feedback_first_model` | `h1` | the rung neurons: `h1` adds them to h1 itself (of h1's kind; h2 and the readouts read them with the rest of h1); `relu`, `er`, `gate` or `clamp` makes each rung a layer of its own, `r<k>`, with that model, which every h1 neuron reads next to the eye and the ears |
| `feedback_first_habituation` | true | rung layers habituate, with the `habituation_*` settings |

Rung layers let the loops use other neurons than the stack: relu rungs
with habituation, for example, pass the upper layers' activity back
without the fatigue of E-R neurons, and fade only what repeats. They
make a cycle of forward connections, so the network's update order is
set explicitly (ears, h1 .. h<depth>, rungs, readouts) and saved with it
in `best.exr`.

```bash
# a fixed ladder: 3 layers, relu rungs of 16 with habituation
python NNtesting/experiments/doom_rl/es.py --out results/doom-es/ladder3 --workers 4 --generations 300 \
    --config '{"depth": 3, "feedback": "none", "evolve": "all", "feedback_first": 16, "feedback_first_model": "relu"}'

# a growing ladder: every new layer brings its rung
python NNtesting/experiments/doom_rl/es.py --out results/doom-es/ladder --workers 4 --generations 300 --grow-to 4 \
    --config '{"depth": 1, "feedback": "none", "evolve": "all", "feedback_first": 16, "feedback_first_model": "relu"}'

# a quick look through the harness (rules, not evolution)
NNtesting/nntest.py run doom_rl --set scenario=defend_the_center --set depth=3 \
    --set feedback_first=0,16 --set feedback_first_model=h1,relu
```

When the ladder grows, every weight is carried over by the input it reads
(the network is built one layer at a time, and `Player.sources` records
each weight's input): old weights keep their values, weights from the new
layer and its rung start at zero, and the new neurons start random. The
grown network plays as before up to float rounding: the extra zero-weight
inputs change the order of summation by about 1e-7, and an E-R neuron at
its threshold can tip the other way, so a long game may part after a
while.

Limits: `feedback_first_from` `top` needs `depth` ≥ 2 and does not grow;
the ladder does not combine with a `reservoir`; it combines with every
`feedback` mode (`recurrent`, `topdown`, `both`). Networks without the
ladder are built exactly as before.

### Search and retest (search.py, retest.py)

`search.py` runs for hours: it samples configs (depth, width, feedback,
reservoir, ticks, pooling, habituation and learning settings; E-R with
habituation always on), trains each with the `doom_rl` learning rule on
100, 300 and 900 games (the top third moves up), and mutates the best.
It can be stopped and resumed with the same `--out`.

```bash
python NNtesting/experiments/doom_rl/search.py --out results/doom-search/dtc --workers 4 --hours 20
cat results/doom-search/dtc/best.md                                 # the current ranking
python NNtesting/experiments/doom_rl/retest.py --search results/doom-search/dtc \
    --top 5 --seeds 10,11,12 --test 30 --out retest.jsonl           # the top 5 on fresh seeds
NNtesting/tools/compare.py retest.jsonl --metric reward --metric reward_control
```

`search.py` options: `--scenario`, `--workers`, `--budgets 100,300,900`,
`--test` (test games per evaluation), `--hours` or `--evals` (when to
stop), `--seed`. The search space is `SPACE` at the top of `search.py`.

### Record what you ran

Every Doom run gets an entry in the [log in doom.md](doom.md#log)
(`D13`, `D14`, ...): the question, the command and config, the result, and
where the data is. Small result files go to `results/dynamic/` (gzipped
JSON Lines); a new best agent goes to `results/dynamic/doom_agent/<name>/`
with its `config.json`, `best.exr`, `best_theta.npy` and `best.json`.

## C++

### What a saved agent expects

`best.exr` holds the whole network: wiring, weights and E-R and
habituation state. It loads in C++ with `exr::network::load`, and
`describe` prints it. The best agent (`untildeath_d1`) has:

| | |
|---|---|
| input `eye` | 300 values: the 160 × 120 gray screen averaged in 8 × 8 blocks to 20 × 15, row by row, divided by 255 |
| input `mic` | 420 values per network tick: 210 left-ear samples, then 210 right-ear samples, divided by 32768 |
| one game step | 4 game tics; set `eye` once, then 6 network ticks, each with the next 210 samples per ear (1260 per ear per step) set to `mic`, then `step()` |
| outputs | 8 values after the step's last tick: move forward, move backward, turn left, turn right, strafe left, strafe right, attack, use; press the largest |
| state | carries on from step to step; load the file again for a fresh agent |

Another agent's numbers follow from its `config.json`: `eye` is
(120 / `pool`) × (160 / `pool`), `mic` is 2 × 1260 / `ticks`, and there are
`ticks` network ticks per step.

### Build and run the example

```bash
./build.sh                                  # installs the library into ./install
cmake -S examples/doom_agent -B doom-agent-build -DCMAKE_PREFIX_PATH=$PWD/install
cmake --build doom-agent-build
OMP_NUM_THREADS=1 ./doom-agent-build/doom_agent results/dynamic/doom_agent/untildeath_d1/best.exr
```

It prints the network, runs one step on a gray screen and silence, and
prints the 8 action values and the chosen action. To play, a C++ program
would feed it frames and sound from ViZDoom's C++ API, which needs
ViZDoom built from source ([ViZDoom's build
guide](https://github.com/Farama-Foundation/ViZDoom)); the repository does
not do that.

### The C++ test harness

The C++ experiments (vision, audio, snake, E-R) run with the same commands
as the Python ones. None of them plays Doom:

```bash
./build/NNtesting/nntest list
./build/NNtesting/nntest describe snake
./build/NNtesting/nntest run snake --trials 3 --out snake.jsonl
NNtesting/tools/compare.py snake.jsonl
```

See [NNtesting/README.md](../NNtesting/README.md).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `vizdoom is not installed` / `No module named 'vizdoom'` | `./build.sh --vizdoom`, or `python -m pip install vizdoom` in the venv you run from |
| `No module named 'exrelaxer'` | `export PYTHONPATH=$PWD/build/EXrelaxer.py/package` (after `build.sh --python` or `--vizdoom`) |
| `error: externally-managed-environment` | you are using the system Python: activate the venv |
| the agent hears nothing | install `libopenal1` |
| very slow | `exr.set_threads(1)` in Python, `OMP_NUM_THREADS=1` in C++ |
| `--live` shows nothing or fails | it needs a display; write a replay on a headless machine |
| `watch.py: ffmpeg not found` | `sudo apt-get install ffmpeg` (only replays need it, `--live` does not) |
| the replay page shows no video | keep `replay.html` next to its `replay.mp4`; an `.mpg` replay plays only in VLC or mpv |
| `--init has N weights, this network M` | the weights are from a different network: use the same `--config` (depth, width, feedback, pool, sound, ticks) |
| `es.py` does not exit, or `vizdoom` processes pile up | fixed on 2026-10-01 (the workers now close their games); remove engines left by older runs with `pkill -9 -f vizdoom/vizdoom` |
| `feedback_first does not combine with a reservoir` | use `reservoir` 0 with the ladder |
| `--grow-to needs feedback_first_from all` | a top-only rung cannot grow; use `all` |
| `_vizdoom/` and `_vizdoom.ini` appear | ViZDoom writes them into the current folder; they are safe to delete |
