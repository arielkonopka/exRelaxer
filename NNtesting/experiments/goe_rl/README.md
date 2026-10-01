# goe_rl

[Gardens of Eris](https://github.com/arielkonopka/Gardens-of-Eris) played by an
E-R network, as doom_rl plays Doom. Every experiment, the designed networks
and how to run them are in [doc/goe.md](../../../doc/goe.md). The game is a grid maze that keeps growing
around the player; its headless build and the Python package `goe` live in
the game's repository (`agent/`, `agent/python`).

```bash
./build.sh --goe                                                      # all in one: clones the game, installs goe (use a venv)
# or by hand:
git clone https://github.com/arielkonopka/Gardens-of-Eris
sudo apt-get install liballegro5-dev libopenal-dev libsndfile1-dev   # the game's libraries
python3 -m pip install ./Gardens-of-Eris/agent/python                  # the goe package (builds the game; events need PR #289)
NNtesting/nntest.py run goe_rl --set model=er,relu                    # untrained network vs a random player
python3 NNtesting/experiments/goe_rl/es.py --out results/goe-es/er_d1 --workers 4 --generations 300
python3 NNtesting/experiments/goe_rl/es.py --out results/goe-es/er_reservoir --config NNtesting/experiments/goe_rl/models/er_reservoir.json
```

## Senses and network

```
vision: (2r + 1)^2 cells around the player (r = 6: 13 x 13), 8 channels -> "eye" (1352 sensors) --\
body: energy / max energy, ammo, spare avatars -> "body" (3) -----------------------------------+-> h1 (128)
                                                                                                    |
      14 readouts: MOVE, SHOOT, INTERACT x {up, down, left, right}, NEXT_GUN, USE (largest one is played)
```

The grid's radius `radius` is `auto` by default: the player's sight grows
with its steps (2 + ln(steps) / 2 cells), so the grid covers the furthest
sight of an episode from the start (6 for 2 minutes, 7 for 30), and cells
beyond the current sight read zero. `seen_channel` adds a ninth channel
marking the cells in sight.

The channels say what each cell holds, from the game's element features: wall
(walls, brick clusters), free (can be stepped on now), enemy (killable, not
the player), collectible, danger (missiles, bombs, landmines), door (doors
and keys), apple (golden apples), moving. Cells out of the player's sight are
zero in every channel.

One game step is one move (8 game ticks, 50 ticks a second); the network runs
`ticks` (3) ticks on each step's view, which is held. `readout` last (the
default) plays the readouts of the step's last tick, `sum` their sum over
the step.

## Designed networks

Besides the plain stack, the network can have a frozen echo-state
`reservoir` reading h1, `skip` (the top layer reads every layer below), a
feedback ladder (`feedback_first`: rung layers r<k> of their own neuron
model, relu with habituation by default, bring h<k> back to h1) and
`readout_from` `all`. `models/` holds three designed networks:
`er_reservoir`, `er_reservoir_grow` (layers grow on top while it learns,
`grow_to`) and `er_ladder`; see [doc/goe.md](../../../doc/goe.md#designed-networks).

## Reward

The game counts what the player's avatar does (goe's `reward_weights`, Gardens-of-Eris
PR #289), and the reward weighs those events with the `w_*` parameters:

| Event | Weight | |
|-------|-------:|-|
| `collect` | +5 | an item collected (each once an episode) |
| `apple` | +20 | a golden apple collected |
| `use` | +2 | the usable in hand used (a broken apple eaten) |
| `open` | +10 | a door opened (each once an episode) |
| `teleport` | +5 | a trip through a teleporter |
| `kill` | +10 | a monster, drone or puppet master killed by the player's shots or blasts |
| `mine` | +5 | a mine or bomb set off by the player's shots |
| `hurt` | -0.2 | per energy point lost |
| `death` | -50 | per avatar lost, the last one included |
| `score` | +0.1 | per point of the game's score, mostly new cells visited, so exploring pays a little |

`--set reward=score` gives the reward of G1 (doc/goe.md): the game's own
score (+1 per new cell visited, +1 per item collected, + the energy of what
the player kills), minus `w_death` (50) per avatar lost. The maze never ends,
so an episode is `episode_ticks` (6000, 2 minutes, 750 moves) unless the
last avatar dies first.

## es.py

OpenAI-ES on the network's weights, as doom_rl/es.py: antithetic pairs,
centred ranks, Adam, steps relative to each layer's weight RMS. By default
(`evolve` all) the hidden layer evolves with the readouts. Every candidate
plays the same worlds; the current weights play 6 fixed validation worlds
every generation (seed 777777 on). The run resumes from `<out>/state.npz`.

There is one game per process (the game keeps its world in static state),
so the workers are separate processes, as in doom_rl. `--config` takes
JSON or a file of it (`models/*.json`); `grow_to` there (or `--grow-to`)
grows the network a layer at a time, carrying every weight over by the
input it reads (`../_shared/wiring.py`).
