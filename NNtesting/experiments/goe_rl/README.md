# goe_rl

[Gardens of Eris](https://github.com/arielkonopka/Gardens-of-Eris) played by an
E-R network, as doom_rl plays Doom. The game is a grid maze that keeps growing
around the player; its headless build and the Python package `goe` live in
the game's repository (`agent/`, `agent/python`).

```bash
git clone https://github.com/arielkonopka/Gardens-of-Eris
sudo apt-get install liballegro5-dev libopenal-dev libsndfile1-dev   # the game's libraries
python3 -m pip install ./Gardens-of-Eris/agent/python                  # the goe package (builds the game)
NNtesting/nntest.py run goe_rl --set model=er,relu                    # untrained network vs a random player
python3 NNtesting/experiments/goe_rl/es.py --out results/goe-es/er_d1 --workers 4 --generations 300
```

## Senses and network

```
vision: (2r + 1)^2 cells around the player (r = 6: 13 x 13), 8 channels -> "eye" (1352 sensors) --\
body: energy / max energy, ammo, spare avatars -> "body" (3) -----------------------------------+-> h1 (128)
                                                                                                    |
      14 readouts: MOVE, SHOOT, INTERACT x {up, down, left, right}, NEXT_GUN, USE (largest one is played)
```

The channels say what each cell holds, from the game's element features: wall
(walls, brick clusters), free (can be stepped on now), enemy (killable, not
the player), collectible, danger (missiles, bombs, landmines), door (doors
and keys), apple (golden apples), moving. Cells out of the player's sight are
zero in every channel.

One game step is one move (8 game ticks, 50 ticks a second); the network runs
`ticks` (3) ticks on each step's view, which is held. `readout` last (the
default) plays the readouts of the step's last tick, `sum` their sum over
the step.

## Reward

The game's own score, gained during the step: +1 for each cell the player
visits for the first time, +1 for each item collected, + the energy of what
the player kills. Each avatar lost costs `w_death` (50). The maze never ends,
so an episode is `episode_ticks` (6000, 2 minutes, 750 moves) unless the
last avatar dies first.

## es.py

OpenAI-ES on the network's weights, as doom_rl/es.py: antithetic pairs,
centred ranks, Adam, steps relative to each layer's weight RMS. By default
(`evolve` all) the hidden layer evolves with the readouts. Every candidate
plays the same worlds; the current weights play 6 fixed validation worlds
every generation (seed 777777 on). The run resumes from `<out>/state.npz`.

There is one game per process (the game keeps its world in static state),
so the workers are separate processes, as in doom_rl.
