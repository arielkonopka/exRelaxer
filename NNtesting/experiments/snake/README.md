# snake / snake_py

Snake without a screen, played by a network that learns from rewards. The
same experiment exists twice and plays the same games:

- C++: `experiments/snake.cpp` (game: `tasks/snake.hpp`), run with
  `./build/NNtesting/nntest run snake`
- Python: this folder (`game.py`, `experiment.py`), run with
  `NNtesting/nntest.py run snake_py`

With the same parameters and seeds both report the same numbers, because
the game uses its own random generator (SplitMix64) and the network is the
same C++ library.

## The game

A 10 × 10 field inside walls. The snake starts with 3 cells in the middle,
facing a random way. Each step it turns left, goes straight or turns right.
An apple makes it one cell longer and a new apple appears on a random free
cell. It dies when it runs into a wall or into itself. The head may move
into the cell the tail is leaving. A game also ends after
`width × height × 2` steps without an apple (starving).

## What the network sees

23 values in [-1, 1], in the snake's own frame (forward is the way the head
points), so each action means the same thing whichever way the snake faces:

| Values | Meaning |
|--------|---------|
| 0–7 | the 8 cells around the head (front-left, front, front-right, left, right, back-left, back, back-right): +1 wall or body, −1 free |
| 8–9 | the direction of the apple: a unit vector (forward, right) |
| 10–17 | 4 points the snake sees: looking forward, left, right and back, the first wall or body cell, as its (x, y) on the map scaled to [−1, 1] |
| 18–21 | the distances to those 4 points as closeness, `2 / d − 1`: 1 when adjacent, towards −1 far away |
| 22 | 1, a bias input (the neurons have no bias of their own) |

## The network and learning

```
state (23 sensors) -> eye (retina, pass-through) -> mix (64 dense, frozen random)
                         \------------------------------+-> left, straight, right (1 neuron each, learned)
```

The snake takes the action whose readout is largest (while training, a
random one with probability `explore`). After the move only that action's
readout learns (`apply_reward_to`), with the reward:

- `eat` (1) for an apple;
- `die` (−1) for a crash;
- otherwise `approach` (0.1) for getting closer to the apple, and −`approach` for moving away.

Learning is error-driven (`reward=error`): a readout learns only when the
sign of its output disagrees with the reward. So each readout learns
whether its action is good in this state. Rewarding every step
(`reward=target`) instead drives every weight to the clamp and plays much
worse (about 2 apples per game).

## Results

Defaults, 200 training games, then 50 test games without learning or
exploration:

| | apples per game | best game |
|---|---|---|
| learned | 13.6 (10.7–16.6 over 10 trials) | 25 |
| the same network, untrained | 0.06 | 1 |

The trained snake finds apples reliably and every test game ends in a
crash (never starving), usually once it is long enough to trap itself. It
only sees its immediate surroundings and the 4 rays, so it cannot plan
around its own body.

Parameters to try: `--set mix=0,64,256`, `--set train=200,1000`,
`--set approach=0,0.1,0.5`, `--set width=15 --set height=15`.

## Other learning rules

`nntest run snake_rules` is the same experiment with a
[learning rule](../../../doc/learning.md) for the readouts (`readout=`) and
for the mixing layer (`mix=`, or `frozen`):

```bash
./build/NNtesting/nntest run snake_rules --set readout=sign --set mix=frozen,sign,trace,fa,perturbation,oja,bcm --set mix_lr=0.0003
./build/NNtesting/nntest run snake_rules --set readout=fa --set lr=0.003 --set bias=true --set mix=frozen,fa
```

It reports apples and steps per game (test and training), apples per 100
steps, and the training cost per step. The sign readouts over a frozen mix
(13.6 apples, 103 steps) are still the best; the results for every rule are
in the [research log](../../../doc/research.md#11-learning-rules).
