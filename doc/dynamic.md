# Dynamic ladder: time-varying input, up to Doom

Is E-R better when the input changes over time? Earlier temporal tests
([nonlinearity](nonlinearity.md#temporal-tasks-nl_temporal), research log
§16) compared E-R only with networks that have **no memory at all**, so any
state wins there. These experiments add the control that matters: the same
stateless networks shown the last frame as well (a frame window, the
standard trick in game-playing networks). Each task's answer lies in how
the input changes, never in one frame. Results: research log
[§20](research.md#20-dynamic-ladder-time-varying-input-and-doom).

## Models

The networks, initialization, learning (feedback alignment with a learned
bias, the error of every step applied once) and activity meters are those
of `nl_temporal`: `relu`, `er`, `er_memoryless` (E-R reset to rest before
every frame: the ablation), `gate` (fixed threshold 0.2) and `clamp`. Each
frame is held for depth + 1 + settle ticks and the output is read on the
last one, so a model without state sees only the current frame. `window=W`
also gives every model the previous W frames as extra inputs. No term
anywhere counts activity.

## The ladder (`dyn_ladder`)

| Task | What the network sees | Target | Needs |
|------|-----------------------|--------|-------|
| `dir` | a dot on a 16-pixel retina (wrapping), velocity ±1 or ±2 px, redrawn with p = 0.1 per step | 1 if it moves right | order of two frames |
| `change` | one of 4 random 16-pixel binary patterns; a different one appears with p = 0.3 | 1 on the step it changed | comparing two frames |
| `vel` | a Gaussian bump (16-pixel population code) at a continuous position, bouncing off both ends; velocity in ±0.08, redrawn with p = 0.05 | velocity / 0.08 (regression) | a graded difference of two frames |
| `catch` | an 8 × 8 field with a falling ball (dx ∈ {±1, ±2}, bouncing off the side walls) and a one-cell paddle row | the action (−1, 0, +1) of an oracle that knows dx and moves to where the ball will land | the ball's direction, then planning |

`catch` is closed loop: the network's own action is executed and every step
it learns the oracle's action for the state it is in (DAgger-style
imitation). Chasing the ball's current column catches 3.5 %; the oracle
catches every ball.

**Score** (higher is better): accuracy (`dir`, `change`), R² = 1 − NMSE
(`vel`), catch rate (`catch`). Also recorded: `input_ceiling_accuracy` (the
best any function of the shown inputs can do), `chase_catch_rate`,
`oracle_agreement`, learning curves, spikes and cost as in `nl_temporal`.

```bash
nntest run dyn_ladder --threads 4 --trials 5 --set task=dir,change,vel,catch \
    --set model=relu,er,er_memoryless,gate --set window=0,1 \
    --set lr=0.001,0.003,0.01,0.03 --set width=16,64 --out ladder.jsonl
python3 NNtesting/tools/dyn_summary.py ladder.jsonl
```

`dyn_summary.py` picks each configuration's learning rate by median
validation score (never the test set) and prints the median test score,
its range over seeds and the spikes per step.

## Doom (`doom`, Python)

[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) (`pip install
vizdoom`; it runs headless, about 5000 game steps per second here). The
screen (160 × 120 gray) is average-pooled 4× to 40 × 30 = 1200 inputs per
frame, feeding one hidden layer (64) and 3 action outputs. Learning is the
same imitation as `catch`, with one-hot targets. See
[NNtesting/experiments/doom/README.md](../NNtesting/experiments/doom/README.md).

| Scenario | Game | Needs |
|----------|------|-------|
| `basic` | a monster stands on the far wall; strafe left or right, shoot | nothing beyond the current frame |
| `predict_position` | a monster walks sideways; turn left or right, one slow rocket | where the monster will be: its walking direction |

The oracle reads the game's object positions (never shown to the network).
On `predict_position` it leads the shot and wins about 0.6 of episodes;
aiming at where the monster is now wins about 0.13.

```bash
python3 -m pip install vizdoom
NNtesting/nntest.py run doom --trials 3 --set scenario=predict_position,basic \
    --set model=relu,er,gate --set window=0,1 --set lr=0.001,0.003,0.01 --out doom.jsonl
python3 NNtesting/tools/dyn_summary.py doom.jsonl
```

## Doom from reward (`doom_rl`, Python)

The screen at ViZDoom's smallest resolution (pooled to 40 × 30) and stereo
sound through a two-ear cochlea feed a frozen random mix of relu, E-R or
gate neurons and 8 action readouts that learn from reward alone. The
reward comes from the game's state on any map: health lost and death are
penalised, kills, ammo, armor, items, keys, opened doors and leaving the
level rewarded, firing and idling cost a little. Sound needs OpenAL
(`libopenal1`). See
[NNtesting/experiments/doom_rl/README.md](../NNtesting/experiments/doom_rl/README.md).

```bash
NNtesting/nntest.py run doom_rl --trials 3 --set scenario=defend_the_center,map01 \
    --set model=relu,er,gate --set sound=true,false --set train=200 --out doom_rl.jsonl
```
