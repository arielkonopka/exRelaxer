# doom

Doom from pixels, the top rung of the dynamic ladder
([doc/dynamic.md](../../../doc/dynamic.md)). Needs ViZDoom:

```bash
python3 -m pip install vizdoom      # headless, no display needed
NNtesting/nntest.py run doom --set scenario=predict_position --set model=er,relu --set window=0,1
```

## Scenarios

- `basic`: a monster stands at a random place on the far wall. Buttons:
  move left, move right, shoot. A single frame is enough.
- `predict_position`: a monster walks sideways across the far side of the
  room. Buttons: turn left, turn right, shoot; the agent has one rocket,
  and rockets are slow, so it must aim where the monster will be. Which way
  the monster walks cannot be read from one frame.

## Network and learning

```
screen 160 x 120 gray -> 4 x 4 average pool -> 40 x 30 = 1200 inputs  [+ the last `window` frames]
    -> hidden (64: relu | er | gate | clamp) -> 3 outputs (one per button; the largest is pressed)
```

Each frame (4 game tics) is held for 2 + `settle` network ticks. The
agent's own action is played; every step the outputs learn the oracle's
action as a one-hot target (feedback alignment with a learned bias, as in
`nl_temporal`). No term counts activity.

The **oracle** reads the game's object positions, which the network never
sees. On `basic` it centres the monster on screen and shoots. On
`predict_position` it estimates the monster's velocity from its positions
since the episode started, solves for where a rocket (17 map units per tic)
meets it, turns towards that point (the turn is 7° per step) and shoots
within 4°. It wins about 0.6 of the episodes; the same oracle aiming at the
monster's current position wins about 0.13 (`oracle_no_lead_win_rate`).

## Metrics

`win_rate` (the monster died: `basic` before the timeout, `predict_position`
reward > 0), `reward`, `oracle_agreement` (the share of steps where the
agent chose the oracle's action), `spikes_per_step` and `active_fraction`
(hidden layer, test episodes), `train_win_rate` (all training episodes; the
analysis uses it to choose the learning rate), `oracle_win_rate`,
`oracle_no_lead_win_rate`.

Training and test games use their own seeds (1000 + seed, 5000 + seed).
