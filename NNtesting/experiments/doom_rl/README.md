# doom_rl

Doom learned from reward alone, seeing and hearing the game at its smallest
settings. Part of the dynamic ladder ([doc/dynamic.md](../../../doc/dynamic.md)).

```bash
python3 -m pip install vizdoom          # headless
sudo apt-get install libopenal1         # sound: without OpenAL the audio buffer is silent
NNtesting/nntest.py run doom_rl --set scenario=defend_the_center --set model=er,relu --set sound=true,false
```

## Senses and network

```
screen 160 x 120 gray (ViZDoom's smallest) -> 4 x 4 average pool -> "eye" 40 x 30 = 1200 sensors ---\
stereo sound, 11025 Hz, 1260 samples per ear per step -> "mic" -> cochlea (2 ears x 16 mel bands) ---+-> hidden (256)
                                                                                                          |
                                         8 readouts: forward, backward, turn left/right, strafe left/right, shoot, use
```

One game step is 4 tics (1/8.75 s) and 3 network ticks; the sound of the
step is split over the ticks (420 samples each), the screen is held. The
hidden layer is a frozen random mix of relu, E-R or gate neurons (as on
snake, where a frozen mix with learned readouts is the best undesigned
design; `learn_hidden=true` lets it learn too). The largest readout picks
the action; while training a random one with probability `explore`.

## Topology and E-R options

- `depth` hidden layers of `width` neurons (h1 reads the eye and the ears,
  each further layer the one below);
- `feedback`: `recurrent` (every hidden neuron also reads its own layer),
  `topdown` (`feedback_width` extra neurons per layer read the layer
  above), `both`, or `none`; recurrent weights are scaled by
  `recurrent_scale` (0.5: at 0.9 E-R layers burst and then fall silent);
- `reservoir`: an echo-state reservoir after the stack, `reservoir` neurons
  reading the top layer plus `reservoir_recurrent` reading the reservoir;
  the readouts read the top layer and the reservoir;
- `habituation` in fade mode (`habituation_decay` per tick from the
  `habituation_fade_after`th repeat, as in the rerun's fade2) with a
  `habituation_tolerance`: with 0 (exact repeats only) habituation never
  acts here, because the sound and any recurrence change every tick;
- `ticks` per game step (more ticks let E-R run longer on each frame).

Every hidden layer is frozen and only the readouts learn, so the audit's
note on the sign rule and feedback edges (appendix 1) does not apply.

## Search (search.py)

`search.py` looks for the E-R + habituation agent: it searches topology,
E-R, habituation, ticks, pooling and learning settings by asynchronous
successive halving with evolution (rungs of 100, 300 and 900 training
episodes on 1, 2 and 3 seeds; the top third of a rung moves up; new
candidates are half random, half mutations of the best). It appends every
evaluation to `<out>/evals.jsonl`, rewrites `<out>/best.md` and saves the
trained networks of the top rung, and resumes from the same `--out`.

```bash
python3 NNtesting/experiments/doom_rl/search.py --out results/doom-search/defend_the_center --workers 4 --hours 20
```

## Evolution (es.py)

The readout rules did not beat the untrained network (research log §20),
so `es.py` evolves the readouts instead: OpenAI-ES (antithetic pairs,
centred ranks, Adam, step sizes relative to the weights' RMS) scored by
the shaped reward below, every candidate on the same episodes, the current
weights on fixed validation episodes. The hidden layers stay the frozen
random network. It writes `<out>/log.jsonl`, `state.npz` (resumes),
`best.exr` and `best_theta.npy`; `--init` starts from saved weights of the
same network. The best topology found so far is one E-R layer with
fading habituation (tolerance 0.05, decay 0.9 from the 2nd repeat) and no
feedback: the defaults plus `--config '{"depth": 1, "feedback": "none"}'`.

```bash
python3 NNtesting/experiments/doom_rl/es.py --out results/doom-es/dtc --workers 4 --generations 500 \
    --config '{"depth": 1, "feedback": "none"}'
python3 NNtesting/experiments/doom_rl/es.py --scenario map01 --out results/doom-es/map01 \
    --init results/doom-es/dtc/best_theta.npy --config '{"depth": 1, "feedback": "none", "episode_tics": 6300}'
```

## Reward (rewards.py)

Only from the game's state, the same on every map; every weight is a
parameter.

| Event | Default |
|-------|---------|
| health lost | −0.01 per point |
| death | −5 |
| kill | +1 |
| ammo picked up | +0.02 per round |
| ammo fired | −0.001 per round |
| armor gained | +0.01 per point |
| item picked up (ITEMCOUNT) | +0.1 |
| key card or skull key | +2 (the key object disappears) |
| door or lift opened | +0.5, once per door (a closed sector's ceiling rises) |
| leaving the level | +10 (the episode ends alive before the timeout) |
| idle | −0.005 per step while moved < 32 units over the last 20 steps |

## Learning

After each step, the chosen action's readout learns that step's reward
minus a running mean of the reward (`baseline`, 0.01), with the sign rule.
`reward_mode=error` (the default, as on snake) applies it only while the
readout's sign disagrees, so a stream of penalties cannot push every
readout to the output clamp. The first pilot, without the baseline and the
error mode, drove all readouts to −10 and the agent stood still.

## Metrics

Per test episode: `reward` and its parts (`reward_hurt`, `reward_kill`, ...),
`kills`, `damage`, `deaths`, `items`, `keys`, `doors`, `ammo_picked`,
`exits`, `distance` (map units walked), `steps`; `spikes_per_step` and
`active_fraction` of the hidden layer; the same network untrained on the
same games (`*_control`).

Scenarios: `map01` (Freedoom 2 MAP01, a whole level) or any ViZDoom
scenario (`defend_the_center`, `health_gathering`, `my_way_home`, ...);
our reward replaces the scenario's own. Set `exr.set_threads(1)` (the
experiment does): OpenMP on steps this small is 20× slower.
