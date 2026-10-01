# doom_rl

Doom learned from reward alone, seeing and hearing the game at its smallest
settings. Part of the dynamic ladder ([doc/dynamic.md](../../../doc/dynamic.md));
every experiment and result is recorded in [doc/doom.md](../../../doc/doom.md).

```bash
./build.sh --vizdoom                    # all in one: ViZDoom from PyPI + the Python package (use a venv)
# or by hand:
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

## Watching an agent (watch.py)

```bash
python3 NNtesting/experiments/doom_rl/watch.py                    # replay.mp4 + .srt + .html: the best agent, 1 game, 3 minutes
python3 NNtesting/experiments/doom_rl/watch.py --agent <es.py output folder> --games 3 --minutes 5 --out games.html
python3 NNtesting/experiments/doom_rl/watch.py --live --speed 0.5  # ViZDoom's own window, needs a display
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
`--seed 4242` (the default) starts the same games as the fresh tests.

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
- the **feedback ladder** (`feedback_first` > 0): every hidden layer above
  h1 comes back to h1 as input through a rung of `feedback_first` neurons
  that read it, one tick late. A signal that climbs to h<k> returns after a
  loop through k layers, so the rungs remember over different spans.
  `feedback_first_from` `top` keeps only the top rung. `feedback_first_model`
  chooses the rung neurons: `h1` (default) adds them to h1 itself, of h1's
  kind, and h2 and the readouts read them; `relu`, `er`, `gate` or `clamp`
  makes each rung a layer of its own, `r<k>`, with that model (and
  habituation unless `feedback_first_habituation` is false), which every h1
  neuron reads next to the eye and the ears; e.g. relu rungs with
  habituation do not fatigue as E-R does. The ladder grows with
  `es.py --grow-to` (each new layer brings its rung; the grown network
  plays as before, up to float rounding).

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
same network. With `evolve` `all` (in `--config`) the hidden layers'
weights evolve too, each layer's steps relative to its own weight RMS;
`--init` then also accepts evolved readouts alone. `--grow-to N` (with
`evolve` `all`) grows the network during the run: it starts at `depth`
and adds a hidden layer on top, up to N, as soon as the current network
learns (the mean validation reward of the last `--grow-window`
generations beats the first window at this depth by `--grow-margin`).
The readouts then read every layer (`readout_from` `all`), and their
weights from a new layer start at zero, so growing does not change play. The best topology found so far is one E-R layer with
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
