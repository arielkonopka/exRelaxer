# Doom experiments

The running record of every experiment that plays Doom: what was asked,
how it was set up, what it showed and where its data is. The goal since
2026-09-27 is **an E-R network with habituation that plays Doom**, with its
topology and settings found by a long search; other networks are only
references.

Each experiment is one entry in the [log](#log), newest last, numbered
`D1`, `D2`, ... so that other pages can cite them. The dynamic ladder that
led here is in [dynamic](dynamic.md); the research log summarises the
findings in [§20](research.md#20-dynamic-ladder-time-varying-input-and-doom).

## Current best agent

| | |
|---|---|
| Network | one hidden layer, 128 E-R neurons with fading habituation (tolerance 0.05, decay 0.9 per tick from the 2nd repeat), no feedback; 6 network ticks per game step |
| Senses | screen 160 × 120 gray pooled 8 × 8 to 20 × 15; stereo sound through a two-ear cochlea |
| Trained by | evolution of every weight, the hidden layer's and the 8 action readouts' (`es.py`, `evolve` `all`, [D9](#d9-evolving-every-weight-depth-1-2-and-3-on-defend_the_center-2026-09-28)) |
| `defend_the_center` | 30 fresh games: reward +3.78 (untrained −5.05), 6.7 kills, survives 57% of games, 172 spikes per step |
| `map01` | not yet tried with every weight evolving; readout-only agents explore but never exit ([D6](#d6-evolution-on-a-whole-level-map01-2026-09-28), [D7](#d7-map01-with-three-minute-episodes-2026-09-28)) |
| Files | `results/dynamic/doom_agent/evolve_all_d1/`: `final_theta.npy`, `config.json` |

Rebuild it in Python:

```python
import json, numpy as np, sys
sys.path.insert(0, "NNtesting/experiments/doom_rl")
import es
d = "results/dynamic/doom_agent/evolve_all_d1"
p = json.load(open(d + "/config.json"))["params"]
player = es.build(p, 0, np.load(d + "/final_theta.npy"))   # net seed 0
```

These agents were evolved before the library defaults of 2026-09-28
(raw sums, spontaneous amplitude 0.01). Their `config.json` has no
`normalize` or `spontaneous_amplitude`, so `doom_rl` rebuilds them that
way; new runs record both.

## Setup shared by all entries

- **Game**: [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
  (`python3 -m pip install vizdoom`), headless, about 5000 game steps per
  second. Sound needs `libopenal1`; without it the audio buffer is silent.
- **Speed**: call `exr.set_threads(1)`. With OpenMP, steps this small are
  20× slower.
- **Game step**: 4 tics (1/8.75 s). The screen is held for the step's
  network ticks, and the step's sound is split across them.
- **Actions**: one readout per button; the largest is pressed.
- **Reward** (`doom_rl/rewards.py`, from game state only, the same on
  every map): health lost −0.01 per point, death −5, kill +1, ammo
  picked up +0.02 per round, ammo fired −0.001 per round, armor +0.01 per
  point, item +0.1, key +2, door opened +0.5, level exit +10, idle
  −0.005 per step. Full table in the
  [doom_rl README](../NNtesting/experiments/doom_rl/README.md#reward-rewardspy).
- **Seeds**: every candidate or model is compared on the same episodes
  (`game.set_seed`); fixed validation episodes use seed 777777, fresh
  test episodes other seeds.
- **Raw data**: `results/dynamic/` in the repo; the project's shared
  folder keeps the full runs under `reports/doom-es/`, `reports/doom-search/`
  and `reports/doom_rl/`.

## Log

### D1. Imitation from pixels (2026-09-27)

**Question.** Can a network learn Doom from pixels by imitating an oracle
that reads object positions?

**Setup.** `nntest run doom` ([README](../NNtesting/experiments/doom/README.md)).
`predict_position`: lead a monster walking sideways with one slow rocket.
40 × 30 pixels, 64 hidden neurons (relu, er, gate), feedback alignment
towards the oracle's action, 600 training episodes, 3 seeds.

**Result.** The oracle wins 0.58; aiming at the monster's current position
wins 0.10. No model learns from pixels without a frame window: win rate
0.07 (er), 0.00 (relu, gate); oracle agreement ≤ 0.5.

**Conclusion.** Imitation from raw pixels with a frozen-feature setup does
not work here. Not pursued further.

### D2. Reward with per-step readout rules (2026-09-27)

**Question.** Can the readouts learn from the shaped reward alone?

**Setup.** `nntest run doom_rl`: frozen 256-neuron hidden mix (relu, er or
gate), with and without sound; after each step the chosen action's
readout learns that step's reward. 200 training episodes, 3 seeds.

**Result.**
- First attempt (raw reward, trace rule): the steady penalties drove every
  readout to the −10 clamp and the agent stood still. Error-driven
  learning (`reward_mode=error`) and a running reward baseline fixed that.
- `defend_the_center`: relu and gate without sound learned to keep
  running (11 000–12 000 map units vs ≈ 800 untrained), took 34–39
  damage instead of 100, and survived to the timeout (reward −0.3 vs
  −6.1). No model learned to kill. E-R did not learn to survive. With
  sound, no model learned (3 seeds; may be seed variance).
- `map01`: nothing beyond the untrained network.

**Conclusion.** One-step reward on the chosen action learns survival at
best.

### D3. Topology search with readout rules (2026-09-27 to 28)

**Question.** Which topology and settings make an E-R + habituation agent
learn from reward?

**Setup.** `doom_rl/search.py`: asynchronous successive halving with
evolution over depth, width, feedback (none, recurrent, top-down, both),
reservoir, habituation settings, ticks, pooling, rule and learning
settings. Rungs of 100, 300 and 900 training episodes on 1, 2 and 3
seeds. `defend_the_center`, 1559 evaluations of 723 configurations.

**Result.** Top configurations scored −2.2 to −3.0 at the top rung, with
no consistent winner shape (1–3 layers, with and without feedback and
reservoir).

**Conclusion.** See D4: the ranking was noise.

### D4. Retest of the search winners (2026-09-28)

**Question.** Is the search winners' score learning or luck?

**Setup.** `doom_rl/retest.py`: the top 5 configurations on 3 fresh seeds,
30 test episodes each, trained vs the same network untrained (`train=0`).

**Result.** Trained networks scored below untrained ones: reward −4.97 vs
−4.21 on average; for the best configuration −4.20 vs −2.90. The
untrained E-R networks already kill about 1–3 monsters per episode through
their own activity.

**Conclusion.** The per-step readout rules do not assign credit in Doom;
the search selected lucky seeds. Replaced by evolution (D5).

Data: `results/dynamic/doom_search_retest_top5.jsonl.gz`.

### D5. Evolution of the readouts: topologies on `defend_the_center` (2026-09-28)

**Question.** If credit assignment is skipped by evolving the readouts on
the game reward directly, which topology plays best?

**Setup.** `doom_rl/es.py`: OpenAI-ES on the 8 readouts' weights, 12
antithetic pairs per generation, centred ranks, Adam (σ 0.1 and step 0.03
relative to the weights' RMS). All candidates play the same 3 episodes;
the current weights are scored on 10 fixed validation episodes that the
gradient never sees. Hidden layers frozen (net seed 0). 500 generations
each.

**Result** (validation reward; kills and spikes per step over the last 100
generations):

| Network | First 20 gens | Last 100 gens | Best | Kills | Spikes |
|---------|---------------|---------------|------|-------|--------|
| **E-R + habituation, 1 layer, no feedback** | −4.79 | **+0.95** | **4.74** | 5.1 | **145** |
| E-R + habituation, 2 layers, recurrent | −2.64 | −1.04 | 0.80 | 5.1 | 363 |
| E-R, no habituation, 2 layers, recurrent | −2.49 | −0.83 | 0.48 | 5.3 | 365 |
| E-R + habituation, 3 layers, feedback both ways, 256 reservoir | −4.10 | −2.25 | 0.29 | 3.8 | 796 |
| ReLU, 2 layers, recurrent | −6.11 | −6.11 | −6.11 | 0 | 763 |

**Conclusion.**
- The simplest network is the best and the most frugal.
- Habituation makes no measurable difference in the 2-layer network.
- Depth, feedback and a reservoir make things worse.
- The ReLU network never moves: every candidate plays the same game, so
  the ranks carry no signal.
- Part of every E-R score comes before evolution (the untrained network
  already kills about one monster per episode).

Data: `results/dynamic/es_cmp_*.jsonl.gz`; agent in
`results/dynamic/doom_agent/cmp_er_d1_none/`.

### D6. Evolution on a whole level: `map01` (2026-09-28)

**Question.** Does the D5 agent's recipe work on a full level, and does
starting from the arena agent help?

**Setup.** As D5 (but 2 episodes per candidate, 6 validation episodes) with
the 1-layer E-R + habituation network on Freedoom 2
MAP01, one-minute episodes, 400 generations; one run from scratch, one
from the D5 agent's weights (`--init`). Tested on 12 fresh episodes.

**Result.**

| | Untrained | Evolved from scratch | Evolved from the arena agent |
|---|---|---|---|
| Reward | −2.68 | −1.54 | −1.57 |
| Distance walked | 1287 | 2694 | 2582 |
| Items | 0.75 | 1.83 | 2.08 |
| Doors | 0.42 | 0.75 | 0.75 |
| Kills | 0.08 | 0.17 | 0.25 |
| Exits | 0 | 0 | 0 |

**Conclusion.** Evolution learns to explore but not to fight or finish
the level; the arena weights give no lasting head start. One minute may
be too short to reach the exit.

Data: `results/dynamic/es_map01_*.jsonl.gz`; agent in
`results/dynamic/doom_agent/map01_er_d1_from_dtc/`.

### D7. `map01` with three-minute episodes (2026-09-28)

**Question.** Does the agent reach the exit when episodes are long enough?

**Setup.** As D6, three-minute episodes (`episode_tics` 6300), 600
generations, starting from the D6 warm-started run's final weights.
Tested on the same 12 fresh episodes as D6, three-minute episodes.

**Result.** Validation reward did not move over 600 generations (−5.95 in
the first 100, −5.95 in the last 100; best −3.36). On the fresh episodes:

| | Untrained | D6 weights (start) | After 600 generations |
|---|---|---|---|
| Reward | −7.69 | −6.82 | −5.30 |
| Distance walked | 1526 | 2486 | 3305 |
| Items | 0.17 | 2.17 | 2.33 |
| Doors | 0.42 | 0.75 | 0.67 |
| Kills | 0.00 | 0.08 | 0.75 |
| Deaths | 0.25 | 0.42 | 0.25 |
| Exits | 0 | 0 | 0 |

**Conclusion.** Longer episodes do not bring the exit within reach. The
evolved agent walks farther and kills a little more than its starting
weights on fresh games, but the flat validation curve means this is
within the run's noise. The readouts alone seem to have reached what the
frozen random features allow on a whole level.

Data: `results/dynamic/es_map01_er_d1_long.jsonl.gz`.

### D8. Every weight evolving on `map01` (2026-09-28, interrupted)

**Question.** Do the hidden layer's features limit the whole-level agent
(D6, D7)? The user chose to evolve the hidden E-R layer too.

**Setup.** As D6 with `evolve` `all` (43 520 weights instead of 1 024),
24 antithetic pairs, σ 0.05 and step 0.02 relative to each layer's
weight RMS, starting from D7's evolved readouts.

**Status.** The project's shared folder, where it wrote, failed at
generation 34 and the run died with it. Validation had reached −1.9
(best 0.82). Not resumed: the arena comparison (D9, D10) came first. Data:
`results/dynamic/es_map01_er_d1_all_interrupted.jsonl.gz`.

### D9. Evolving every weight: depth 1, 2 and 3 on `defend_the_center` (2026-09-28)

**Question.** The user: sound should help (monsters grunt), but the
network may be unable to link sound and action; depth might help, and
earlier depth hurt. Train deeper networks, all E-R with habituation and
no feedback lines, with every weight evolving.

**Setup.** `es.py` with `evolve` `all`, 12 antithetic pairs, 3 episodes
per candidate, 10 validation episodes, σ 0.05 and step 0.02 relative to
each layer's weight RMS, 500 generations, one seed each. 128 E-R neurons
per layer with fading habituation, sound on. Final weights tested on 30
fresh games (seed 4242). The runs used the defaults before 2026-09-28.

**Result.**

| Network | Validation, first 20 gens | Last 100 gens | Fresh games: reward | Kills | Deaths | Spikes |
|---------|---------------------------|---------------|---------------------|-------|--------|--------|
| **1 layer** | −5.00 | +2.95 | **+3.78** | 6.7 | 0.43 | **172** |
| 2 layers | −4.02 | +3.02 | +3.18 | 6.1 | 0.47 | 334 |
| 3 layers | −4.78 | −1.02 | −0.80 | 5.3 | 1.00 | 488 |
| 1 layer, readouts only (D5), same fresh games | | | +1.36 | 4.6 | 0.50 | 145 |
| untrained (1 layer) | | | −5.05 | 1.0 | 1.00 | 169 |

**Conclusion.**
- Evolving the hidden layer too roughly triples the 1-layer agent's
  reward on fresh games (+3.78 vs +1.36) and adds two kills per game.
- With every weight evolving, a second layer no longer hurts (+3.18,
  within one seed's noise of 1 layer) but does not help either; a third
  layer still fails (never survives a game).
- One seed per depth; the 1- vs 2-layer difference is not established.

Data: `results/dynamic/es_evolve_all_all_d{1,2,3}.jsonl.gz`; agent in
`results/dynamic/doom_agent/evolve_all_d1/`.

### D10. Growing layers during evolution (2026-09-28)

**Question.** The user: add layers while the network runs, as soon as it
starts learning.

**Setup.** As D9, starting with 1 layer, `--grow-to 3`: a layer is added
on top once the mean validation reward of the last 20 generations beats
the first 20 at the current depth by 1.0. The readouts read every layer,
and their weights from a new layer start at zero, so growing does not
change play (checked: identical reward and kills before and after two
growths on a test game).

**Result.** It grew to 2 layers at generation 77 and to 3 at 133.
Validation, mean of the last 100 generations: **+4.36**, the best of all
runs (best single generation 8.06), 7.9 kills. On the 30 fresh games:
reward +2.21, 6.4 kills, 0.67 deaths, 496 spikes per step (untrained 1
layer −5.27).

**Conclusion.**
- A grown 3-layer network plays far better than one evolved at 3 layers
  from the start (+2.21 vs −0.80 on fresh games): growing makes depth
  trainable.
- It does not beat the plain 1-layer network on fresh games (+2.21 vs
  +3.78), although it led on its validation games; the gap between its
  validation and fresh scores suggests it fitted the validation seeds'
  situations more than the others (it had the most weights), or seed
  noise. It fires about three times as many spikes.

Data: `results/dynamic/es_evolve_all_grow3.jsonl.gz`; agent in
`results/dynamic/doom_agent/evolve_all_grow3/` (3 layers, `readout_from`
`all`).

### D11. Sound, the new defaults and how the readouts are read (2026-09-29)

**Question.** Does hearing help the D9 winner (1 layer, every weight
evolving)? And does it keep its score with the library defaults of
2026-09-28 (normalised sums, spontaneous amplitude 0.1)?

**A flaw found first.** The action values were the readouts at the last
of the step's 6 ticks. Without sound the screen is held for all 6 ticks,
the E-R neurons fire early and are silent by the last tick, all readouts
read 0, and the agent always takes action 0: every candidate plays the
same game and evolution gets no signal. With sound the audio changes every
tick and keeps the neurons firing. So in D1–D10 sound mainly kept the
network active at decision time. `readout` `sum` (the readouts summed
over the step's ticks) was added so a silent-sound agent can act.

**Setup.** As D9 (1 layer, `evolve` `all`, sound on unless stated, 500
generations, one seed each), final weights tested on the same 30 fresh
games (seed 4242).

**Result.**

| Library defaults | Readout | Sound | Fresh games: reward | Kills | Survives | Spikes |
|------------------|---------|-------|---------------------|-------|----------|--------|
| old (raw sums, amplitude 0.01) | last | yes | **+3.78** (D9) | 6.7 | 57% | 172 |
| new | last | yes | +3.35 | 7.7 | 30% | 165 |
| old | sum | yes | +0.99 | 7.5 | 0% | 173 |
| new | sum | yes | −0.85 | 5.3 | 0% | 161 |
| new | sum | no | −0.76 | 5.3 | 0% | 63 |

**Conclusion.**
- **Summing the readouts costs about 3 reward** (and all survival) with
  either library; reading the last tick stays the default.
- **The new defaults cost little** (+3.35 vs +3.78, one seed each; kills
  rise, survival falls).
- **Hearing adds nothing** where both can act (summed readouts: −0.85
  with sound, −0.76 without), and the deaf agent fires 60% fewer spikes.
  Sound's value so far is as a changing input that keeps E-R active,
  not as information about monsters.

Data: `results/dynamic/es_d11_*.jsonl.gz`.

## Open questions

- **Features.** Every hidden layer is frozen and random; only 8 readouts
  (about 1000 weights) evolve. A whole level likely needs learned
  features or a larger evolved part.
- **Where habituation matters.** It made no difference in the 2-layer
  network (D5); it has not been compared on the 1-layer winner.
- **Sound.** No experiment has yet shown that hearing helps (D2, D11); in
  the default setup it keeps the E-R layer active between frames. A test
  with the sound replaced by unrelated audio of the same loudness would
  separate the two.
- **Baseline.** The fair reference for any temporal claim is a stateless
  network given past frames ([dynamic](dynamic.md)); it has not been run
  under evolution.
