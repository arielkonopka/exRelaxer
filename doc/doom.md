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
| Trained by | evolution of the 8 action readouts (`es.py`, [D5](#d5-evolution-of-the-readouts-topologies-on-defend_the_center-2026-09-28)); the hidden layer is the frozen random network |
| `defend_the_center` | validation reward −4.8 → +0.95 (mean of the last 100 of 500 generations), about 5 kills per episode, 145 spikes per step |
| `map01` | explores (twice the untrained distance, more items and doors); no exits ([D6](#d6-evolution-on-a-whole-level-map01-2026-09-28)) |
| Files | `results/dynamic/doom_agent/cmp_er_d1_none/` (arena), `.../map01_er_d1_from_dtc/` (MAP01): `best.exr`, `best_theta.npy`, `config.json` |

Rebuild it in Python:

```python
import json, numpy as np, sys
sys.path.insert(0, "NNtesting/experiments/doom_rl")
import es
d = "results/dynamic/doom_agent/cmp_er_d1_none"
p = json.load(open(d + "/config.json"))["params"]
player = es.build(p, 0, np.load(d + "/best_theta.npy"))   # net seed 0
```

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

### D7. `map01` with three-minute episodes (2026-09-28, running)

**Question.** Does the agent reach the exit when episodes are long enough?

**Setup.** As D6, three-minute episodes (`episode_tics` 6300), 600
generations, starting from the D6 warm-started run's final weights.

**Status.** At generation 196 validation has not moved (−6.4 in the first
50 generations, −5.9 in generations 150–196; the scale is lower than D6
because longer episodes collect more penalty). No exits so far.

## Open questions

- **Features.** Every hidden layer is frozen and random; only 8 readouts
  (about 1000 weights) evolve. A whole level likely needs learned
  features or a larger evolved part.
- **Where habituation matters.** It made no difference in the 2-layer
  network (D5); it has not been compared on the 1-layer winner.
- **Sound.** No experiment has yet shown that hearing helps (D2).
- **Baseline.** The fair reference for any temporal claim is a stateless
  network given past frames ([dynamic](dynamic.md)); it has not been run
  under evolution.
