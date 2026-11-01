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
How to run everything below is in the [guide](doom_guide.md).
Doom validates mechanisms end to end; they are debugged first in
controlled benchmarks ([protocol](protocol.md#1-controlled-first-doom-last)).

## Current best agent

| | |
|---|---|
| Network | one hidden layer, 128 E-R neurons with fading habituation (tolerance 0.05, decay 0.9 per tick from the 2nd repeat), no feedback; 6 network ticks per game step |
| Senses | screen 160 × 120 gray pooled 8 × 8 to 20 × 15; stereo sound through a two-ear cochlea |
| Trained by | evolution of every weight, the hidden layer's and the 8 action readouts' (`es.py`, `evolve` `all`), first on 2-minute games ([D11](#d11-sound-the-new-defaults-and-how-the-readouts-are-read-2026-09-29)), then 300 generations on games that last until death ([D12](#d12-games-that-last-until-death-2026-09-29)) |
| Library | defaults of 2026-09-28 (normalised sums, spontaneous amplitude 0.1, E-R recovery 0.9); action values read at the last tick. It needs recovery 0.9, which `doom_rl` now pins ([D15](#d15-the-agents-under-the-new-default-recovery-2026-10-28)) |
| `defend_the_center` | 30 fresh games with no time limit (capped at 30 minutes): reward +16.5 (untrained −5.3), 19.8 kills, survives 21 minutes on average, lives through half of the games to the cap, 157 spikes per step |
| `map01` | not yet tried with every weight evolving; readout-only agents explore but never exit ([D6](#d6-evolution-on-a-whole-level-map01-2026-09-28), [D7](#d7-map01-with-three-minute-episodes-2026-09-28)) |
| Files | `results/dynamic/doom_agent/untildeath_d1/`: `best.exr` (the whole network in the library's own format), `best_theta.npy` (its evolved weights), `config.json`, `best.json` (generation 296, validation +20.3) |

Watch it play: `python3 NNtesting/experiments/doom_rl/watch.py` writes
`replay.mp4` (what it saw), `replay.srt` (what it did, as subtitles) and
`replay.html`, a step-by-step replay of both (`--live`
shows the game window instead; [doom_rl README](../NNtesting/experiments/doom_rl/README.md)).

Load the saved network with the library alone (`best.exr`); it plays
only through `doom_rl`, which feeds it the screen and sound and reads its
8 outputs:

```python
import exrelaxer as exr
net = exr.Network.load("results/dynamic/doom_agent/untildeath_d1/best.exr")
print(net.describe())
```

Or rebuild the playing agent from its weights:

```python
import json, numpy as np, sys
sys.path.insert(0, "NNtesting/experiments/doom_rl")
import es
d = "results/dynamic/doom_agent/untildeath_d1"
p = json.load(open(d + "/config.json"))["params"]
player = es.build(p, 0, np.load(d + "/best_theta.npy"))   # net seed 0
```

Agents evolved before the library defaults of 2026-09-28 (D1–D10, and
`results/dynamic/doom_agent/evolve_all_d1/`) have no `normalize` or
`spontaneous_amplitude` in their `config.json`, so `doom_rl` rebuilds them
with raw sums and amplitude 0.01; new runs record both.

## Setup shared by all entries

- **Game**: [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
  (`python3 -m pip install vizdoom`, or `./build.sh --vizdoom`, which
  also builds the Python package), headless, about 5000 game steps per
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
  −0.005 per step. From D13 (2026-09-29, the user's suggestion): a new map
  square (64 units) entered +0.05, and firing costs −0.0005 per round only
  in steps that kill nothing. On MAP01 also +0.1 per 64 units of new
  closest walking distance to the exit (the user's suggestion; about
  3600 units from the start). Full table in the
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

### D12. Games that last until death (2026-09-29)

**Question.** The agents so far played games cut off at about two minutes.
If a game lasts until the agent dies, does it learn to stay alive longer?

**Setup.** Started from the D11 winner (new defaults, last-tick readout,
sound, 1 layer, every weight evolving). `episode_tics` 63000, so a game
ends at death or after 30 minutes. 300 generations, 12 antithetic pairs,
3 games per candidate, σ 0.05, lr 0.02. Tested on the same 30 fresh games
(seed 4242) with the same 30-minute cap.

**Result.** Validation rose from +2.7 (first 20 generations, 8.8 kills)
to +13.3 (last 20, 17.5 kills); the best validation was +20.3 at
generation 295.

| Agent | Reward | Kills | Deaths per game | Survives (mean) | Spikes |
|-------|--------|-------|-----------------|-----------------|--------|
| untrained | −5.3 | 0.8 | 1.0 | 11 s | 166 |
| start (D11 winner) | +2.8 | 8.8 | 1.0 | 62 s | 167 |
| final generation | +12.6 | 17.1 | 0.7 | 16 min | 159 |
| best validation | **+16.5** | **19.8** | 0.5 | **21 min** | 157 |

**Conclusion.**
- **Survival was learnable once the games allowed it.** The start agent
  always died in about a minute; after 300 generations the agent lives
  to the 30-minute cap in half of the fresh games, and kills more than
  twice as many monsters. Two-minute games gave little reward for staying
  alive, so evolution had not selected for it.
- The network and its activity are unchanged (157 spikes per step vs
  167); the gain is in the weights, not in more firing.
- Kills are probably bounded by the ammunition: the agent starts with 26
  rounds (seen in `watch.py`'s replay), so the long
  games are likely about not being hit once ammunition runs out. How the
  agent survives has not been examined.
- One seed; the best-validation weights were picked on validation games,
  not on the fresh ones.

Data: `results/dynamic/es_d12_untildeath.jsonl.gz`, agent and test in
`results/dynamic/doom_agent/untildeath_d1/`.

### D13. `map01` with rewards for new places and for nearing the exit (2026-09-29 to 30)

**Question.** On a whole level the agents so far wandered and never left
(D6 to D8). Do rewards for reaching places not visited before and for
getting closer to the exit make them move through the level?

**Setup.** Reward changes (all in `rewards.py`):
- +0.05 for each 64 × 64-unit map square first entered in a game
  (`w_explore`);
- the fire penalty is charged only in steps with no kill, 0.0005 a round
  (was 0.001 for every round);
- +0.1 for each 64 units by which the agent gets closer to the exit than
  it has been in that game (`w_approach`). The distance is the walking
  distance around walls (`exitmap.py`: the exit lines come from the WAD,
  then a breadth-first search on a 16-unit grid). On MAP01 the start is
  about 3600 units from the exit by path and 600 in a straight line.

A game ends at death, after 30 minutes, or after 2 minutes without a new
square (`stall_steps` 1050); without that stop the untrained agent
idles for the full 30 minutes. 1 E-R layer with habituation, every weight
evolving, 12 antithetic pairs, 3 games per candidate, σ 0.05, lr 0.02.
First 108 generations with the new-place reward only (`map01_explore`,
from scratch; validation −4.6 to −3.7). Then 300 generations with the
exit reward added, warm from that run's best (`map01_exit`). Tested on
30 fresh games (seed 4242) with the full reward.

**Result.** Validation rose from −1.5 (first 20 generations) to +0.3
(last 20); the best was +2.4 at generation 106.

| Agent | Reward | Kills | Squares visited | Closest to exit | Exits | Game length |
|-------|--------|-------|-----------------|-----------------|-------|-------------|
| untrained | −3.18 | 0.2 | 23 | 2614 | 0 | 1.9 min |
| start (new places only) | −0.40 | 0.7 | 48 | 2241 | 0 | 1.7 min |
| final generation | **+0.69** | 0.8 | **68** | **2193** | 0 | 2.1 min |
| best validation | +0.09 | 0.7 | 55 | 2333 | 0 | 1.8 min |

Closest to exit is the mean over games of the smallest walking distance
reached, in map units. About three games in four ended in death, the
rest by the 2-minute stall rule.

**Conclusion.**
- **Rewarding new places works.** Trained agents cover two to three times
  more of the map than the untrained one, pick up more items and open
  doors.
- **Rewarding nearness to the exit barely moved the agent toward it.**
  The final agent gets 50 units closer than the agent trained without
  that reward, and ends about 2200 units (by path) from the exit. No
  agent left the level. The approach reward is paid once per new closest
  distance, and exploring earns more, so evolution did not trade one for
  the other.
- Games are short because the agent dies: on MAP01 it meets monsters it
  cannot yet handle, unlike the arena where it learned to survive (D12).
- The best-validation weights test worse than the final ones: with 5
  validation games, the pick is noisy.
- One seed.

Data: `results/dynamic/es_d13_map01_explore.jsonl.gz`,
`results/dynamic/es_d13_map01_exit.jsonl.gz`, agent and test in
`results/dynamic/doom_agent/map01_exit_d1/`.

### D14. Growing the D12 agent layer by layer, with and without threshold readouts (2026-09-29 to 10-06)

**Question.** Does training a small network first and adding an E-R layer
each time it has learned (the "biological" schedule) give a better agent
than the one-layer D12 agent? Do deeper layers give a longer memory, and
does it help to let the readouts read each neuron's E-R threshold, where
its history is kept?

**Setup.** Both runs start from the D12 agent (1 E-R layer with
habituation, 128 neurons, every weight evolving) on `defend_the_center`,
games until death or 30 minutes, D12's reward. `es.py --grow-rule
plateau --grow-to 5 --grow-window 30 --grow-margin 0.5`: the first network
grows once its 30-generation validation mean stops gaining 0.5; each grown
network grows again once its mean beats the level at its own growth by
0.5. A new layer's readout weights start at zero, so play is unchanged when
it is added. 12 antithetic pairs, 3 games per candidate, 5 validation
games, σ 0.05, lr 0.02.
- Without thresholds: grew to 2 layers at generation 60 and to 3 at 193;
  stopped by hand at generation 289 (last 20 generations: validation 14.7).
- With `readout_thresholds` (the readouts also read every hidden neuron's
  threshold; zero weights at the start, so the D12 agent plays identically):
  grew to 2 layers at generation 60, 3 at 90, 4 at 266; ran to generation
  688. With 4 layers, validation stayed at 13 to 14 for 420 generations
  and never reached the 16.3 needed for a fifth layer.

Tested on D12's 30 fresh games (seed 4242, until death or 30 minutes).

| Agent | Layers | Reward | Kills | Deaths per game | Survives (mean) | Spikes per step |
|-------|--------|--------|-------|-----------------|-----------------|-----------------|
| D12 agent (start of both runs) | 1 | **+16.6** | **19.8** | 0.50 | **21 min** | **157** |
| grown, no thresholds, generation 289 | 3 | +13.5 | 17.7 | 0.67 | 17 min | 406 |
| grown, thresholds, best validation (gen 264) | 3 | +11.5 | 15.7 | 0.67 | 18 min | 404 |
| grown, thresholds, generation 688 | 4 | +12.5 | 16.9 | 0.70 | 16 min | 538 |

How far back each layer remembers (`memprobe.py`, 12 fresh games of 3
minutes): a ridge readout from a layer's state now predicts the screen k
steps ago; the table gives the gain in R² over the network's current input
alone.

| Source | k = 1 | k = 16 (about 2 s) |
|--------|-------|--------------------|
| D12 agent, outputs | +0.024 | +0.015 |
| D12 agent, thresholds | +0.006 | +0.015 |
| 3 layers (no thresholds, gen 289), layer 1 outputs | +0.025 | +0.027 |
| same, layer 2 outputs | +0.011 | +0.016 |
| same, layer 3 outputs | +0.003 | +0.010 |
| same, layer 1 thresholds | +0.005 | +0.030 |

Past actions are recoverable 1 to 2 steps back from any source (72 % with
outputs and thresholds together against 65 % with outputs, 45 to 48 %
majority), and at chance from 4 steps back.

**Conclusion.**
- **Growing did not beat the one-layer agent.** Every grown network scores
  below the D12 agent it started from (+11.5 to +13.5 against +16.6), dies
  more often and fires 2.6 to 3.4 times more spikes. Per unit of activity
  it is far worse. Validation (5 games) made the grown networks look as
  good as D12 at times; the 30 fresh games do not.
- **Deeper layers do not remember longer.** Each further layer carries less
  of the current screen (R² 0.65, 0.36, 0.20 for layers 1 to 3) and less of
  the past. The trace of the past in any layer is weak (a few hundredths of
  R²) and does not grow with depth.
- **Thresholds hold the same short trace as the outputs.** Reading them
  made the network grow faster (3 layers at generation 90 against 193) but
  did not raise its score.
- With 12 pairs and noisy until-death games, evolution of 78 000 to 100 000
  weights may simply be too slow to use the extra layers; one seed per run.

Data: `results/dynamic/es_d14_grow.jsonl.gz`,
`results/dynamic/es_d14_grow_thresholds.jsonl.gz`, agents, tests and probes
in `results/dynamic/doom_agent/grow_d14/`.

### D15. The agents under the new default recovery (2026-10-28)

**Question.** The library's default E-R recovery became 0.5 on 2026-10-08
(research [§36](research.md#36-reproducibility-and-the-new-default-recovery)).
Every Doom agent was evolved at 0.9, and `doom_rl` had no setting for it, so
since then it rebuilt saved agents at 0.5. Do they still play?

**Setup.** The D12 agent (`best_theta.npy`) and the untrained network it
grew from, rebuilt by `es.build` at recovery 0.9 and at 0.5, on D12's 30
fresh games (seed 4242, until death or 30 minutes).

| Network | Recovery | Reward | Kills | Survives (mean) | Spikes per step |
|---------|----------|--------|-------|-----------------|-----------------|
| D12 agent (recorded) | 0.9 | +16.5 | 19.8 | 21 min | 157 |
| D12 agent | 0.9 | +16.55 | 19.8 | 21 min | 157 |
| D12 agent | 0.5 | −3.2 | 2.8 | 1.8 min | 313 |
| untrained | 0.9 | −5.3 | 0.8 | 11 s | 166 |
| untrained | 0.5 | −5.9 | 0.2 | 9 s | 329 |

**Conclusion.**
- **At 0.9 the agent reproduces**: the same kills, deaths and survival to
  the step; the reward differs by 0.04 because the fire penalty has counted
  only misses since D13.
- **At 0.5 it no longer plays**: it fires twice as many spikes, and the
  evolved weights, tuned to the slower-recovering thresholds, give almost
  nothing (−3.2, barely above untrained).
- `doom_rl` now has a `recovery` setting, 0.9 by default and for saved
  settings without it, as `goe_rl` has had since G5.

Data: `results/dynamic/doom_agent/untildeath_d1/recovery_retest.json`
(`recovery_retest.py`).

### D16. A layer that grows when it saturates, from one neuron (2026-10-28/29)

**Question.** The Snake growth findings (research
[§37-§39](research.md#37-growth-re-evaluated-new-recovery-more-seeds-from-one-neuron)):
a layer that starts with one E-R neuron and gains 4 whenever it is
saturated in 25 % of ticks, with grown neurons "young for good" (resting
threshold × 0.5, learning gain × 3) and nothing frozen. Do they carry over
to Doom, where evolution, not a learning rule, sets the weights?

**Setup.** `es.py --grow-width 128` (h1 widens between generations when the
10 validation games saw it saturated in ≥ 25 % of ticks; the new neurons
are read with weight zero, so play does not change; every weight keeps
evolving, so nothing is frozen). "Young": `--young-resting 0.5
--young-gain 3` (ES noise and steps × 3 on the grown neurons' weights, the
evolution analogue of a learning gain). Against fixed layers of 128 (D12's
size) and 8. All at recovery 0.5 (the library default), 1 E-R layer with
fading habituation, sound, `defend_the_center`, 1-minute games, σ 0.05,
lr 0.02, 12 pairs × 3 games, 10 validation games, 300 generations; 3 seeds
(ES seed = network seed = 1, 2, 3). The best validation weights of each
run then played 30 fresh games (seed 4242).

| Arm | Neurons at the end | Reward, 1-minute games (seeds 1/2/3) | Mean | Until death: survives | Spikes per step |
|-----|--------------------|---------------------------------------|------|-----------------------|-----------------|
| fixed 128 | 128 | 7.5 / 7.2 / 1.2 | +5.3 | 79 s | 325-338 |
| fixed 8 | 8 | 11.9 / 8.9 / 0.9 | +7.2 | 85 s | 17-24 |
| grows from 1 | 5 (all seeds) | 12.8 / 6.0 / 4.4 | +7.7 | 111 s | 13-17 |
| grows from 1, young | 5 (all seeds) | 0.5 / 10.6 / 1.9 | +4.3 | 233 s* | 17-18 |
| untrained (any arm) | | −4.7 to −5.7 | | 9-13 s | |

\* one seed survives long; the other two rush in, kill about 6 and die in
under 20 s.

How often the untrained layer is saturated (10 validation games of 1
minute): 36 % of ticks with 1 neuron, 33 % with 2, 5 % with 4, 1 % with 8,
never with 16 or more (at recovery 0.9: 63 % with 1, 36 % with 4).

**Conclusion.**
- **The saturation trigger stops at 5 neurons.** In every seed the layer
  grew once, in the first generation (1 → 5), and never again: with the
  screen and sound always present, some neuron fires on most ticks. On
  Doom the trigger is a one-off size finder, not ongoing growth.
- **Five to eight evolved E-R neurons play as well as 128**, with 15 to
  20 times fewer spikes. The differences in reward between the arms are
  within the spread between seeds (1 to 13 for the same arm); the
  difference in activity is not.
- **Young grown neurons do not help under evolution.** Their arm is the
  most erratic; with ES there is no learning rule for a gain to speed up,
  and larger noise on 4 of 5 neurons mostly makes the search coarser.
- The D12 agent (+16.5 until death) remains the best; it was evolved at
  recovery 0.9 for 300 more generations from a trained start, so it is not
  a like-for-like comparison.

Data: `results/dynamic/doom_agent/growth_d16/` (per run `config.json`,
`best.json`, `best_theta.npy`, `log.jsonl.gz`; `fresh30.jsonl` from
`fresh.py`; `run.sh`, `run_seeds.sh`).

### D17. A TD critic instead of the raw reward: actor-critic on `basic`, `defend_the_center` and `map01` (2026-10-30)

**Question.** (User's spec.) Does an actor-critic help, with δ = r + γV(s′) − V(s) from a
linear TD(λ) critic as the modulator of per-action readouts over frozen
features, learning on every step?

**Setup.** `NNtesting/experiments/td_actor_critic/doom_td.py`
([README](../NNtesting/experiments/td_actor_critic/README.md)). Not the
shared setup above: the scenario's own 3 buttons and reward (`basic` / 100;
`defend_the_center` as is), no sound, frame skip 4, one network tick per
step. Screen 40 × 30 gray → frozen Gabor + centre-surround bank, max-pooled
(500 values) → frozen 256-neuron recurrent E-R reservoir (recovery
0.5-0.99); critic and actor read both. γ 0.97, λ 0.9, softmax actor with
post onehot − π, features centred. 3 seeds; greedy test on fresh seeds.

**Result.**
- `basic` (3000 episodes): δ 0.99-1.00 kill rate and +69 on test, against
  raw reward 0.56 (−105) and random actions 0.62 (−141). The reservoir
  alone (no bank) reaches 0.67.
- `defend_the_center` (2000 episodes): δ +6.1 (7.1 kills), raw reward +5.4
  (6.4 kills), random +0.3 (1.3 kills); seeds spread 3.5-10.3. An entropy
  bonus keeps exploring but gives the same mean.
- Before it worked: a timeout scored as terminal rewarded running out the
  clock (the agent stopped shooting); uncentred E-R thresholds kept the
  critic from learning; pixels without the bank never beat random. Details
  in [research §41](research.md#41-a-td-critic-δ-as-the-modulator-actor-critic).

**Conclusion.** δ makes online learning work on `basic` where the raw
reward does not; on `defend_the_center` it is slightly ahead. The frozen
vision bank matters more than the E-R reservoir.

**Data.** `results/td-actor-critic/doom/` (config, test, gzipped training
log per run); the project's `reports/doom-td/`.

**Addendum (2026-10-30, the user's follow-ups: games until death with
survival times, and a whole level).** `doom_td.py --set until_death=M`
(no time limit, a cap of M minutes; survival seconds logged),
`--scenario map01` (doom_rl's 8 buttons and shaped reward) and
`--checkpoint N` (resume after a restart).

- `defend_the_center` until death (30-minute cap, 3 seeds, 30 test games):
  δ 7.1 kills, survives 16.8 s (median 17.6, longest 30.9); δ + entropy
  6.5 kills, 16.5 s; raw reward 6.4 kills, 16.2 s; random 1.3 kills,
  9.5 s. The same numbers as with the 60 s limit: these agents always died
  before it. Training survival rises 13.8 → 16.1 s (raw 12.9 → 15.0 s). They
  shoot well but never dodge; the evolved D12 agent survives 21 minutes.
- `map01`, 600 games until death (5-minute cap), 3 seeds, 10 test games:

  | | test reward | kills | survival | squares visited | doors | items | exits |
  |---|---|---|---|---|---|---|---|
  | δ | −9.3 | 0.5 | 283 s | 28 | 0.0 | 1.1 | 0 |
  | raw reward | −4.1 | 1.0 | 89 s | 37 | 1.2 | 2.0 | 0 |
  | random | −3.7 | 0.2 | 142 s | 38 | 0.0 | 0.5 | 0 |

  δ's greedy policy hides: it outlives the cap in most test games but
  explores less than random and collects idle penalties; in training its
  reward falls (−2.7 → −5.7) as the policy narrows (entropy 1.07 → 0.28).
  The raw reward learns to kill (0.34 → 0.96 per game in training), open
  doors and pick up items, and dies sooner. Neither beats random on reward
  or reaches the exit. A first run with the reward × 0.2 and an entropy
  bonus did not move the policy at all (entropy 2.07 of 2.08) and is kept
  only as data (`map01/`; the comparison is `map01b/`).

**Conclusion (addendum).** On an arena δ helps a little; on a whole level a
linear critic over frozen vision with a 3.7 s horizon does not see what
leads to reward, and δ teaches hiding. A level needs memory or a place
representation (the spec's own open question), and probably a critic that
sees more than the current frame.

### D18. Homeostatic drives: health as an input, drive reduction as the reward (2026-10-30)

**Question.** (User's spec, phase 2.) Can the D17 actor-critic learn from an
internal drive instead of the game's score: health fed into the reservoir,
reward r = D(h_prev) − D(h_now) with D(h) = |100 − h| / 100?

**Setup.** As D17 (`doom_td.py`, frozen Gabor bank + 256-neuron E-R
reservoir, critic and actor read both), on ViZDoom `health_gathering`
(acid floor, medkits; 3 buttons) and `health_gathering_supreme` (also
poison vials, a harder layout). `--set drives=1`: health / 100 is one more
input of the reservoir (its weights × 5) and a raw feature. Arms: δ + drive
reward + health input; δ + the scenario's reward (+1 per tic, −100 at
death, × 0.01); δ + drive reward without the health input ("blind"); the
raw drive reward as modulator (no δ); random. Games until death, 5-minute
cap; 3 seeds; 20 greedy test games on fresh seeds. No ammo: these
scenarios have no weapon.

**Result.**
- `health_gathering`, 150 games:

  | | test survival | medkits | test deaths | first game at the cap |
  |---|---|---|---|---|
  | δ + drive + health | 300 s (cap) | 161 | 0% | 15-23 |
  | δ + scenario reward | 295 s | 160 | 2% | 29-61 |
  | blind | 254 s | 136 | 20% | 5-22 |
  | raw drive (no δ) | 252 s | 138 | 22% | 34-144 |
  | random | 15 s | 1.5 | 100% | - |

- `health_gathering_supreme`, 150 games: every arm dies within about 30 s;
  δ + drive 22 s (7.5 medkits), blind 16 s, raw drive 15 s, scenario reward
  12 s, random 12 s.
- `health_gathering_supreme`, 1000 games (no random or raw arm):

  | | test survival (per seed) | longest | medkits | training survival, per 200 games |
  |---|---|---|---|---|
  | δ + drive + health | 30 s (29, 26, 35) | 78 s | 12.8 | 14 → 20 → 22 → 24 → 27 |
  | blind | 29 s (30, 31, 27) | 81 s | 12.4 | 13 → 18 → 22 → 25 → 26 |
  | δ + scenario reward | 17 s (20, 19, 12) | 44 s | 4.2 | 12 → 15 → 17 → 18 → 19 |

  Still rising at the end; every test game ends in death.

**Conclusion.** Drive reduction teaches the agent to gather medkits as well
as the score does on the easy map, about twice as fast, and much better on
the hard one, where +1 per tic says little about which action helped.
Feeling health makes the easy-map policy reliable (no deaths) but adds
nothing on `supreme`, where the medkit itself in view is what matters. A
drive-reduction return telescopes (D(h_0) − D(h_end), about −1 for every
game that ends in death), so only γ < 1 makes postponing damage worth
anything; it is enough here.

**Data.** `results/td-actor-critic/doom/{health_gathering,hg_supreme,hg_supreme_1k}/`
(config, test, gzipped training log per run, `run.sh`); the project's
`reports/doom-td/`.

### D19. A place layer: place and head-direction cells on `my_way_home` and `map01` (2026-10-30)

**Question.** (Phase 3, after D17's `map01` agent learned to hide.) Does
knowing where it is let the actor-critic learn to navigate: place cells
from the map position next to vision?

**Setup.** As D17 (`doom_td.py`, frozen Gabor bank + 256-neuron E-R
reservoir), plus `--set place=1`: the position tile-coded by 4 offset grids
of 128-unit squares (hashed into 1024 cells, 4 active) and 8
head-direction cells, appended to the features (a linear critic over them
is a smoothed table of values per place). `features=place`: place cells
alone. γ 0.99 on `my_way_home` (5 buttons, +1 for the vest, 60 s limit,
600 games, 30 test games); `map01` as D17's `map01b` (600 games until
death, 5-minute cap, 10 test games) at γ 0.97 and 0.99. 3 seeds.

**Result.**
- `my_way_home`:

  | features | test success (per seed) | training success, last 150 games |
  |---|---|---|
  | vision + place | 33% (37, 30, 33) | 29% |
  | place only | 28% (10, 27, 47) | 46% |
  | vision only | 9% (0, 20, 7) | 17% |
  | random | 16% | 14% |

  A 300-game pilot at γ 0.97 was weaker (place only 20%, vision 5%).
- `map01` (δ, with D17's runs and random for reference):

  | | test reward (per seed) | kills | survival | squares | doors | items | exits |
  |---|---|---|---|---|---|---|---|
  | place, γ 0.99 | **+0.1** (1.0, −2.9, 2.3) | 0.3 | 285 s | **45** | 0.0 | 1.1 | 0 |
  | place, γ 0.97 | −4.3 (−0.9, −6.2, −5.8) | 0.6 | 61 s | 38 | 0.9 | 1.6 | 0 |
  | D17 δ, no place | −9.3 | 0.5 | 283 s | 28 | 0.0 | 1.1 | 0 |
  | D17 raw reward | −4.1 | 1.0 | 89 s | 37 | 1.2 | 2.0 | 0 |
  | random | −3.7 | 0.2 | 142 s | 38 | 0.0 | 0.5 | 0 |

**Conclusion.** Place cells make navigation learnable: on `my_way_home`
vision alone ends below random, place doubles it. On `map01` place cells
with a longer horizon turn δ's hiding into exploring: the first
actor-critic above random on the level (+0.1 against −3.7), visiting more
squares than random, though it still opens no doors and finds no exit.

**Data.** `results/td-actor-critic/doom/{my_way_home,map01_place}/`
(config, test, gzipped training log per run, `run.sh`); the project's
`reports/doom-td/`.

### D20. A lifetime novelty bonus on `map01`; place cells learned without coordinates on `my_way_home` (2026-10-31)

**Question.** (Phase 4, the two next steps of D19.) Does a novelty bonus
make the D19 agent open doors? Can place cells be learned from the
reservoir, vision and self-motion instead of read from the game?

**Setup.** As D19. Novelty: `--set novelty=β` adds β / √(1 + N) for
entering a 64-unit square entered N times before over all training games
(not reset between games; none in tests), on top of the γ 0.99 place-cell
agent; β 0.02 and 0.1. Learned place: `--set vplace=256` appends 256 place
cells, prototypes of the (unit-length) reservoir + vision feature vector
found by online competitive learning (the winner moves toward the input,
rate 0.02; the 4 closest fire); `motion=1` feeds forward and sideways
speed and turning into the reservoir; `vplace_settle=20000` lets the rate
fall as 20000 / (20000 + updates). Test `field_spread`: the spread of the
positions where each learned cell wins over the spread of all positions
(1 = no place information). 3 seeds, 600 games.

**Result.**
- `map01` with novelty (10 test games per seed):

  | | test reward (per seed) | squares | survival | doors (test / training, last 150) |
  |---|---|---|---|---|
  | place, no novelty (D19) | +0.1 (1.0, −2.9, 2.3) | 45 | 285 s | 0 / 0.11 |
  | + novelty 0.02 | −2.8 (−10.6, 0.6, 1.5) | 32 | 266 s | 0 / 0.29 |
  | + novelty 0.1 | −0.6 (3.3, −7.0, 1.9) | 47 | 300 s | 0 / 0.02 |

  One seed at β 0.02 opens about one door per training game from game 150
  on, but its greedy test policy opens none; no exit anywhere. The agents
  see 120-210 distinct squares in their lifetime.
- `my_way_home`, learned place (30 test games per seed):

  | features | test success | training success, last 150 | field spread |
  |---|---|---|---|
  | coordinate place cells (D19) | 33% | 29% | |
  | vision + self-motion | 13% (0, 7, 33) | 38% | |
  | vision only (D19) | 9% | 17% | |
  | learned place + motion, settling | 7% | 29% | 0.45 |
  | learned place + motion | 6% | 20% | 0.38 |
  | learned place | 2% | 14% | 0.47 |
  | random (D19) | 16% | 14% | |

**Conclusion.** A count-based novelty bonus does not open doors or the
exit within 600 games: the agent's explored territory stays small and the
door it learns to open in one seed is lost by the greedy policy. Place
cells learned by competitive learning are place-like (each covers 40-50%
of the overall spread) but the critic cannot use them; they are
view-dependent (the same spot facing another way is another prototype)
and drift while the critic learns. Self-motion alone helps the most of the
coordinate-free inputs.

**Data.** `results/td-actor-critic/doom/{map01_novelty,learned_place}/`
(config, test, gzipped training log per run, `run.sh`); the project's
`reports/doom-td/`.

### D21. A door reflex and path integration (2026-10-31)

**Question.** (Phase 5, the two next steps of D20.) Does a door reflex get
the D19 agent through `map01`'s doors? Can place cells read a position the
agent integrates from its own movement instead of the game's coordinates?

**Setup.** As D19 (place cells, γ 0.99). Reflex: `--set door_reflex=1`
adds USE to the chosen action whenever the previous step tried to move
and moved less than 1 map unit (pushing against a wall). Path integration:
`--set path=1` feeds the place cells a dead-reckoned pose: each step's
forward and sideways movement and turn (from the game, each with 5%
relative noise, `path_noise`), summed from (0, 0, 0) at every spawn.
`my_way_home` spawns in different rooms; `map01` always at the same spot.
3 seeds; random with the reflex 100 games.

**Result.**
- `map01`:

  | | test reward (per seed) | squares | survival | doors, training (first → last 100) | doors at test |
  |---|---|---|---|---|---|
  | place (D19) | +0.1 (1.0, −2.9, 2.3) | 45 | 285 s | 0.11 (last 150) | 0 |
  | place + reflex | +0.3 (1.2, −2.3, 1.8) | 33 | 290 s | 0.11 → 0.01 | 0 |
  | path-integrated place | −1.6 (−0.9, 0.9, −4.9) | 29 | 275 s | | 0 |
  | no place (D17) | −9.3 | 28 | 283 s | | 0 |
  | random + reflex | −3.6 | 46 | 166 s | 0.10 | 0.2 |
  | random (D17) | −3.7 | 38 | 142 s | | 0 |

- `my_way_home`, path-integrated place cells: 2% test success with 5%
  noise, 3% exact (coordinate place cells 33%, random 16%).

**Conclusion.** The reflex makes random play open doors, but the δ agent
learns to stop: it opens one in eight games early on and almost none
after 100-200 games, while its survival grows; behind `map01`'s doors are
monsters, and the critic prices a door by what follows (+0.5 for the door,
more lost to damage). Path integration needs an anchor: on `my_way_home`
(random spawns) it is useless even when exact; on `map01` (fixed spawn) it
recovers much of what the game's coordinates give (−1.6 against +0.1,
against −9.3 without place).

**Data.** `results/td-actor-critic/doom/{map01_reflex,path_integration,map01_path}/`
(config, test, gzipped training log per run, `run.sh`); the project's
`reports/doom-td/`.

### D22. A reason to go on (exit approach, boredom) and landmark-anchored path integration (2026-10-31)

**Question.** (Phase 6, the two next steps of D21.) Does a goal that
outweighs the danger behind doors get the D21 agent (place cells, γ 0.99,
door reflex) further into `map01`? Can recognised views anchor path
integration when the spawn changes?

**Setup.** As D21 with the reflex. `--set approach=0.1`: doom_rl's
`w_approach`, +0.1 per 64 units of new closest walking distance to the
exit (D13). `--set boredom=0.01`: a drive that grows while no new square
is entered, r −= 0.01 × min(1, steps since a new square / 100), also given
to the critic and actor as an input. Both: both. `--set path=2`: view
prototypes (the vision features, competitive learning) remember the
integrated pose where they win; a life starts lost (no place cells) and is
placed when two different recognised views (similarity > 0.97, seen ≥ 5
times) agree within 64 units; placed, recognised views pull the estimate
by 0.1. 3 seeds, 600 games; test reward below is without the approach
reward.

**Result.**
- `map01`:

  | | test reward (per seed) | doors | squares | items | closest to exit | survival |
  |---|---|---|---|---|---|---|
  | reflex only (D21) | +0.3 (1.2, −2.3, 1.8) | 0.0 | 33 | 0.0 | | 290 s |
  | + approach | −4.8 (−4.5, −7.2, −2.6) | 1.3 | 48 | 1.2 | 2350 | 166 s |
  | + boredom | −1.6 (0.1, 0.7, −5.6) | 1.1 | **68** | **3.5** | | 80 s |
  | + both | −2.3 (−3.9, −0.2, −2.8) | 1.0 | 63 | 2.6 | 2400 | 83 s |
  | random (D17) | −3.7 | 0.0 | 38 | 0.5 | | 142 s |

  In training (last 150 games) every arm opens about one door per game
  (reflex only: 0.01). The exit is about 3600 units away by path at the
  start; the closest any test game gets is about 2200. No exits.
- `my_way_home`, anchored path integration: with random play the strict
  setting places the agent only 8% of the time but within 55-77 units
  (life 0's pure drift: 35); looser matching (one view, similarity
  0.93-0.97) places it 50-65% of the time but 400-600 units off. With the
  learner (strict): 3% test success (7, 0, 3), placed 0-30% of the time,
  errors back at 200-500 units.

**Conclusion.** A goal that grows while the agent stays put is what the
door reflex needed: with boredom the agent opens a door a game, explores
twice the squares and picks up seven times the items of the reflex-only
agent, and keeps a reward above random; it dies sooner, because what is
behind the doors fights back. The exit-approach reward pulls it about
1200 units of the 3600 toward the exit. Anchoring path integration on
single recognised views fails: look-alike rooms misplace the agent, and
each misplacement teaches more landmarks the wrong pose.

**Data.** `results/td-actor-critic/doom/{map01_goal,anchored_path}/`
(config, test, gzipped training log per run, `run.sh`; `anchored_path/tuning/`
the random-play sweeps); the project's `reports/doom-td/`.

### D23. A killer instinct: strike and orienting reflexes on the boredom agent (2026-10-31)

**Question.** (Phase 7, the user's "let's add a killer instinct".) D22's
boredom agent opens doors but dies to what is behind them. Do reflexes
that fight back keep it alive?

**Setup.** D22's boredom agent (place cells, γ 0.99, door reflex,
boredom 0.01). `--set attack_reflex=1`: ATTACK is added to the chosen
action whenever a monster's centre is within the middle eighth of the
screen; `orient_reflex=1`: TURN_LEFT or TURN_RIGHT is added toward the
biggest monster on screen when it is outside that cone; `kill_weight=3`
("bloodlust"): doom_rl's w_kill 3 instead of 1. Monsters are found in
ViZDoom's labels buffer (a stand-in for a learned detector, as the door
reflex uses the game's movement). 3 seeds, 600 games; random with both
reflexes 100 games.

**Result.**

| `map01` | test reward (per seed) | kills | survival | squares | doors | items |
|---|---|---|---|---|---|---|
| boredom (D22) | −1.6 (0.1, 0.7, −5.6) | 0.7 | 80 s | 68 | 1.1 | 3.5 |
| + strike | +0.7 (1.5, −2.0, 2.6) | 1.2 | 223 s | 77 | 0.7 | 3.1 |
| + strike + orient | **+13.1** (12.6, 9.7, 17.1) | **3.9** | **300 s** | **172** | **1.5** | **8.5** |
| + strike + orient + kill × 3 | +10.8 (14.0, 3.0, 15.3) | 2.8 | 272 s | 107 | 1.2 | 4.7 |
| random + strike + orient | +1.3 | 1.0 | 279 s | 60 | 0.2 | 1.6 |

No exit. Every strike + orient test game lasts to the 5-minute cap.

**Conclusion.** With an orienting reflex the boredom-driven learner
survives what it lets in and keeps exploring: +13.1 against the previous
best +0.3 (D21), with 2.5 times the squares and 4 kills a game. The
reflexes alone (random actions) score +1.3: the learned policy and the
reflexes need each other. A stronger kill reward does not add to it.

**Data.** `results/td-actor-critic/doom/map01_killer/` (config, test,
gzipped training log per run, `run.sh`); the project's `reports/doom-td/`.

### D24. Landmarks taught in a guided childhood (2026-10-31)

**Question.** (Phase 8.) D22's view landmarks taught themselves wrong
poses. Do sequences of views, a filter on landmarks seen in more than one
place, or being told the spawn pose during the first lives fix
`my_way_home` anchoring?

**Setup.** As D22's `path=2`, plus `landmark_seq` (a landmark is the
sequence of the last L distinct winning views), `landmark_spread` (a
landmark is trusted only while the poses it was seen at stay within that
many units) and `landmark_guided` (the spawn pose is given for the first
N lives). Random-play sweeps (60-80 games), then the learner (600 games,
guided 100 lives, 3 seeds): "spread" (match 0.95, seen 3, spread 64) and
"strict" (D22's setting).

**Result.**
- Sequences of 2-3 views never recur exactly and never place the agent;
  the spread filter alone does not lower the error (430-660 units).
- Guided childhood (random play, 40 guided lives): afterwards placed
  9-31% of the time, 115-230 units off (D22 unguided: 200-600).
- Learner: 10% test success in both settings (D22 unguided 3%,
  coordinates 33%, random 16%); after the guided lives placed 12-42% of
  the time, median error 100-320 units.

**Conclusion.** A guided childhood halves the anchoring error but not
enough: place cells 128 units wide need errors well under 100. View
landmarks of this kind (one winning prototype among 256 over pooled
Gabor features) alias too much in `my_way_home`'s look-alike rooms.

**Data.** `results/td-actor-critic/doom/guided_landmarks/` (`tuning/`:
the random-play sweeps); the project's `reports/doom-td/`.

### D25. Reaching for the exit (2026-11-01)

**Question.** (Phase 9.) D23's instinct agent explores and survives but
never exits. Does the exit-approach reward or a longer game get it there?

**Setup.** D23's "orient" agent (boredom 0.01, door reflex, place cells
γ .99, strike and orient reflexes), 3 seeds, 600 games:
- "approach": plus `approach=0.1` (doom_rl's reward for new closest
  distance to the exit), 5-minute games;
- "long": no approach reward, 15-minute games (`until_death=15`).

**Result.**
- No exit in any training or test game.
- approach: test score without the approach part +6.9 (3.2, 7.2, 10.5),
  1.0-3.2 kills, 0-1.5 doors, closest 2060-2740 units from the exit
  (D23 without it: +13.1, 3.9 kills).
- long: −1.5 (−7.3, 2.7, 0.1); seed 2 hides (1 kill, 84 squares, no
  doors, survives 900 s); seeds 1 and 3 reach 144-185 squares, 3.2-3.4
  kills, 1.3-1.7 doors and die in 20-30% of the games.

**Conclusion.** Straight-line distance to the exit points through walls
on MAP01, so the approach reward pulls the agent into dead ends and costs
it exploration. Three times the game time does not find the exit either.

**Data.** `results/td-actor-critic/doom/map01_exit/`; the project's
`reports/doom-td/`.

### D26. A learned threat detector (2026-11-01)

**Question.** (Phase 10.) Can the reflexes of D23 run on a detector
learned from vision instead of ViZDoom's labels?

**Setup.** D23's "orient" agent with `detector=100`: during the first 100
games a per-column detector over the frozen pool features learns online
from the labels (the reflexes still use the labels); from game 100 on,
and in every test game, the reflexes use the detector. 3 seeds, 600
games. `detector_probe.py` compares detectors offline on 80 random games.
- "logistic": a logistic unit, monster columns weighted 5×, threshold 0.5.
- "mlp": 64 tanh units, the column and its two neighbours, a replay
  buffer (`detector_replay=8`), threshold 0.95.
- "spray": a control that strikes every tick and never orients.

**Result.**

| detector | test score | kills | seconds | squares | died |
|---|---|---|---|---|---|
| labels (D23) | +13.1 | 3.9 | 300 | 172 | 0% |
| mlp + replay | +4.8 (2.5, 6.7, 5.3) | 2.6 | 196 | 112 | 60% |
| logistic | +2.0 (−2.5, 9.7, −1.2) | 1.4 | 122 | 98 | 83% |
| spray | −0.1 (1.9, 2.9, −5.1) | 0.9 | 159 | 69 | 60% |

- Logistic: two seeds stop firing after childhood, one fires on about
  one wrong column per tick and does well by spraying.
- Probe: offline the MLP reaches AUC 0.94; learned online in one pass
  every detector drops to 0.55-0.74; with replay 0.92.
- The mlp detector's false columns per tick go from 0.02 in childhood
  to 0.3-0.4 once it steers the gaze.

**Conclusion.** The replay detector is the first label-free instinct
agent above the boredom agent (−1.6) and random reflexes (+1.3), but it
keeps only about a third of the labelled gain; it should keep learning
after its childhood.

**Data.** `results/td-actor-critic/doom/map01_detector/` (`logistic/`,
`probe/`); the project's `reports/doom-td/`.

### D27. A detector that keeps learning (2026-11-01)

**Question.** (Phase 11.) D26's detector stops learning after its
childhood and its false alarms grow once it steers the gaze. Can it keep
learning from its own strikes, or wait for evidence?

**Setup.** D26's "mlp" agent (64 hidden units, neighbours, replay,
threshold 0.95, childhood 100 games), 3 seeds, 600 games:
- "feedback" (`detector_feedback=1`): a strike that fires a shot (ammo
  drops within two steps) teaches the middle columns: monster if
  `HITCOUNT` rose, none if not;
- "evidence" (`detector_evidence=0.5`): the reflexes read a leaky
  accumulator of the detector's output;
- "child300": 300-game childhood, 1 seed.

**Result.**

| detector | test score | kills | seconds | squares | died |
|---|---|---|---|---|---|
| labels (D23) | +13.1 | 3.9 | 300 | 172 | 0% |
| mlp (D26) | +4.8 | 2.6 | 196 | 112 | 60% |
| mlp + feedback | +5.6 (−0.3, 9.6, 7.6) | 2.2 | 147 | 132 | 83% |
| mlp + evidence | +0.3 (−1.0, −1.4, 3.3) | 0.7 | 174 | 65 | 57% |
| child300 | −10.1 | 0.2 | 228 | 23 | 40% |

- Feedback: 4-12 judged shots per game; hits per shot rise from 0.3 to
  0.5-0.7, false columns per tick fall to 0.04-0.14, recall falls to
  0.07-0.13. A first run that waited only one step for the shot judged
  2-6 shots per game and scored +3.3.
- child300: false columns per tick 0.07 in childhood, 0.57 right after.

**Conclusion.** Hit-or-miss is a label-free teacher that makes the
detector more precise but less sensitive; it does not fix the side false
alarms that drive the orienting loop. Accumulating evidence slows the
strike and loses. The best label-free agent stays around +5.

**Data.** `results/td-actor-critic/doom/map01_feedback/`; the project's
`reports/doom-td/`.

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
- **Memory through the feedback ladder.** `feedback_first` brings every
  hidden layer back to h1 as input (rungs of h1 neurons, or rung layers
  of their own model, e.g. relu with habituation, which do not fatigue as
  E-R does), and grows with `--grow-to`. It is untested: does a grown
  ladder beat the grown stack of D10, and do non-fatiguing rungs keep
  more of the past than E-R rungs? ([how to run it](doom_guide.md#feedback-ladder))
- **Baseline.** The fair reference for any temporal claim is a stateless
  network given past frames ([dynamic](dynamic.md)); it has not been run
  under evolution.
