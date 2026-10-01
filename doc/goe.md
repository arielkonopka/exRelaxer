# Gardens of Eris experiments

The running record of every experiment in which an E-R network plays
[Gardens of Eris](https://github.com/arielkonopka/Gardens-of-Eris) (GoE),
and how to run them. Each experiment is one entry in the [log](#log),
newest last, numbered `G1`, `G2`, ... so that other pages can cite them.
The research log summarises the findings in
[§28](research.md#28-gardens-of-eris). Doom, the other game, has its own
record ([doom](doom.md), [guide](doom_guide.md)); GoE is the cheaper test
bed: a 2-minute game takes about a second, against minutes for Doom.

## The game for the network

GoE is a grid maze that keeps growing around the player. Its repository
has a headless build and a Python package, `goe` (`agent/python`), shaped
like ViZDoom's `DoomGame`: `new_episode(seed)`, `get_state()`,
`make_action()`, `is_episode_finished()`. The same seed builds the same
world. There is one game per process: the game keeps its world in static
state, so parallel players are separate processes.

| | |
|---|---|
| Senses: eye | the cells around the player, (2r + 1)² of them, in 8 channels: wall, free, enemy, collectible, danger, door, apple, moving; optionally a 9th, `seen` (the cell is in sight) |
| Senses: body | energy / max energy, ammo (up to 10), spare avatars (up to 3) |
| Actions | 14 readouts: move, shoot and interact × up, down, left, right; next gun; use. The largest is played |
| Time | one game step is one move (8 game ticks, 50 ticks a second); the network runs `ticks` (3) ticks on each step's view |
| Reward | `reward` events (default since Gardens-of-Eris PR #289): the game's events weighed by `w_*`: +5 per item, +20 per golden apple, +2 per use, +10 per door opened, +5 per teleport, +10 per monster killed, +5 per mine set off, −0.2 per energy point lost, −`w_death` (50) per avatar lost, +0.1 per score point. `reward` score (G1): the game's score gained in the step (+1 per new cell visited, +1 per item, + the energy of what the player kills), −`w_death` (50) per avatar lost |
| Episode | `episode_ticks` (6000: 2 minutes, 750 moves) unless the last avatar dies first |

**The sight grows.** The player sees 2 + ln(steps) / 2 cells far
(`player::getViewRadius` in the game): 2.35 at the first step, about 5.3
after 750 steps, 6.7 after 11 250, without a bound. The eye therefore
covers from the start the furthest sight an episode can reach, and cells
beyond the current sight read zero in every channel until the sight
reaches them. `radius` `auto` (the default) sizes the grid for the
episode: ⌈2 + ln(episode_ticks / 8) / 2⌉, so 6 (13 × 13 cells, 1352 eye
inputs) for 2 minutes, 7 for 30 minutes, 9 for 10 hours. `seen_channel`
adds a channel that marks the cells in sight, so the network can tell an
empty cell from an unseen one and how far it sees.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
sudo apt-get install liballegro5-dev libopenal-dev libsndfile1-dev   # the game's libraries
PYTHON="$PWD/.venv/bin/python" ./build.sh --goe     # or --all: also ViZDoom
export PYTHONPATH="$PWD/build/EXrelaxer.py/package"
```

`--goe` clones the game from its repository into
`third_party/Gardens-of-Eris` (or updates that clone; `--goe-dir DIR` and
`GOE_REPO` choose another place or repository), pip-installs `goe` from it
(which builds the game), then builds and tests everything with the Python
package. The package finds the game's data in the clone it was built from,
so keep the clone. By hand: `git clone
https://github.com/arielkonopka/Gardens-of-Eris` and `pip install
./Gardens-of-Eris/agent/python`.

## Running

### The untrained network against a random player

```bash
NNtesting/nntest.py describe goe_rl
NNtesting/nntest.py run goe_rl --set model=er,relu --set games=10 --out goe.jsonl
NNtesting/tools/compare.py goe.jsonl --metric reward --metric reward_random
```

Every parameter can be set with `--set`, including those of the designed
networks below (`--set reservoir=256 --set seen_channel=true ...`).

### Evolution (es.py)

```bash
python NNtesting/experiments/goe_rl/es.py --out results/goe-es/er_reservoir --workers 4 --generations 300 \
    --config NNtesting/experiments/goe_rl/models/er_reservoir.json
python NNtesting/experiments/goe_rl/es.py --out results/goe-es/try --config '{"width": 256, "recurrent": true}'
```

OpenAI-ES, as for Doom: antithetic pairs, centred ranks, Adam, steps
relative to each layer's weight RMS. Every candidate plays the same
worlds; the current weights play fixed validation worlds (seed 777777 on)
every generation. `--config` takes JSON or a file of it.

| Option | Default | Meaning |
|--------|---------|---------|
| `--out DIR` | required | results folder; the same `--out` resumes (also a grown run, at its depth) |
| `--config` | `{}` | goe_rl parameters (JSON or a file), plus `evolve` (`all`, `readout`) and `grow_to`, `grow_window`, `grow_margin` |
| `--generations` | 300 | generations in total |
| `--workers` | 4 | game processes |
| `--pairs` | 12 | antithetic pairs per generation |
| `--episodes` | 2 | worlds per candidate |
| `--validation` | 6 | fixed validation worlds per generation |
| `--sigma`, `--lr` | 0.05, 0.02 | perturbation and Adam step, relative to the weights' RMS |
| `--net-seed`, `--seed` | 0, 1 | the random starting network, the evolution |
| `--grow-to N` | from the config | grow up to N hidden layers |

It writes `config.json`, `log.jsonl` (a line per generation: `depth`,
`validation_reward`, `validation_score`, `validation_avatars_lost`,
`validation_events` (each event's count, with the events reward),
spikes, population mean and best), `best.exr` (the network with the best
validation, in the library's format), `best_theta.npy`, `best.json` and
`state.npz` (for resuming). One generation with the defaults is 24
candidates × 2 worlds + 6 validation worlds, about 54 seconds of play,
spread over the workers.

### Network options

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model`, `width`, `depth` | `er`, 128, 1 | hidden neurons and layers |
| `habituation`, `habituation_*` | on | fade-mode habituation of the hidden layers |
| `recurrent`, `recurrent_scale` | off, 0.5 | each hidden layer also reads itself |
| `skip` | off | the top hidden layer reads every layer below it, not only the one under it |
| `readout_from` | `top` | the readouts read the top layer, or `all` hidden layers |
| `reservoir`, `reservoir_recurrent` | 0, 128 | echo-state reservoir of E-R neurons reading h1, plus neurons reading the reservoir; the readouts read it |
| `reservoir_evolve` | off | es.py evolves the reservoir too; by default it stays the frozen random one |
| `feedback_first` | 0 | feedback ladder: neurons per rung layer `r<k>`, which reads h<k> and which h1 reads, one tick late |
| `feedback_first_from` | `all` | every layer above h1 has a rung, or only the `top` one |
| `feedback_first_model`, `feedback_first_habituation` | `relu`, on | the rungs' neurons (relu, er, gate) and their habituation |
| `radius`, `seen_channel` | `auto`, off | the eye ([above](#the-game-for-the-network)) |
| `ticks`, `readout` | 3, `last` | network ticks per move; readouts at the last tick or summed |

With the reservoir, skip, ladder or `readout_from all`, the network is
built one layer at a time, as `es.py` grows it: h1 (eye, body) and its own
loop, the reservoir, the readouts, then each further layer with its
connections, its rung and its readout lines. `_shared/wiring.py` records
what every weight reads, which is how growing carries the weights over.
The plain stack keeps its original construction (G1 checks it plays as
before).

**Growing** (`grow_to`): the network starts at `depth` hidden layers and
gains one on top whenever it learns, when the mean validation reward of
the last `grow_window` (20) generations beats the first 20 at this depth
by `grow_margin` (1.0). The new layer starts random, every old weight
keeps its value by the input it reads, and the weights from the new
layer (and its rung) start at zero, so the grown network plays as before.
The workers restart with the deeper network. Growing needs `evolve` `all`
and sets `readout_from` `all`.

## Designed networks

Three networks for the questions of the log, in
`NNtesting/experiments/goe_rl/models/`. All three use E-R neurons with
fading habituation (the Doom winner's), 128 per hidden layer, the eye with
`radius` `auto` and the `seen` channel, and the same frozen reservoir: 256
E-R neurons reading h1 and 128 reading the reservoir (recurrent scale
0.5). The reservoir never evolves; everything else does.

| File | Network | Evolving weights | Neurons |
|------|---------|------------------|---------|
| `er_reservoir.json` | h1 + frozen reservoir; the readouts read both | 202 240 | 512 |
| `er_reservoir_grow.json` | `er_reservoir`, growing E-R layers on top of h1 up to 4; the readouts read every layer and the reservoir | 202 240 at the start, +18 176 per layer | 512 + 128 per layer |
| `er_ladder.json` | h1, h2, h3 (E-R); h3 reads h1 and h2; h2 and h3 come back to h1 through rungs of 32 relu neurons with habituation (no E-R); the reservoir; the readouts read h3 and the reservoir | 267 776 | 832 |

```
er_reservoir          eye, body ─► h1 ─► reservoir (frozen) ─┐
                                    └──────────────────────────► readouts

er_reservoir_grow     eye, body ─► h1 ─► h2 ─► h3 ─► h4      (h2.. added while it learns)
                                    │     └─────┴─────┴──────► readouts
                                    ├──────────────────────────►
                                    └─► reservoir (frozen) ───►

er_ladder             ┌──── r3 ◄──────────────┐   r2, r3: relu + habituation
                      │ ┌── r2 ◄──┐           │
                      ▼ ▼         │           │
              eye ──► h1 ───────► h2 ───────► h3 ──► readouts
              body    │                       ▲
                      └───────────────────────┘ (h3 reads h1 too)
                      └─► reservoir (frozen) ─────► readouts
```

`er_reservoir_grow` starts as exactly the `er_reservoir` network (same
seed, same weights), so the two runs differ only by growing.
`er_ladder` asks whether loops back to the first layer, through neurons
that do not fatigue, keep more of the past than the stack alone; the
rungs make cycles, so its update order is set explicitly (h1, h2, h3, the
rungs, the reservoir, the readouts) and saved in `best.exr`.

```bash
for m in er_reservoir er_reservoir_grow er_ladder; do
    python NNtesting/experiments/goe_rl/es.py --out results/goe-es/$m --workers 4 --generations 300 \
        --config NNtesting/experiments/goe_rl/models/$m.json
done
```

## Log

### G1. Untrained networks and the designed models (2026-10-01)

**Question.** Before any evolution: how do the plain network and the
three designed networks play untrained, against a random player, and do
the new parts work?

**Setup.** 10 worlds per network seed (seeds 10000 (s + 1) + 0..9, as the
nntest experiment), net seeds 0, 1, 2, 2-minute episodes. Plain: the
goe_rl defaults (one E-R layer, 128 neurons, no `seen` channel). The
random player draws a uniform action every move on the same worlds.

**Result.** Mean over the three seeds:

| Network | Reward | Score | Avatars lost | Spikes per move | Seconds per world |
|---------|--------|-------|--------------|-----------------|-------------------|
| plain | 7.7 | 7.7 | 0.00 | 2.2 | 1.0 |
| `er_reservoir` | 10.2 | 10.2 | 0.00 | 11.7 | 1.0 |
| `er_reservoir_grow` (start) | 10.2 | 10.2 | 0.00 | 11.7 | 1.0 |
| `er_ladder` | 8.6 | 8.6 | 0.00 | 18.4 | 1.1 |
| random player | 71.8 | 80.1 | 0.17 | | |

Checks of the new parts:
- The plain network is built and plays exactly as before the change
  (same spikes, 2.1533 per move, on the nntest worlds).
- Growing carries the weights over: on one world, 300 moves, the grown
  network's readouts equal the shallower network's (largest difference
  0, depth 1 → 2 → 3, for `er_reservoir_grow` and for the ladder with
  recurrence); with the same weights shuffled they differ by 0.34–0.39,
  so the check can see a wrong mapping. `es.py` grew a run twice and
  resumed it at depth 3.
- Every weight's recorded input matches the library's weight count in
  all three networks.

**Conclusion.**
- **Untrained, every network stands still.** It plays `MOVE_UP` in 93–99% of
  the moves (on world 10000, net seed 0: plain 741 of 750,
  `er_reservoir` 731, `er_ladder` 696): the E-R layers are nearly silent, the readouts stay near zero, and
  the tie goes to the first action. It scores only by the few cells it
  enters. A random player scores 7–9 times more and loses an avatar in
  about one game in six. Evolution has to start from there.
- The reservoir and the ladder make the network fire more (11.7 and 18.4
  spikes per move against 2.2) but not play better untrained.
- At about one second per world, a generation (54 worlds) takes about
  14 seconds with 4 workers, so 300 generations take a little over an
  hour per network.

Data: `results/goe/untrained_models.jsonl.gz` (one line per network and
seed).

### G2. Evolution on the event reward (2026-10-01)

**Question.** With a reward that pays for what the player does
(collecting, apples, use, doors, teleports, kills, mines) and charges
for harm (energy lost, avatars lost), does evolution teach the plain
E-R network to play, and does it beat a random player?

**Setup.** `reward` events with the default weights (+5 item, +20 apple,
+2 use, +10 door, +5 teleport, +10 kill, +5 mine, −0.2 per energy point
lost, −50 per avatar lost, +0.1 per score point). One E-R layer of 128
neurons, `evolve` all, net seed 0, 2-minute games, radius 6 (what `auto`
gives for 2 minutes). Two networks:
- `er_rec`: habituation on, and h1 reads itself (`recurrent`), which
  keeps it firing (about 31 spikes per move; without recurrence the
  plain network fires about 2, G1);
- `er_nohab`: habituation off, no recurrence (about 33 spikes per move).

`es.py` defaults (12 pairs, sigma 0.05, lr 0.02, 2 worlds per candidate,
6 validation worlds), 300 generations, 2 workers each, about 55 seconds
per generation. Then the untrained, best (best validation) and final
weights played 30 fresh worlds (seeds 4242–4271) next to a random
player (`results/goe/events/fresh.py`). The best weights reproduce
their logged validation exactly (47.68 and 27.05). Before this, two
runs on the G1 reward (score minus avatars lost) showed nothing in 35
generations and were stopped.

**Result.** Validation reward, 15-generation means: `er_rec` +7 at the
start, +15 by generations 31–45, +18 at 136–150, best single
generation +47.7. `er_nohab` hovered at +9 to +14 all along, best +27.
On the 30 fresh worlds:

| Player | Reward | Score | Items | Doors | Kills | Energy lost | Game over |
|--------|-------:|------:|------:|------:|------:|------------:|----------:|
| `er_rec` untrained | 10.1 | 51.8 | 1.50 | 0.77 | 0 | 10.5 | 10% |
| `er_rec` best | **18.4** | 69.6 | 1.90 | 1.03 | 0.03 | 10.5 | 10% |
| `er_rec` final | 17.3 | 55.9 | 1.97 | 1.20 | 0.03 | 10.5 | 13% |
| `er_nohab` untrained | 12.3 | 33.5 | 1.00 | 0.63 | 0 | 3.5 | 3% |
| `er_nohab` best | **18.9** | 47.7 | 1.50 | 0.77 | 0 | 0 | 3% |
| `er_nohab` final | 9.2 | 49.0 | 1.13 | 0.33 | 0 | 7.0 | 7% |
| random player | 13.8 | 71.7 | 1.87 | 1.03 | 0 | 7.0 | 13% |

No player used an item, teleported or set off a mine; an apple was
collected once in 30 games (`er_nohab` best).

**Conclusion.**
- **Evolution helps, a little, on this reward.** Both best networks
  beat a random player on fresh worlds (+18.4 and +18.9 against +13.8)
  and their untrained selves (+10.1, +12.3). With recurrence the gain
  holds to the end (final +17.3); without habituation and recurrence
  the final weights fall back to the untrained level (+9.2), so its best
  was partly luck of the validation worlds.
- **What it learned is to stay alive and open a door, not to fight.**
  The gains come from items and doors (the recurrent network now opens
  as many doors as the random player and collects as many items) and,
  for `er_nohab`, from taking no damage. Kills stay at about one in 30
  games, and teleports, apples, use and mines never happen: in 2-minute
  games these events are too rare for ES to see.
- **The score alone gave no signal** (two runs, 35 generations, no
  gain). Rewarding events gives ES something to climb, but slowly: 300
  generations for +8.

Data: `results/goe/events/` (per run: `config.json`, `log.jsonl.gz`,
`best.json`, `best_theta.npy`; `fresh.json` and `fresh.py` for the
fresh-world test).

## Open questions

- **Does evolution get past standing still?** Run the three designed
  networks for 300 generations (above) and compare against the random
  player and against each other on fresh worlds. G2: the plain network
  with recurrence does, slowly, on the event reward.
- **Rare events.** Kills, teleports, apples and mines almost never happen
  in 2-minute games, so ES cannot reward them (G2). Longer games, or
  worlds seeded with more monsters near the start, would show whether
  the network can learn them.
- **Growing.** Does `er_reservoir_grow` grow at all, and does it beat
  `er_reservoir`, which it starts as?
- **Memory.** Does the ladder (loops to h1 through non-fatiguing relu
  rungs) beat the stack? A maze rewards remembering where the player has
  been (+1 only for new cells).
- **Sight.** Does the `seen` channel help the network use its growing
  sight? Compare with `seen_channel` false.
- **Longer games.** The sight grows without a bound; longer episodes
  need a larger eye (`radius` `auto` follows `episode_ticks`).
