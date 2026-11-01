# td_actor_critic

An actor-critic with local three-factor learning, the user's spec of
2026-10-30: a linear **TD(λ) critic** learns to predict future reward V(s),
and its prediction error

    δ_t = r_t + γ·V(s_t) − V(s_{t−1})        (V = 0 at a terminal step)

replaces the raw reward as the modulator of the actor's readouts: one
global scalar, the analogue of phasic dopamine, so learning stays local
(pre × post × δ) but is no longer sparse or late. Results: research log
§41 (drives §42, place cells §43, novelty and learned place §44, door reflex and path integration §45, boredom and anchoring §46, killer instinct §47, guided landmarks §48, exit §49, learned threat detector §50, detector feedback §51-§52, colour §53, practice §54, paths §55), Doom log D17-D31.

```
input -> frozen features (Gabor + centre-surround bank, max-pooled)
      -> frozen random E-R reservoir (reads itself one tick late)
           features = reservoir outputs + thresholds (+ the frozen features)
             |-> CRITIC  V(s) = w·x + b       e ← γλe + x,          w += α_c δ e
             |-> ACTOR   one readout per action, E ← γλE + post ⊗ x,  W += α_a δ E
```

The learner is `exrelaxer.actor_critic` (`Critic`, `Actor`, `Agent`), pure
Python over any feature vector, so it can sit on any frozen network.

## Files

| File | What |
|---|---|
| `common.py` | `Reservoir`: optional frozen early vision, the frozen E-R layer, its features |
| `chain.py` | step 1: cue, then reward N steps later; the "dopamine shift" |
| `snake_td.py` | step 2: Snake, δ vs raw reward vs the original sign-rule snake |
| `doom_td.py` | steps 3–4: ViZDoom `basic`, `defend_the_center`; whole maps (`map01`) |

```bash
python3 chain.py --out runs/chain --repr csc,er --recovery 0.745 --spread 0.245 --wait-max 40 --trials 1500
python3 snake_td.py --out td.jsonl --arm td --seeds 11-20 --set post=advantage --set critic_rate=0.03 --set actor_rate=0.03 --set tau=0.03
python3 doom_td.py --out runs/basic_td --scenario basic --arm td --episodes 3000 --set critic_rate=0.1 \
    --set center=0.001 --set actor_rate=0.03 --set features=both_pool
```

`--set until_death=M` plays each game until death (no time limit, a cap
of M minutes) and logs survival seconds; `--scenario map01` plays the
whole level with doom_rl's 8 buttons and shaped reward; `--checkpoint N`
saves the learner every N games and resumes from it after a restart.

Phase 2, homeostatic drives (`--scenario health_gathering`): `--set drives=1`
feeds health / 100 into the reservoir (weights × `drive_gain`) and into the
features; `--set reward=drive` replaces the scenario's reward by drive
reduction r = D(h_prev) − D(h_now), D(h) = |target − h| / 100
(`reward=drive_level`: −D(h) × `drive_cost` every step). The log counts the
medkits picked up.

Phase 3, a place layer (`--scenario my_way_home`, or `map01`): `--set place=1`
appends place cells (the map position tile-coded: `tilings` grids of
`place_size`-unit squares, hashed into `place_bins`) and `headings`
head-direction cells to the features; `place_input=1` also feeds them into
the reservoir; `features=place` uses them alone. The log counts the 64-unit
squares visited; the test reports `success` (the `my_way_home` vest found).

Phase 4: `--set novelty=β` adds β / √(1 + N) for entering a 64-unit square
entered N times before in the agent's life (training only); `motion=1`
feeds self-motion (forward and sideways speed, turning) into the
reservoir; `vplace=K` appends K place cells learned by competitive
learning from the reservoir and vision features (`vplace_settle`: a
falling rate); the test reports `field_spread` (1 = no place information).

Phase 5: `--set door_reflex=1` adds USE to the chosen action after a step
that tried to move and was blocked; `--set path=1` gives the place cells a
pose integrated from self-motion from the spawn (`path_noise` relative
noise per step) instead of the game's coordinates.

Phase 6 (whole maps): `--set approach=0.1` pays doom_rl's exit-approach
reward; `--set boredom=c` adds a cost that grows to c per step over
`boredom_steps` steps without a new square (felt as an input); `--set
path=2` anchors path integration on recognised views (`landmarks`,
`landmark_match`, `landmark_seen`, `landmark_agree`, `landmark_pull`).

Phases 7-8: `--set attack_reflex=1` adds ATTACK when a monster (ViZDoom's
labels) is in the middle `attack_cone` of the view, `orient_reflex=1` a
turn toward the biggest monster on screen, `kill_weight` sets doom_rl's
kill reward; `landmark_seq`, `landmark_spread` and `landmark_guided` vary
the landmarks of `path=2`.

Phase 10: `--set detector=N` teaches a per-column monster detector over
the frozen vision features from the labels for N games; afterwards (and in
test games) it, not the labels, drives the reflexes (`detector_hidden`,
`detector_neighbours`, `detector_replay`, `detector_threshold`). Phase 11:
`detector_feedback=1` keeps it learning from whether its shots hit,
`detector_evidence=d` puts a leaky accumulator before the reflexes.
Phase 12: `detector_feedback=2` also judges turns (a hit within
`detector_turn_steps`). Phase 13: `detector_colour=1` adds colour-opponent
features (the game runs in RGB; the rest of the agent sees gray).
Phases 14-15: `detector_steer=M` lets the detector drive the reflexes from
game M while the labels still teach; `detector_freeze_stats=1` stops its
input normalisation with its weights; `detector_holdout=1` learns only in
every other childhood game, to measure it on held-out views.

`where_agents_go.py --runs DIR ... --games N --out paths.png [--record videos]`
replays trained MAP01 runs and draws where they go (phase 17); `--record`
keeps three videos (most kills, longest play, most doors).
`detector_probe.py` records random games and compares detectors offline
and online.

`doom_td.py --arm reward` is the control with the raw reward as the
modulator (the critic still learns and is logged); `--arm random` plays
random actions. Each run writes `config.json`, `log.jsonl` (per training
episode: return, kills, length, mean |δ|, mean δ, mean V, action entropy,
share of sampled actions that were not the greedy one, reservoir spikes per
step, share of actor weights at the ±10 clamp), `values_<episode>.json`
(V(s) and δ over one episode, every `trace_every` episodes) and `test.json`
(greedy episodes on fresh seeds, no learning).

## What had to change from the spec

- **post = onehot − π**, the softmax policy gradient (`post=advantage`). With
  post = 1 for the taken action only, the policy collapses within ~40 Snake
  games (5.8 apples instead of 16.7).
- **Time limits are not terminal.** In `basic` (−1 per tic) the critic
  expects V ≈ −1.3; a timeout scored as V = 0 gives δ ≈ +0.9, a reward for
  running out the clock, and the actor learns to stop shooting. The last
  state bootstraps instead.
- **Centred features** (`center=0.001`: each feature minus its running
  mean). E-R thresholds rest well above 0; uncentred they inflate |x|² and
  the normalised steps become too small for the critic to learn even the
  mean value before the actor collapses.
- **Frozen early vision.** Neither raw pixels nor the E-R reservoir reading
  them tell a linear readout where the monster is; with an oracle input
  (the monster's screen column) the same learner solves `basic`, and a
  frozen Gabor + centre-surround bank does it from pixels.
- **E-R recovery spread** (`spread`): with one shared recovery every
  threshold decays at the same rate and V cannot tell how long ago the cue
  was; recoveries 0.5–0.99 extend the chain from N = 10 to N = 20.
- Only actor rate / τ matters (linear readouts starting at 0). The critic
  diverges when rate·|x|²/(1 − γλ) exceeds about 1, so critic rate ≤ 0.1
  at γλ = 0.87 (the normalisation divides by a decaying peak of |x|²).
