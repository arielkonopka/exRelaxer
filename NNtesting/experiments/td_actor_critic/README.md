# td_actor_critic

An actor-critic with local three-factor learning, the user's spec of
2026-10-30: a linear **TD(λ) critic** learns to predict future reward V(s),
and its prediction error

    δ_t = r_t + γ·V(s_t) − V(s_{t−1})        (V = 0 at a terminal step)

replaces the raw reward as the modulator of the actor's readouts: one
global scalar, the analogue of phasic dopamine, so learning stays local
(pre × post × δ) but is no longer sparse or late. Results: research log
§41 (phase 2, drives: §42), Doom log D17 (D18).

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
