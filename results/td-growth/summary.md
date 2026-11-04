# Snake: TD-error growth, shadow candidates, significance pruning, synaptic scaling (research §69)

`NNtesting/experiments/snake_growth/td_growth.py` (defaults: recovery 0.5, width only, trigger td), 3,000 games, seeds 1-20, `max_width=64` active neurons. last300 = mean apples per game over the last 300 training games (exploration on); test = 100 greedy games after training. ± = standard error over the 20 seeds. Settings list what differs from td_growth.py's defaults. Raw rows: `runs.jsonl.gz` (one per run, with curves, sizes, critic error per window, growth events). Commands: `run.sh`.

| arm | settings | last300 | test | active | layer | grows | undos | mutes | restores | scales | splits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed1 | start=1, grow=False | 0.38 ± 0.07 | 0.21 ± 0.11 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fixed4 | grow=False | 7.25 ± 0.36 | 10.53 ± 0.79 | 4.0 | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fixed16 | start=16, grow=False | 9.81 ± 0.08 | 13.21 ± 0.30 | 16.0 | 16.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fixed32 | start=32, grow=False | 9.65 ± 0.09 | 13.09 ± 0.27 | 32.0 | 32.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fixed64 | start=64, grow=False | 9.77 ± 0.10 | 13.57 ± 0.22 | 64.0 | 64.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tired | start=1, trigger=saturation | 7.83 ± 0.22 | 10.40 ± 0.76 | 5.0 | 5.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| td_rand | start=1 | 9.41 ± 0.09 | 12.89 ± 0.36 | 44.1 | 44.1 | 12.9 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| timer | start=1, trigger=timer | 9.89 ± 0.06 | 13.82 ± 0.22 | 64.0 | 64.0 | 16.8 | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| td_cand | start=1, candidates=8 | 9.16 ± 0.10 | 12.54 ± 0.37 | 47.0 | 55.0 | 13.5 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| td_cand0 | start=1, candidates=8, candidate_rate=0.0 | 9.51 ± 0.11 | 13.01 ± 0.22 | 41.7 | 49.7 | 12.3 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| td_cand_sig | start=1, candidates=8, candidate_rate=0.0, significance=True, memory=80 | 9.60 ± 0.08 | 12.54 ± 0.36 | 42.6 | 50.7 | 13.7 | 1.5 | 12.4 | 4.7 | 0.0 | 0.0 |
| timer_sig | start=1, trigger=timer, significance=True | 9.51 ± 0.10 | 13.04 ± 0.24 | 52.1 | 64.0 | 22.9 | 0.0 | 20.6 | 8.8 | 0.0 | 0.0 |
| over64_sig | start=64, minimum=1, trigger=none, significance=True | 9.71 ± 0.08 | 13.42 ± 0.31 | 62.6 | 64.0 | 0.0 | 0.0 | 2.1 | 0.7 | 0.0 | 0.0 |
| over64_hard | start=64, minimum=1, trigger=none, significance=True, sig_every=1000, sig_fraction=0.7 | 9.76 ± 0.11 | 13.43 ± 0.36 | 37.4 | 64.0 | 0.0 | 0.0 | 42.9 | 16.2 | 0.0 | 0.0 |
| over64_harder | start=64, minimum=1, trigger=none, significance=True, sig_every=500, sig_fraction=1.0, sig_tolerance=0.2 | 9.82 ± 0.11 | 12.95 ± 0.19 | 29.4 | 64.0 | 0.0 | 0.0 | 35.8 | 1.1 | 0.0 | 0.0 |
| sat1_l3 | start=1, trigger=none, saturation=True, weight_limit=3.0 | 0.37 ± 0.06 | 0.23 ± 0.11 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 |
| sat16_l3 | start=16, trigger=none, saturation=True, weight_limit=3.0 | 9.82 ± 0.13 | 13.78 ± 0.22 | 16.0 | 16.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 | 0.0 |
| td_sat_l3 | start=1, saturation=True, weight_limit=3.0 | 9.41 ± 0.09 | 12.89 ± 0.36 | 44.1 | 44.1 | 12.9 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| sat1_l15 | start=1, trigger=none, saturation=True, weight_limit=1.5 | 1.90 ± 0.63 | 2.09 ± 0.80 | 1.9 | 1.9 | 0.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.8 |
| sat4_l15 | trigger=none, saturation=True, weight_limit=1.5 | 7.82 ± 0.17 | 10.01 ± 0.68 | 4.7 | 4.7 | 0.0 | 0.0 | 0.0 | 0.0 | 15.9 | 0.7 |

## Paired comparisons (difference of means ± combined standard error, last300 / test)

| comparison | last300 | test |
|---|---|---|
| td_rand − tired (new trigger vs the tiredness trigger, both from ER(1)) | +1.58 ± 0.24 | +2.49 ± 0.84 |
| td_rand − fixed16 | −0.40 ± 0.12 | −0.32 ± 0.47 |
| timer − td_rand (growth on a clock vs on the error plateau) | +0.48 ± 0.11 | +0.93 ± 0.42 |
| timer − fixed16 | +0.08 ± 0.10 | +0.61 ± 0.37 |
| td_cand − td_cand0 (candidates trained on δ vs untrained, picked at random) | −0.35 ± 0.15 | −0.47 ± 0.43 |
| td_cand0 − td_rand (pool vs plain random growth) | +0.10 ± 0.14 | +0.12 ± 0.42 |
| over64_hard − fixed64 (64 pruned to 37 active) | −0.01 ± 0.15 | −0.14 ± 0.42 |
| over64_harder − fixed64 (64 pruned to 29 active) | +0.05 ± 0.15 | −0.62 ± 0.29 |
| timer_sig − timer | −0.38 ± 0.12 | −0.78 ± 0.33 |
| sat4_l15 − fixed4 (scaling + split at limit 1.5) | +0.57 ± 0.40 | −0.52 ± 1.04 |
| sat1_l15 − fixed1 | +1.51 ± 0.63 | +1.87 ± 0.81 |

Critic error (unexplained share of the TD target) per 2,000-tick window, td_rand, mean of 20 seeds per 10 windows: 0.65, 0.78, 0.76, 0.74, 0.74, 0.74, 0.73, 0.73; window-to-window s.d. 0.10. It never falls while the layer grows from 1 to 44 neurons, so the plateau test (2% progress) passes at nearly every chance: the trigger behaves like a clock with back-off after undos. Mean |correlation| with δ of promoted candidates: 0.076 trained, 0.042 untrained.

Before these runs: no weight reaches the ±10 clamp in snake_growth after 1,500 games (largest hidden |w| 4.7 at ER(16) rec 0.5, 6.3 at rec 0.9, 4.2 growing from ER(1); readouts below 0.8). And in `td_actor_critic/snake_td.py` (critic and actor on state + frozen mix), the mix size does not matter: 0, 4, 16 and 64 mix neurons all reach 18-20.5 greedy test apples (seeds 1-2, 1,000 games), so that set-up cannot test growth.
