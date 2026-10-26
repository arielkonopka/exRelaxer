# snake-grown-threshold: width-grown neurons that start easier to fire

Runs of `NNtesting/experiments/snake_growth/copy_init.py` (research log §38). Setups as
results/snake-copy-init/ (§35): `snake_growth` defaults (recovery 0.9, increment 2, max width 16,
window 200, old neurons frozen, depth growth on plateau), `one_*` from ER(1), `four_*` from ER(4);
`one05t_*` from ER(1) at recovery 0.5 with §37's retuned trigger (25% saturated ticks, 4 neurons
per growth). Arms: `rest<x>` = the grown neurons' E-R resting threshold is the layer's × 0.x
(`grown_resting`; rest01 = × 0.1, rest025 = × 0.25, ...); `start0` = their threshold is set to the
minimum right after growth (`grown_start 0`); `random_*` / `copy_*` = random or copy init (§35).
The `random`, `copy_fatigued`, `copy_active` rows of `one_*` / `four_*` are §35's runs (same code).

Paired by seed against random init with the current threshold (`random`): Δ final = last three
50-game windows, Δ all = mean over every window (learning speed), mean ± s.e.

## 1,500 games, seeds 1-10, recovery 0.9

| arm | ER(1) final | Δ final | Δ all | ER(4) final | Δ final | Δ all |
|---|---|---|---|---|---|---|
| random | 3.91 | | | 5.62 | | |
| random_start0 | 4.03 | +0.12 ± 0.28 | +0.07 ± 0.15 | 5.62 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| random_rest05 | 3.57 | -0.33 ± 0.31 | -0.27 ± 0.24 | 5.48 | -0.14 ± 0.13 | -0.01 ± 0.01 |
| random_rest01 | 3.48 | -0.43 ± 0.26 | -0.29 ± 0.23 | 5.51 | -0.11 ± 0.13 | -0.01 ± 0.01 |
| random_rest01_start0 | 3.48 | -0.43 ± 0.26 | -0.29 ± 0.23 | 5.51 | -0.11 ± 0.13 | -0.01 ± 0.01 |
| copy_fatigued | 3.31 | -0.60 ± 0.39 | -0.25 ± 0.31 | 5.57 | -0.05 ± 0.10 | -0.04 ± 0.03 |
| copy_fatigued_start0 | 3.24 | -0.66 ± 0.44 | -0.24 ± 0.33 | 5.64 | +0.02 ± 0.15 | +0.00 ± 0.03 |
| copy_fatigued_rest01 | 2.92 | -0.98 ± 0.34 | -0.63 ± 0.27 | 5.60 | -0.02 ± 0.21 | -0.03 ± 0.02 |
| copy_active | 3.81 | -0.10 ± 0.39 | +0.01 ± 0.32 | 5.59 | -0.03 ± 0.06 | +0.02 ± 0.06 |
| copy_active_start0 | 3.56 | -0.35 ± 0.39 | -0.07 ± 0.32 | 5.54 | -0.08 ± 0.17 | +0.01 ± 0.07 |
| copy_active_rest01 | 3.45 | -0.46 ± 0.54 | -0.13 ± 0.45 | 5.48 | -0.14 ± 0.21 | -0.03 ± 0.02 |

## ER(1), recovery 0.5, retuned trigger

| arm | seeds | games | final | neurons | Δ final | Δ all |
|---|---|---|---|---|---|---|
| random | 1-30 | 1,500 | 7.13 | 9.5 | | |
| random_start0 | 1-10 | 1,500 | 7.55 | 9.8 | +0.26 ± 0.15 | +0.12 ± 0.11 |
| random_rest075 | 1-30 | 1,500 | 7.51 | 9.3 | +0.39 ± 0.13 | +0.24 ± 0.07 |
| random_rest05 | 1-30 | 1,500 | 7.45 | 9.1 | +0.32 ± 0.14 | +0.40 ± 0.08 |
| random_rest025 | 1-30 | 1,500 | 7.40 | 9.1 | +0.27 ± 0.18 | +0.39 ± 0.10 |
| random_rest01 | 1-30 | 1,500 | 7.22 | 9.4 | +0.09 ± 0.14 | +0.29 ± 0.09 |
| random_rest01_start0 | 1-10 | 1,500 | 7.60 | 9.4 | +0.32 ± 0.26 | +0.54 ± 0.12 |
| copy_fatigued | 1-10 | 1,500 | 7.48 | 14.2 | +0.19 ± 0.34 | +0.02 ± 0.25 |
| copy_fatigued_rest01 | 1-10 | 1,500 | 7.61 | 11.4 | +0.33 ± 0.30 | +0.45 ± 0.18 |
| copy_active | 1-10 | 1,500 | 7.42 | 13.4 | +0.13 ± 0.40 | -0.09 ± 0.30 |
| copy_active_rest01 | 1-30 | 1,500 | 7.30 | 12.9 | +0.17 ± 0.21 | +0.33 ± 0.14 |
| random | 1-10 | 5,000 | 7.28 | 11.4 | | |
| random_rest05 | 1-10 | 5,000 | 7.98 | 11.4 | +0.70 ± 0.17 | +0.55 ± 0.09 |
| random_rest01 | 1-10 | 5,000 | 7.50 | 11.4 | +0.22 ± 0.18 | +0.39 ± 0.19 |

Share of ticks on which a neuron of the trained ER(1) layer can learn (feedback alignment needs
E-R eligibility: threshold above its resting value), 20 test games, seeds 1-6, recovery 0.5
retuned: random base 0.88 / grown 0.88; rest05 0.86 / 0.97; rest01 0.88 / 1.00. Share of ticks
with a non-zero output: grown 0.57 (random) vs 0.59 (rest05).

`runs.jsonl.gz`: one line per run (as results/snake-copy-init/, plus `games_run`).
