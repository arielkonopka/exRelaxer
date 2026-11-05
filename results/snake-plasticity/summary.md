# snake-plasticity: a plasticity potential per neuron

Runs of `NNtesting/experiments/snake_growth/plasticity.py` (one task) and `switch.py` (task
switch), research log §73. Setup: `snake_growth`, recovery 0.5; growth arms start from ER(1)
(§37's trigger: 25% saturated ticks, 4 neurons per growth), fixed arms are ER(16) without growth.
Arms (`ARMS` in plasticity.py):

- `fixed16`: ER(16), random feedback alignment (FA); `grow`: growth, older neurons frozen at
  growth; `grow_nofreeze`: nothing frozen; `young_nofreeze`: §39's best (grown neurons rest at
  × 0.5 with 3 × the learning gain for good, nothing frozen);
- `P_a<alpha>_e<epsilon>`: plasticity potential on the E-R layer (`with_plasticity(alpha, 1,
  epsilon)`; a01 = 0.01, a001 = 0.001, e01 = 0.1, e001 = 0.01), nothing frozen at growth;
  `_readouts`: the readouts too; `Pp_`: with participation (error × min(1, |output|));
  `wfb`: weight feedback (`with_weight_feedback`: the error through the readouts' real weights).

With P on, steps where the readout agrees with the reward apply a zero error (so P can fade)
instead of being skipped. `single_runs.jsonl.gz` / `switch_runs.jsonl.gz`: one JSON object per
run (single: per-window curve and mean P; switch: the scores below, the curves and every
neuron's P after A and after B).

## One task: 3,000 games, seeds 1-30

Δ paired by seed against `fixed16`, last three 50-game windows (final) or every window (all).

| arm | seeds | final | Δ final vs fixed16 | Δ all vs fixed16 | neurons | widths | mean P (last 3) | share P < 0.1 |
|---|---|---|---|---|---|---|---|---|
| fixed16 | 30 | 9.73 |  |  | 16.0 | 0.0 | - | - |
| fixed16_P_a01_e01 | 30 | 8.93 | -0.80 ± 0.13 | -0.89 ± 0.06 | 16.0 | 0.0 | 0.08 | 0.75 |
| fixed16_P_a001_e001 | 30 | 9.97 | +0.25 ± 0.12 | -0.00 ± 0.04 | 16.0 | 0.0 | 0.53 | 0.01 |
| grow | 30 | 7.14 | -2.58 ± 0.20 | -2.61 ± 0.17 | 10.6 | 1.7 | - | - |
| grow_nofreeze | 30 | 7.77 | -1.96 ± 0.14 | -2.00 ± 0.11 | 10.2 | 1.5 | - | - |
| young_nofreeze | 30 | 7.86 | -1.86 ± 0.20 | -1.60 ± 0.13 | 11.8 | 2.0 | - | - |
| P_a001_e001 | 30 | 7.64 | -2.09 ± 0.16 | -2.05 ± 0.13 | 10.1 | 1.6 | 0.78 | 0.00 |
| P_a01_e001 | 30 | 7.42 | -2.30 ± 0.21 | -2.50 ± 0.17 | 10.2 | 1.4 | 0.35 | 0.07 |
| P_a001_e01 | 30 | 7.47 | -2.26 ± 0.16 | -2.25 ± 0.16 | 10.5 | 1.7 | 0.54 | 0.04 |
| P_a01_e01 | 30 | 6.34 | -3.39 ± 0.24 | -3.39 ± 0.23 | 9.8 | 1.4 | 0.20 | 0.47 |
| P_a01_e01_readouts | 30 | 7.41 | -2.32 ± 0.29 | -2.69 ± 0.26 | 9.8 | 1.3 | 0.15 | 0.55 |
| fixed16_Pp_a01_e01 | 30 | 8.67 | -1.06 ± 0.16 | -1.22 ± 0.08 | 16.0 | 0.0 | 0.07 | 0.79 |
| fixed16_Pp_a001_e001 | 30 | 9.58 | -0.15 ± 0.14 | -0.10 ± 0.04 | 16.0 | 0.0 | 0.35 | 0.03 |
| fixed16_Pp_a01_e01_readouts | 30 | 9.51 | -0.21 ± 0.15 | -1.25 ± 0.15 | 16.0 | 0.0 | 0.01 | 0.95 |
| Pp_a01_e01 | 30 | 5.04 | -4.68 ± 0.35 | -4.95 ± 0.30 | 11.0 | 1.8 | 0.17 | 0.52 |
| Pp_a001_e001 | 30 | 7.57 | -2.15 ± 0.15 | -2.27 ± 0.12 | 9.9 | 1.5 | 0.59 | 0.01 |
| Pp_a01_e01_readouts | 30 | 4.51 | -5.21 ± 0.47 | -5.48 ± 0.40 | 10.2 | 1.6 | 0.09 | 0.70 |
| fixed16_wfb | 30 | 9.67 | -0.06 ± 0.15 | +0.16 ± 0.05 | 16.0 | 0.0 | - | - |
| fixed16_wfb_P_a01_e01 | 30 | 9.56 | -0.17 ± 0.12 | -0.26 ± 0.05 | 16.0 | 0.0 | 0.05 | 0.80 |
| fixed16_wfb_P_a001_e001 | 30 | 9.99 | +0.26 ± 0.10 | +0.21 ± 0.06 | 16.0 | 0.0 | 0.42 | 0.02 |
| wfb_nofreeze | 30 | 8.18 | -1.54 ± 0.15 | -1.48 ± 0.10 | 10.3 | 1.4 | - | - |
| wfb_P_a01_e01 | 30 | 7.17 | -2.55 ± 0.16 | -2.43 ± 0.10 | 9.8 | 1.3 | 0.13 | 0.61 |
| wfb_P_a001_e001 | 30 | 8.09 | -1.63 ± 0.16 | -1.52 ± 0.09 | 9.8 | 1.3 | 0.65 | 0.00 |

## Task switch: seeds 1-20

A = Snake, 3,000 learning games; B = the same Snake with the 23 sensors in a fixed random order
(per seed), 1,500 learning games; then A again, 500 learning games. "A before / after B" and
"B (learned)": 100 greedy test games without learning. Δ after: paired against `fixed16`.

| arm | seeds | A before | A after B | retention | B (learned) | A back, first window | A back, final | Δ after vs fixed16 | mean P after A | mean P after B | neurons |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed16 | 20 | 12.63 | 0.54 | 0.04 | 12.54 | 2.28 | 8.42 |  | nan | nan | 16.0 |
| fixed16_P_a01_e01 | 20 | 12.58 | 1.10 | 0.09 | 11.48 | 2.69 | 6.83 | +0.57 ± 0.28 | 0.07 | 0.15 | 16.0 |
| fixed16_P_a001_e001 | 20 | 13.35 | 0.68 | 0.05 | 13.18 | 2.61 | 8.68 | +0.14 ± 0.12 | 0.56 | 0.62 | 16.0 |
| grow | 20 | 10.90 | 0.58 | 0.06 | 8.12 | 1.42 | 4.77 | +0.04 ± 0.15 | nan | nan | 5.0 |
| grow_nofreeze | 20 | 10.88 | 0.78 | 0.10 | 9.44 | 1.23 | 5.01 | +0.24 ± 0.18 | nan | nan | 5.4 |
| young_nofreeze | 20 | 11.45 | 0.70 | 0.08 | 7.99 | 0.90 | 6.19 | +0.16 ± 0.09 | nan | nan | 5.0 |
| P_a001_e001 | 20 | 10.79 | 0.75 | 0.07 | 8.86 | 1.33 | 4.59 | +0.21 ± 0.19 | 0.73 | 0.83 | 5.4 |
| P_a01_e001 | 20 | 9.99 | 1.10 | 0.14 | 8.02 | 1.36 | 4.75 | +0.57 ± 0.30 | 0.32 | 0.40 | 5.2 |
| P_a001_e01 | 20 | 10.47 | 0.72 | 0.08 | 8.68 | 1.00 | 4.67 | +0.18 ± 0.16 | 0.54 | 0.65 | 5.0 |
| P_a01_e01 | 20 | 9.69 | 1.05 | 0.16 | 6.31 | 1.28 | 3.91 | +0.51 ± 0.26 | 0.14 | 0.20 | 5.2 |
| P_a01_e01_readouts | 20 | 12.06 | 1.02 | 0.08 | 8.31 | 1.31 | 4.63 | +0.48 ± 0.24 | 0.10 | 0.19 | 5.2 |
| fixed16_Pp_a01_e01 | 20 | 11.86 | 0.63 | 0.05 | 10.57 | 3.55 | 6.92 | +0.09 ± 0.13 | 0.05 | 0.09 | 16.0 |
| fixed16_Pp_a001_e001 | 20 | 13.66 | 0.68 | 0.05 | 12.84 | 3.03 | 8.54 | +0.14 ± 0.11 | 0.35 | 0.42 | 16.0 |
| fixed16_Pp_a01_e01_readouts | 20 | 13.24 | 0.54 | 0.04 | 11.14 | 2.66 | 8.11 | +0.01 ± 0.13 | 0.02 | 0.02 | 16.0 |
| Pp_a01_e01 | 20 | 8.84 | 1.13 | 0.17 | 4.27 | 1.93 | 3.87 | +0.59 ± 0.19 | 0.12 | 0.19 | 5.4 |
| Pp_a001_e001 | 20 | 11.59 | 0.89 | 0.08 | 8.25 | 1.04 | 4.77 | +0.35 ± 0.16 | 0.52 | 0.62 | 5.0 |
| Pp_a01_e01_readouts | 20 | 10.64 | 1.14 | 0.10 | 6.75 | 2.43 | 4.85 | +0.60 ± 0.33 | 0.03 | 0.08 | 5.0 |
| fixed16_wfb | 20 | 13.65 | 0.38 | 0.03 | 13.59 | 8.02 | 9.64 | -0.16 ± 0.07 | nan | nan | 16.0 |
| fixed16_wfb_P_a01_e01 | 20 | 13.42 | 0.34 | 0.03 | 11.31 | 8.16 | 9.48 | -0.20 ± 0.09 | 0.04 | 0.04 | 16.0 |
| fixed16_wfb_P_a001_e001 | 20 | 13.34 | 0.16 | 0.01 | 13.33 | 7.90 | 9.64 | -0.38 ± 0.11 | 0.42 | 0.44 | 16.0 |
| wfb_nofreeze | 20 | 11.17 | 0.20 | 0.02 | 10.21 | 4.27 | 7.88 | -0.34 ± 0.12 | nan | nan | 5.0 |
| wfb_P_a01_e01 | 20 | 10.53 | 0.12 | 0.01 | 9.31 | 3.37 | 7.12 | -0.42 ± 0.16 | 0.18 | 0.24 | 5.0 |
| wfb_P_a001_e001 | 20 | 11.48 | 0.13 | 0.01 | 11.01 | 4.14 | 7.97 | -0.41 ± 0.13 | 0.72 | 0.77 | 5.0 |
