# snake-copy-init: grown neurons copy an existing neuron

Runs of `NNtesting/experiments/snake_growth/copy_init.py` (research log §35, doc/development.md
"Copy-init"). Defaults of `snake_growth/experiment.py` (recovery 0.9, increment 2, max width 16,
window 200, freeze old neurons, depth growth on plateau) unless the setup says otherwise; seeds
1-10; `one_*` start from ER(1), `four_*` from ER(4).

Arms: `random` = the current init (random incoming, zero outgoing); `copy_<source>` = WidthGrowth
init="copy" copying the most fatigued / active / read / a random neuron, noise 0.05, split 0.5;
`copy_nosplit` split 0 (incoming copied, readers unchanged); `copy_takeover` split 1;
`copy_noiseX` noise X (x the source's weight RMS). Columns: final = mean of the last three
50-game windows ± standard error over seeds; neurons = adaptive neurons at the end (all layers);
widths = width growths per run; change across a growth = score of the first whole window after a
width growth minus the window before it.

`runs.jsonl.gz`: one line per run (metrics, curve, sizes, event log, `copies` = (tick,
[(source, new)]) per copy growth, `games_run`).

## 1,500 games

| setup | seeds | final (last 3 windows) | mean of all windows | best window | neurons | widths | undos+prunes | change across a growth |
|---|---|---|---|---|---|---|---|---|
| one_fixed | 10 | 0.25 ± 0.08 | 0.19 | 0.70 | 1.0 | 0.0 | 0.0 | - |
| one_random | 10 | 3.91 ± 0.23 | 3.15 | 4.79 | 7.8 | 1.4 | 0.0 | +0.90 (n=4) |
| one_copy_fatigued | 10 | 3.31 ± 0.36 | 2.90 | 4.16 | 14.0 | 4.8 | 0.3 | +0.09 (n=19) |
| one_copy_active | 10 | 3.81 ± 0.29 | 3.16 | 4.43 | 13.4 | 4.3 | 0.1 | +0.85 (n=12) |
| one_copy_read | 10 | 2.76 ± 0.27 | 2.64 | 3.78 | 13.3 | 4.2 | 0.0 | +1.68 (n=12) |
| one_copy_random | 10 | 3.56 ± 0.35 | 3.13 | 4.42 | 13.4 | 4.3 | 0.1 | +1.21 (n=14) |
| one_copy_nosplit | 10 | 3.91 ± 0.32 | 3.32 | 4.61 | 14.2 | 4.7 | 0.1 | +0.36 (n=20) |
| one_copy_takeover | 10 | 3.20 ± 0.29 | 2.93 | 4.02 | 11.8 | 3.6 | 0.0 | +1.89 (n=11) |
| one_copy_noise02 | 10 | 3.11 ± 0.49 | 2.70 | 3.86 | 12.0 | 3.8 | 0.1 | +0.27 (n=9) |
| one_copy_noise05 | 10 | 3.24 ± 0.48 | 2.87 | 4.16 | 11.0 | 3.2 | 0.2 | +0.06 (n=10) |
| one_copy_noise10 | 10 | 3.55 ± 0.45 | 3.04 | 4.50 | 9.0 | 2.3 | 0.3 | +0.58 (n=8) |
| four_fixed | 10 | 5.32 ± 0.22 | 4.57 | 6.59 | 4.0 | 0.0 | 0.0 | - |
| four_random | 10 | 5.62 ± 0.22 | 4.67 | 6.69 | 8.4 | 0.5 | 0.1 | -0.24 (n=5) |
| four_copy_fatigued | 10 | 5.57 ± 0.22 | 4.63 | 6.74 | 9.6 | 0.9 | 0.1 | -0.21 (n=8) |
| four_copy_active | 10 | 5.59 ± 0.23 | 4.69 | 6.69 | 10.4 | 1.6 | 0.2 | -0.17 (n=15) |
| four_copy_read | 10 | 5.40 ± 0.22 | 4.59 | 6.57 | 9.0 | 0.8 | 0.3 | +0.33 (n=7) |
| four_copy_random | 10 | 5.68 ± 0.23 | 4.69 | 6.68 | 9.6 | 1.1 | 0.1 | +0.33 (n=7) |
| four_copy_nosplit | 10 | 5.64 ± 0.27 | 4.67 | 6.54 | 9.6 | 1.1 | 0.3 | +0.40 (n=9) |
| four_copy_takeover | 10 | 5.71 ± 0.22 | 4.66 | 6.69 | 10.2 | 1.5 | 0.2 | -0.86 (n=14) |
| four_copy_noise02 | 10 | 5.65 ± 0.22 | 4.70 | 6.67 | 9.0 | 0.8 | 0.3 | +0.05 (n=8) |
| four_copy_noise05 | 10 | 5.43 ± 0.32 | 4.48 | 6.43 | 10.0 | 1.1 | 0.1 | -0.09 (n=11) |
| four_copy_noise10 | 10 | 5.58 ± 0.24 | 4.62 | 6.66 | 9.0 | 0.6 | 0.1 | +0.11 (n=6) |

## 5,000 games

| setup | seeds | final (last 3 windows) | mean of all windows | best window | neurons | widths | undos+prunes | change across a growth |
|---|---|---|---|---|---|---|---|---|
| one_random | 10 | 3.91 ± 0.30 | 3.61 | 5.18 | 8.0 | 1.7 | 0.2 | +0.89 (n=7) |
| one_copy_fatigued | 10 | 3.23 ± 0.33 | 3.12 | 4.34 | 15.8 | 6.2 | 0.8 | +0.09 (n=34) |
| one_copy_active | 10 | 3.73 ± 0.37 | 3.50 | 4.85 | 14.8 | 5.3 | 0.4 | +0.34 (n=21) |
| one_copy_nosplit | 10 | 3.87 ± 0.37 | 3.63 | 4.89 | 15.6 | 5.8 | 0.5 | +0.23 (n=31) |
| one_copy_noise10 | 10 | 3.45 ± 0.35 | 3.31 | 4.63 | 10.4 | 3.5 | 0.8 | +0.12 (n=20) |
| four_random | 10 | 5.71 ± 0.20 | 5.33 | 7.08 | 9.4 | 1.3 | 0.6 | -0.43 (n=12) |
| four_copy_fatigued | 10 | 5.62 ± 0.23 | 5.25 | 7.16 | 11.0 | 1.9 | 0.4 | -0.23 (n=19) |
| four_copy_active | 10 | 5.57 ± 0.19 | 5.30 | 7.14 | 13.0 | 3.0 | 0.5 | -0.11 (n=30) |

## Paired by seed (copy arm minus random, final score, mean ± s.e. over 10 seeds)

| arm | ER(1), 1,500 | ER(1), 5,000 | ER(4), 1,500 | ER(4), 5,000 |
|---|---|---|---|---|
| copy_fatigued | -0.60 ± 0.39 | -0.68 ± 0.38 | -0.05 ± 0.10 | -0.09 ± 0.13 |
| copy_active | -0.10 ± 0.39 | -0.18 ± 0.49 | -0.03 ± 0.06 | -0.15 ± 0.08 |
| copy_read | -1.15 ± 0.36 | | -0.23 ± 0.17 | |
| copy_random | -0.35 ± 0.42 | | +0.06 ± 0.19 | |
| copy_nosplit | -0.00 ± 0.48 | -0.04 ± 0.57 | +0.02 ± 0.27 | |
| copy_takeover | -0.71 ± 0.38 | | +0.08 ± 0.14 | |
| copy_noise02 | -0.80 ± 0.55 | | +0.02 ± 0.16 | |
| copy_noise05 | -0.67 ± 0.49 | | -0.20 ± 0.27 | |
| copy_noise10 | -0.36 ± 0.39 | -0.46 ± 0.37 | -0.04 ± 0.07 | |
