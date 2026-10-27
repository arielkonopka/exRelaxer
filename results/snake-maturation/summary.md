# snake-maturation: a critical period for grown neurons

Runs of `NNtesting/experiments/snake_growth/maturation.py` (research log §39). Setup: §37's best
one-cell start (`snake_growth` from ER(1), recovery 0.5, growth on 25% saturated ticks, 4 neurons
per growth), 1,500 games (seeds 1-30) and 5,000 games (seeds 1-20). Arms (`ARMS` in the script):

- `random`: the current init; `rest05`: §38, grown neurons rest at × 0.5 for good;
- `gain3`: grown neurons learn with 3 × the learning gain for good; `rest05_gain3`: both;
- `mat_<what>_<N>`: Maturation: a newborn starts at rest × 0.5 (`rest`), gain × 3 (`gain`) or both,
  and moves linearly to the layer's values over N ticks (a run is ~90,000 ticks at 1,500 games,
  ~314,000 at 5,000); `mat_rest025_to05_10k`: from rest × 0.25 / gain × 3 to rest × 0.5 / gain × 1;
- `..._close`: matured neurons are frozen and width growth no longer freezes the older neurons
  (the critical period replaces freezing at growth); `random_nofreeze`: its control, nothing is
  frozen at growth; `nofreeze_rest05_gain3`: nothing frozen, grown neurons young for good;
- `entropy_<x>`: width growth when the layer's firing-pattern entropy per neuron over 1,000 ticks
  falls below 0.x instead of on saturation; `either_090`: on either.

Δ = paired by seed against `random`, last three 50-game windows (final) or every window (all),
mean ± s.e. Diversity columns (PatternDiversity, last three windows of the newest layer): entropy
of the binary firing patterns, the same per neuron, distinct patterns, redundancy (1 - entropy /
sum of single-neuron entropies), share of neurons firing per tick.

## 1,500 games, seeds 1-30

| arm | seeds | final | Δ final vs random | Δ all vs random | neurons | widths | entropy (bits) | entropy / neuron | distinct | redundancy | active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 30 | 7.13 |  |  | 9.5 | 1.1 | 3.92 | 0.88 | 37 | 0.099 | 0.54 |
| rest05 | 30 | 7.45 | +0.32 ± 0.14 | +0.40 ± 0.08 | 9.1 | 1.1 | 3.95 | 0.90 | 32 | 0.088 | 0.54 |
| gain3 | 30 | 7.46 | +0.33 ± 0.15 | +0.68 ± 0.10 | 9.8 | 1.2 | 4.10 | 0.88 | 50 | 0.116 | 0.54 |
| rest05_gain3 | 30 | 7.43 | +0.30 ± 0.15 | +0.75 ± 0.12 | 9.9 | 1.3 | 4.09 | 0.87 | 62 | 0.134 | 0.54 |
| mat_rest_2k | 30 | 7.18 | +0.05 ± 0.21 | +0.07 ± 0.15 | 9.5 | 1.2 | 3.89 | 0.88 | 37 | 0.105 | 0.54 |
| mat_rest_10k | 30 | 7.23 | +0.11 ± 0.17 | +0.11 ± 0.08 | 9.4 | 1.2 | 3.91 | 0.90 | 40 | 0.112 | 0.54 |
| mat_rest_50k | 30 | 7.10 | -0.02 ± 0.15 | +0.19 ± 0.09 | 9.8 | 1.3 | 4.18 | 0.89 | 55 | 0.117 | 0.54 |
| mat_gain_10k | 30 | 7.02 | -0.11 ± 0.15 | +0.41 ± 0.10 | 10.1 | 1.3 | 4.21 | 0.86 | 54 | 0.113 | 0.54 |
| mat_gain_50k | 30 | 7.26 | +0.14 ± 0.16 | +0.42 ± 0.10 | 9.7 | 1.3 | 3.97 | 0.87 | 45 | 0.124 | 0.54 |
| mat_both_2k | 30 | 7.00 | -0.13 ± 0.15 | +0.20 ± 0.09 | 9.7 | 1.2 | 3.94 | 0.87 | 41 | 0.111 | 0.54 |
| mat_both_10k | 30 | 7.08 | -0.05 ± 0.15 | +0.48 ± 0.11 | 9.4 | 1.2 | 4.00 | 0.93 | 43 | 0.114 | 0.53 |
| mat_both_50k | 30 | 7.49 | +0.36 ± 0.17 | +0.72 ± 0.11 | 9.8 | 1.4 | 4.05 | 0.88 | 58 | 0.120 | 0.54 |
| mat_rest025_to05_10k | 30 | 7.19 | +0.06 ± 0.15 | +0.61 ± 0.08 | 9.5 | 1.3 | 3.93 | 0.89 | 41 | 0.116 | 0.54 |
| mat_both_10k_close | 30 | 6.89 | -0.24 ± 0.25 | +0.42 ± 0.19 | 9.8 | 1.4 | 3.98 | 0.86 | 42 | 0.113 | 0.53 |
| mat_both_50k_close | 30 | 7.79 | +0.66 ± 0.19 | +1.17 ± 0.15 | 9.9 | 1.3 | 4.02 | 0.88 | 55 | 0.125 | 0.52 |
| random_nofreeze | 30 | 7.75 | +0.62 ± 0.19 | +0.65 ± 0.15 | 9.3 | 1.1 | 3.81 | 0.90 | 26 | 0.091 | 0.55 |
| nofreeze_rest05_gain3 | 30 | 7.89 | +0.77 ± 0.23 | +1.29 ± 0.17 | 9.8 | 1.3 | 4.06 | 0.87 | 54 | 0.121 | 0.53 |
| mat_both_20k_close | 30 | 7.39 | +0.27 ± 0.20 | +0.96 ± 0.17 | 9.9 | 1.3 | 4.19 | 0.88 | 52 | 0.112 | 0.53 |
| entropy_085 | 30 | 2.40 | -4.72 ± 0.64 | -4.44 ± 0.48 | 15.1 | 3.0 | 1.89 | 0.30 | 64 | 0.665 | 0.57 |
| entropy_090 | 30 | 2.97 | -4.15 ± 0.69 | -3.96 ± 0.54 | 17.3 | 3.7 | 2.92 | 0.35 | 315 | 0.633 | 0.56 |
| entropy_095 | 30 | 4.10 | -3.03 ± 0.76 | -2.84 ± 0.65 | 19.5 | 4.3 | 4.33 | 0.46 | 697 | 0.515 | 0.57 |
| either_090 | 30 | 7.57 | +0.44 ± 0.22 | +0.53 ± 0.20 | 18.6 | 3.8 | 6.45 | 0.84 | 745 | 0.149 | 0.55 |
| either_090_rest05 | 30 | 7.70 | +0.57 ± 0.28 | +0.79 ± 0.21 | 17.4 | 3.4 | 6.42 | 0.85 | 772 | 0.152 | 0.55 |

Per evaluation window, every run: correlation of the score with
| measure | r (all windows) | r (last 3 windows, per run) | partial r given size |
|---|---|---|---|
| entropy | -0.19 | -0.10 | -0.03 |
| entropy / neuron | +0.01 | +0.27 | -0.05 |
| distinct | -0.05 | -0.16 | +0.20 |
| redundancy | +0.03 | -0.33 | +0.08 |
| size | -0.16 | -0.21 | +nan |

## 5,000 games, seeds 1-20

| arm | seeds | final | Δ final vs random | Δ all vs random | neurons | widths | entropy (bits) | entropy / neuron | distinct | redundancy | active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 20 | 7.24 |  |  | 11.2 | 2.0 | 4.41 | 0.78 | 117 | 0.214 | 0.54 |
| rest05_gain3 | 20 | 7.57 | +0.33 ± 0.20 | +0.32 ± 0.11 | 12.4 | 2.6 | 4.83 | 0.72 | 155 | 0.269 | 0.55 |
| mat_both_50k | 20 | 7.22 | -0.02 ± 0.20 | +0.30 ± 0.10 | 12.0 | 2.5 | 4.79 | 0.73 | 141 | 0.264 | 0.53 |
| mat_both_50k_close | 20 | 7.73 | +0.50 ± 0.17 | +0.57 ± 0.12 | 13.2 | 3.1 | 5.29 | 0.72 | 201 | 0.279 | 0.53 |
| random_nofreeze | 20 | 7.60 | +0.36 ± 0.16 | +0.40 ± 0.14 | 11.0 | 1.9 | 4.21 | 0.77 | 87 | 0.236 | 0.55 |
| nofreeze_rest05_gain3 | 20 | 7.91 | +0.67 ± 0.22 | +0.65 ± 0.17 | 13.0 | 2.9 | 4.73 | 0.69 | 194 | 0.303 | 0.54 |
| mat_both_100k | 20 | 7.64 | +0.41 ± 0.20 | +0.39 ± 0.10 | 11.4 | 2.2 | 4.60 | 0.78 | 113 | 0.242 | 0.53 |
| mat_both_100k_close | 20 | 7.68 | +0.44 ± 0.22 | +0.57 ± 0.13 | 12.0 | 2.5 | 4.60 | 0.73 | 133 | 0.277 | 0.53 |

Per evaluation window, every run: correlation of the score with
| measure | r (all windows) | r (last 3 windows, per run) | partial r given size |
|---|---|---|---|
| entropy | -0.14 | -0.20 | -0.04 |
| entropy / neuron | +0.08 | +0.38 | -0.07 |
| distinct | -0.08 | -0.28 | +0.11 |
| redundancy | -0.08 | -0.38 | +0.07 |
| size | -0.13 | -0.32 | +nan |

Paired, 5,000 games: `nofreeze_rest05_gain3` − `random_nofreeze` +0.31 ± 0.25 (all +0.25 ± 0.22);
`mat_both_50k_close` − `nofreeze_rest05_gain3` −0.17 ± 0.20; `mat_both_100k_close` −
`nofreeze_rest05_gain3` −0.23 ± 0.18; `mat_both_100k` − `rest05_gain3` +0.08 ± 0.11.
Redundancy of the E-R layer, mean over seeds at windows 1 / 11 / 31 / 61 / 100: random 0.03 /
0.04 / 0.11 / 0.17 / 0.21; nofreeze_rest05_gain3 0.03 / 0.04 / 0.16 / 0.26 / 0.30.

`runs.jsonl.gz`: one line per run (metrics, score curve, `diversity` per evaluation window, final
sizes, params, `games_run`).
