# E-R experiments at recovery 0.9 and 0.5

Runs of `results/recovery/run.sh` (current defaults, same seeds, only the recovery differs).

## Activity experiments (20 trials, means)

| experiment | metric | 0.9 | 0.5 |
|---|---|---|---|
| er_economy | E-R accuracy | 0.998 | 1 |
| er_economy | E-R active fraction | 0.474 | 0.719 |
| er_economy | fixed threshold (matched) active fraction | 0.484 | 0.738 |
| er_economy | E-R spikes per inference | 127 | 193 |
| er_history | history: pattern change after path A's use | 0.141 | 0.0681 |
| er_history | history: pattern change after all paths' use | 0.792 | 0.396 |
| er_history | history: decision change after all paths' use | 0.163 | 0 |
| er_fatigue | fatigue (all paths): first active fraction | 0.0513 | 0.511 |
| er_fatigue | fatigue (all paths): first accuracy | 0.9 | 1 |
| er_fatigue | fatigue (path B): first active fraction | 0.61 | 0.762 |
| er_silence | silence: self-active ticks | 0.0999 | 0.222 |
| er_silence | silence: spikes after / before | 1.75 | 1.2 |
| er_habituation | habituation off, hold 50: spikes per sample | 2.57e+03 | 2.77e+03 |
| er_habituation | hold 50: accuracy at the end of the hold | 1 | 0.992 |
| er_paths | paths: mean active run during a dwell | 16 | 9 |

## nl_static (test MSE) and nl_temporal (test accuracy; t4: -MSE), E-R only, 5 seeds

| task | 0.9: best (depth, width) | 0.9 active | 0.5: best (depth, width) | 0.5 active |
|---|---|---|---|---|
| l0 | 0.04758 (1, 64) | 0.42 | 0.001255 (1, 64) | 0.62 |
| l1 | 0.05167 (2, 64) | 0.38 | 0.06151 (2, 64) | 0.63 |
| l2 | 0.04551 (2, 64) | 0.38 | 0.04973 (2, 64) | 0.63 |
| l3 | 0.0616 (1, 64) | 0.42 | 0.08645 (2, 64) | 0.63 |
| l4 | 0.1207 (1, 64) | 0.43 | 0.1894 (2, 64) | 0.65 |
| t1 n=2 | 0.999 (1, 16) | 0.85 | 0.736 (3, 16) | 0.79 |
| t2 n=2 | 0.7975 (2, 64) | 0.70 | 0.774 (3, 8) | 0.81 |
| t3 n=4 | 0.5805 (2, 16) | 0.63 | 0.553 (3, 8) | 0.79 |
| t3 n=8 | 0.515 (1, 4) | 0.67 | 0.5185 (1, 64) | 0.92 |
| t4 n=2 | -0.0953 (2, 4) | 0.41 | -0.09247 (1, 4) | 0.64 |

## Other experiments (means)

| experiment | metric | 0.9 | 0.5 |
|---|---|---|---|
| dyn_ladder dir | test score | 0.527 | 0.544 |
| dyn_ladder change | test score | 0.589 | 0.693 |
| dyn_ladder vel | test score | -0.047 | -0.019 |
| dyn_ladder catch | test score | 0.216 | 0.206 |
| snake_rules E-R mix, sign readouts, frozen mix | test apples | 12.76 | 13.37 |
| snake_rules E-R mix, sign readouts, fa mix | test apples | 12.13 | 13.25 |
| snake_rules E-R mix, fa readouts, frozen mix | test apples | 11.86 | 11.94 |
| snake_rules E-R mix, fa readouts, fa mix | test apples | 11.88 | 11.05 |
| bar_orientation, E-R readout | accuracy | 0.434 | 0.520 |
