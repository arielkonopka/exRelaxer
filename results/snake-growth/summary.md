# snake_growth: Snake learned always-on by a growing network

Runs of `NNtesting/experiments/snake_growth/sweep.py` (defaults of experiment.py unless the setup says otherwise;
1,500 games, 10 x 10 board, seeds 1-3). `runs.jsonl.gz`: one line per run with metrics, the score curve
(mean apples per 50-game window), final sizes and the event log. See doc/snake_growth_guide.md and research log §30.

| setup | final (mean, min-max) | best window | best game | final sizes per seed | width / prune / depth events (sum) |
|---|---|---|---|---|---|
| one_fixed | 0.21 (0.04-0.40) | 0.83 | 9 | [1] [1] [1] | 0 / 0 / 0 |
| one_grow | 4.50 (3.57-5.45) | 5.54 | 21 | [5, 4] [5, 4] [5, 4] | 6 / 0 / 3 |
| one_grow_plastic | 4.00 (2.46-5.50) | 5.35 | 23 | [3, 4] [7, 4] [3, 4] | 5 / 0 / 3 |
| one_grow_rec099 | 2.47 (1.35-3.10) | 3.06 | 14 | [11, 10] [16, 12] [11, 16] | 32 / 1 / 3 |
| four_fixed | 5.12 (4.85-5.54) | 6.01 | 20 | [4] [4] [4] | 0 / 0 / 0 |
| four_grow | 5.76 (5.04-6.64) | 6.97 | 23 | [6] [6, 4] [4, 6] | 3 / 0 / 2 |
| four_grow_plastic | 5.76 (5.45-6.32) | 6.57 | 23 | [4, 4] [6, 4] [4, 6] | 3 / 1 / 3 |
| eight_fixed | 7.02 (6.75-7.16) | 8.35 | 25 | [8] [8] [8] | 0 / 0 / 0 |

Saturation of the untrained network (`sweep.py OUT --saturation`): share of saturated ticks,
30 games with 30% random moves.

| recovery | ER(1) | ER(2) | ER(4) | ER(8) | ER(16) |
|---|---|---|---|---|---|
| 0.9 | 0.75 | 0.38 | 0.19 | 0.00 | 0.00 |
| 0.99 | 0.90 | 0.76 | 0.65 | 0.39 | 0.11 |
