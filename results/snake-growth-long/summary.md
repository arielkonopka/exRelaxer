# snake-growth-long: how big does a Snake network grow from 16 neurons?

Runs of `NNtesting/experiments/snake_growth/long_run.py` (research log §31). The snake_growth agent
starts from ER(16) and plays game after game (each game ends at the snake's death), learning and
developing all the time, until its size has not changed for 200 evaluation windows (10,000 games)
or 50,000 games. "freeze4": depth growth leaves the older layers learning until the 4th
adaptive layer is added, which freezes every older layer (default: each new layer freezes the one
before). Controls without growth play 12,000 games. Defaults of experiment.py unless the
setup says otherwise; "open" lifts the caps to 4,096 neurons per layer and 64 layers. Seeds 1-3.

`runs.jsonl.gz`: one line per run with the score and sizes after every 50-game window and every
growth, undo and depth event. `nets/`: the final network of each run (`Network.load`).

| setup | final sizes per seed (total) | largest ever | games played | last 500 games, apples/game | width / undo / depth events (sum) | last size change at game |
|---|---|---|---|---|---|---|
| s16_fixed_rec05 | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 9.75 (8.91-10.44) | 0 / 0 / 0 | -/-/- |
| s16_open_rec05 | [16, 4, 4, 4] (28) [16, 4, 4] (24) [16, 4, 4, 4] (28) | 28 | 18700/14200/18650 | 8.99 (8.79-9.22) | 0 / 0 / 8 | 8700/4200/8650 |
| s16_open_rec05_freeze4 | [16, 4, 4, 4] (28) [16, 4, 4] (24) [16, 4, 4, 4, 4] (32) | 32 | 12200/11800/16950 | 9.44 (8.94-9.94) | 0 / 0 / 9 | 2200/1800/6950 |
| s16_fixed_rec075 | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 9.68 (9.33-10.08) | 0 / 0 / 0 | -/-/- |
| s16_open_rec075 | [16, 4, 6] (26) [16, 4, 4, 4] (28) [16, 4, 4, 4] (28) | 28 | 20150/17600/20300 | 8.92 (8.28-9.48) | 4 / 3 / 8 | 10150/7600/10300 |
| s16_open_rec075_freeze4 | [16, 4, 4, 4] (28) [16, 4, 4, 4, 4] (32) [16, 4, 4, 4] (28) | 32 | 12900/19950/13350 | 9.54 (9.29-9.77) | 0 / 0 / 10 | 2900/9950/3350 |
| s16_fixed | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 8.22 (7.63-8.93) | 0 / 0 / 0 | -/-/- |
| s16_capped | [16, 6] (22) [16, 6] (22) [16, 6] (22) | 22 | 11950/13400/11450 | 8.43 (7.87-9.01) | 3 / 0 / 3 | 1950/3400/1450 |
| s16_open | [16, 6, 6] (28) [16, 4, 4] (24) [16, 6, 6, 4] (32) | 32 | 20250/11950/20200 | 8.34 (7.94-8.85) | 5 / 1 / 7 | 10250/1950/10200 |
| s16_open_noundo | [16, 6, 6] (28) [16, 4, 4] (24) [16, 6, 8] (30) | 30 | 20250/11950/20700 | 8.37 (8.02-8.85) | 5 / 0 / 6 | 10250/1950/10700 |
| s16_open_freeze4 | [16, 10, 4] (30) [16, 4, 4] (24) [16, 4, 4] (24) | 30 | 21700/11950/10950 | 8.23 (7.36-8.87) | 4 / 1 / 6 | 11700/1950/950 |
| s16_fixed_rec099 | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 2.22 (2.12-2.31) | 0 / 0 / 0 | -/-/- |
| s16_open_rec099 | [20, 10, 4] (34) [20, 10, 4] (34) [18, 14] (32) | 34 | 12200/13900/11050 | 3.87 (3.76-3.95) | 18 / 2 / 5 | 2200/3900/1050 |
| s16_open_rec099_noundo | [20, 12, 4] (36) [20, 12, 4] (36) [18, 14] (32) | 36 | 11950/15100/11050 | 3.86 (3.68-3.96) | 18 / 0 / 5 | 1950/5100/1050 |

No run pruned a neuron (pruning removes only invalid or never-firing grown neurons).
