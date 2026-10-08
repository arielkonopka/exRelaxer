# snake-growth-long: how big does a Snake network grow from 16 neurons?

Runs of `NNtesting/experiments/snake_growth/long_run.py` (research log §31). The snake_growth agent
starts from ER(16) and plays game after game (each game ends at the snake's death), learning and
developing all the time, until its size has not changed for 200 evaluation windows (10,000 games)
or 50,000 games. "freeze4": depth growth leaves the older layers learning until the 4th
adaptive layer is added, which freezes every older layer (default: each new layer freezes the one
before). "wide": a layer added by depth growth is as wide as the layer it grows behind,
protected down to 3 neurons (the outputs), and its original neurons may be pruned. "weak": pruning also removes neurons the next layers barely read (read strength
below 0.2 x the layer mean, judged from 20,000 ticks after the layer appears). Controls without growth play 12,000 games. Defaults of experiment.py unless the
setup says otherwise; "open" lifts the caps to 4,096 neurons per layer and 64 layers. Seeds 1-3.

`runs.jsonl.gz`: one line per run with the score and sizes after every 50-game window and every
growth, undo and depth event. `nets/`: the final network of each run (`Network.load`).

| setup | final sizes per seed (total) | largest ever | games played | last 500 games, apples/game | width / undo / depth / prune events (sum) | last size change at game |
|---|---|---|---|---|---|---|
| s16_fixed_rec05 | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 9.75 (8.91-10.44) | 0 / 0 / 0 / 0 | -/-/- |
| s16_open_rec05 | [16, 4, 4, 4] (28) [16, 4, 4] (24) [16, 4, 4, 4] (28) | 28 | 18700/14200/18650 | 8.99 (8.79-9.22) | 0 / 0 / 8 / 0 | 8700/4200/8650 |
| s16_open_rec05_freeze4 | [16, 4, 4, 4] (28) [16, 4, 4] (24) [16, 4, 4, 4, 4] (32) | 32 | 12200/11800/16950 | 9.44 (8.94-9.94) | 0 / 0 / 9 / 0 | 2200/1800/6950 |
| s16_open_rec05_freeze4_wide | [16, 16, 16, 16, 16] (80) [16, 16, 16, 16] (64) [16, 16, 16, 16, 16] (80) | 80 | 15400/22050/12800 | 9.76 (9.23-10.41) | 0 / 0 / 11 / 0 | 5400/12050/2800 |
| s16_open_rec05_freeze4_wide_weak | [16, 6, 5, 3] (30) [16, 10, 8, 3] (37) [16, 8, 6, 3] (33) | 42 | 13600/14700/13500 | 9.36 (9.20-9.58) | 0 / 0 / 9 / 38 | 3600/4700/3500 |
| s16_fixed_rec075 | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 9.68 (9.33-10.08) | 0 / 0 / 0 / 0 | -/-/- |
| s16_open_rec075 | [16, 4, 6] (26) [16, 4, 4, 4] (28) [16, 4, 4, 4] (28) | 28 | 20150/17600/20300 | 8.92 (8.28-9.48) | 4 / 3 / 8 / 0 | 10150/7600/10300 |
| s16_open_rec075_freeze4 | [16, 4, 4, 4] (28) [16, 4, 4, 4, 4] (32) [16, 4, 4, 4] (28) | 32 | 12900/19950/13350 | 9.54 (9.29-9.77) | 0 / 0 / 10 / 0 | 2900/9950/3350 |
| s16_open_rec075_freeze4_wide | [16, 16, 16, 16] (64) [16, 16, 16, 16, 16] (80) [16, 16, 16, 16] (64) | 80 | 12900/17900/16200 | 9.69 (9.53-9.78) | 0 / 0 / 10 / 0 | 2900/7900/6200 |
| s16_open_rec075_freeze4_wide_weak | [16, 9, 6, 3, 3] (37) [16, 11, 3, 3] (33) [16, 6, 3, 3, 3] (31) | 38 | 22250/15550/17550 | 9.69 (9.59-9.85) | 0 / 0 / 11 / 39 | 12250/5550/7550 |
| s16_fixed | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 8.22 (7.63-8.93) | 0 / 0 / 0 / 0 | -/-/- |
| s16_capped | [16, 6] (22) [16, 6] (22) [16, 6] (22) | 22 | 11950/13400/11450 | 8.43 (7.87-9.01) | 3 / 0 / 3 / 0 | 1950/3400/1450 |
| s16_open | [16, 6, 6] (28) [16, 4, 4] (24) [16, 6, 6, 4] (32) | 32 | 20250/11950/20200 | 8.34 (7.94-8.85) | 5 / 1 / 7 / 0 | 10250/1950/10200 |
| s16_open_noundo | [16, 6, 6] (28) [16, 4, 4] (24) [16, 6, 8] (30) | 30 | 20250/11950/20700 | 8.37 (8.02-8.85) | 5 / 0 / 6 / 0 | 10250/1950/10700 |
| s16_open_freeze4 | [16, 10, 4] (30) [16, 4, 4] (24) [16, 4, 4] (24) | 30 | 21700/11950/10950 | 8.23 (7.36-8.87) | 4 / 1 / 6 / 0 | 11700/1950/950 |
| s16_open_freeze4_wide | [16, 16] (32) [16, 16, 16] (48) [16, 16, 16] (48) | 48 | 10950/12250/11700 | 7.73 (7.59-7.82) | 0 / 0 / 5 / 0 | 950/2250/1700 |
| s16_open_freeze4_wide_weak | [16, 3] (19) [16, 6, 3, 3] (28) [16, 11, 3] (30) | 38 | 13850/14350/12600 | 8.17 (7.79-8.49) | 39 / 7 / 6 / 78 | 3850/4350/2600 |
| s16_fixed_rec099 | [16] (16) [16] (16) [16] (16) | 16 | 12000/12000/12000 | 2.22 (2.12-2.31) | 0 / 0 / 0 / 0 | -/-/- |
| s16_open_rec099 | [20, 10, 4] (34) [20, 10, 4] (34) [18, 14] (32) | 34 | 12200/13900/11050 | 3.87 (3.76-3.95) | 18 / 2 / 5 / 0 | 2200/3900/1050 |
| s16_open_rec099_noundo | [20, 12, 4] (36) [20, 12, 4] (36) [18, 14] (32) | 36 | 11950/15100/11050 | 3.86 (3.68-3.96) | 18 / 0 / 5 / 0 | 1950/5100/1050 |

Only the "weak" runs pruned any neuron (inactive pruning never fired).
