### memory_readability: balanced accuracy by blank length (mean ± sd over seeds; chance 0.50)

The State layer's values are what a downstream neuron reads; ridge = best linear readout, online = the library's delta rule.

**recovery=0.9, habituation=off** (20 seeds)

| readout | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| ridge: outputs (A) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.01 | 0.99 ± 0.03 | 0.94 ± 0.08 | 0.89 ± 0.04 | 0.80 ± 0.08 | 0.69 ± 0.06 | 0.57 ± 0.05 |
| ridge: thresholds via State (B) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.95 ± 0.01 | 0.89 ± 0.04 | 0.73 ± 0.06 | 0.57 ± 0.05 |
| ridge: outputs + State (D) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.96 ± 0.01 | 0.93 ± 0.03 | 0.77 ± 0.06 | 0.58 ± 0.05 |
| online: outputs (A) | 0.95 ± 0.08 | 0.91 ± 0.12 | 0.88 ± 0.14 | 0.92 ± 0.10 | 0.85 ± 0.11 | 0.74 ± 0.08 | 0.67 ± 0.09 | 0.59 ± 0.05 | 0.52 ± 0.03 |
| online: thresholds via State (B) | 0.98 ± 0.03 | 0.99 ± 0.01 | 0.94 ± 0.13 | 0.91 ± 0.09 | 0.79 ± 0.10 | 0.73 ± 0.08 | 0.72 ± 0.06 | 0.62 ± 0.07 | 0.52 ± 0.04 |
| online: outputs + State (D) | 0.99 ± 0.01 | 1.00 ± 0.00 | 0.98 ± 0.04 | 0.97 ± 0.09 | 0.96 ± 0.05 | 0.92 ± 0.04 | 0.87 ± 0.06 | 0.71 ± 0.06 | 0.56 ± 0.04 |

**recovery=0.9, habituation=on (steps 4, tolerance 0.5)** (20 seeds)

| readout | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| ridge: outputs (A) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.01 | 0.99 ± 0.02 | 0.94 ± 0.08 | 0.89 ± 0.04 | 0.80 ± 0.08 | 0.69 ± 0.06 | 0.57 ± 0.05 |
| ridge: thresholds via State (B) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.95 ± 0.01 | 0.89 ± 0.05 | 0.73 ± 0.05 | 0.57 ± 0.04 |
| ridge: habituation streaks via State | 1.00 ± 0.00 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.50 ± 0.01 | 0.51 ± 0.03 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.49 ± 0.02 |
| ridge: outputs + State (D) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.96 ± 0.02 | 0.92 ± 0.04 | 0.76 ± 0.05 | 0.58 ± 0.04 |
| online: outputs (A) | 0.95 ± 0.08 | 0.91 ± 0.12 | 0.88 ± 0.14 | 0.93 ± 0.09 | 0.83 ± 0.11 | 0.74 ± 0.08 | 0.67 ± 0.09 | 0.59 ± 0.05 | 0.52 ± 0.03 |
| online: thresholds via State (B) | 0.98 ± 0.03 | 0.99 ± 0.01 | 0.94 ± 0.13 | 0.91 ± 0.09 | 0.79 ± 0.09 | 0.73 ± 0.08 | 0.73 ± 0.06 | 0.62 ± 0.07 | 0.52 ± 0.04 |
| online: habituation streaks via State | 0.94 ± 0.17 | 0.50 ± 0.03 | 0.49 ± 0.03 | 0.50 ± 0.02 | 0.51 ± 0.02 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.49 ± 0.02 | 0.50 ± 0.02 |
| online: outputs + State (D) | 0.94 ± 0.11 | 0.95 ± 0.10 | 0.97 ± 0.06 | 0.90 ± 0.15 | 0.88 ± 0.10 | 0.86 ± 0.08 | 0.78 ± 0.10 | 0.68 ± 0.07 | 0.55 ± 0.03 |
| neurons habituated at the query | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**recovery=0.99, habituation=off** (20 seeds)

| readout | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| ridge: outputs (A) | 1.00 ± 0.00 | 1.00 ± 0.01 | 1.00 ± 0.01 | 1.00 ± 0.01 | 1.00 ± 0.01 | 0.99 ± 0.01 | 0.99 ± 0.01 | 0.96 ± 0.04 | 0.95 ± 0.04 |
| ridge: thresholds via State (B) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.98 ± 0.01 |
| ridge: outputs + State (D) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.99 ± 0.01 |
| online: outputs (A) | 0.91 ± 0.15 | 0.91 ± 0.14 | 0.96 ± 0.05 | 0.89 ± 0.15 | 0.90 ± 0.11 | 0.89 ± 0.14 | 0.83 ± 0.17 | 0.87 ± 0.09 | 0.78 ± 0.10 |
| online: thresholds via State (B) | 0.98 ± 0.06 | 0.98 ± 0.04 | 0.96 ± 0.15 | 0.99 ± 0.01 | 0.99 ± 0.04 | 0.98 ± 0.05 | 0.94 ± 0.10 | 0.85 ± 0.11 | 0.79 ± 0.11 |
| online: outputs + State (D) | 0.99 ± 0.01 | 0.99 ± 0.02 | 1.00 ± 0.00 | 0.99 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.01 | 0.91 ± 0.16 | 0.95 ± 0.10 | 0.95 ± 0.05 |

**recovery=0.99, habituation=on (steps 4, tolerance 0.5)** (20 seeds)

| readout | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| ridge: outputs (A) | 1.00 ± 0.00 | 1.00 ± 0.01 | 1.00 ± 0.01 | 1.00 ± 0.01 | 1.00 ± 0.01 | 0.99 ± 0.01 | 0.99 ± 0.01 | 0.96 ± 0.04 | 0.95 ± 0.04 |
| ridge: thresholds via State (B) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.98 ± 0.01 |
| ridge: habituation streaks via State | 1.00 ± 0.00 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.50 ± 0.01 | 0.51 ± 0.03 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.49 ± 0.02 |
| ridge: outputs + State (D) | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.99 ± 0.01 |
| online: outputs (A) | 0.91 ± 0.15 | 0.91 ± 0.14 | 0.96 ± 0.05 | 0.89 ± 0.15 | 0.90 ± 0.11 | 0.89 ± 0.14 | 0.83 ± 0.17 | 0.87 ± 0.09 | 0.77 ± 0.10 |
| online: thresholds via State (B) | 0.98 ± 0.06 | 0.98 ± 0.04 | 0.96 ± 0.15 | 0.99 ± 0.01 | 0.99 ± 0.04 | 0.98 ± 0.05 | 0.94 ± 0.10 | 0.84 ± 0.11 | 0.80 ± 0.09 |
| online: habituation streaks via State | 0.94 ± 0.17 | 0.50 ± 0.03 | 0.49 ± 0.03 | 0.50 ± 0.02 | 0.51 ± 0.02 | 0.50 ± 0.02 | 0.50 ± 0.02 | 0.49 ± 0.02 | 0.50 ± 0.02 |
| online: outputs + State (D) | 0.95 ± 0.10 | 0.93 ± 0.15 | 0.98 ± 0.04 | 0.95 ± 0.13 | 0.96 ± 0.08 | 0.96 ± 0.07 | 0.90 ± 0.16 | 0.91 ± 0.11 | 0.88 ± 0.07 |
| neurons habituated at the query | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### delayed_credit: success rate over seeds by delay

Success: the rewarded cue's weight ends above every irrelevant weight and the punished cue's below every one; for state=tap separately for the direct input weights (inputs) and the relay-threshold weights (State).

| rule | lr | state | relay recovery | weights | d=0 | d=1 | d=2 | d=4 | d=8 | d=16 | d=32 | d=64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sign | 0.001 | none | - | inputs | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sign | 0.001 | tap | 0.9 | inputs | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sign | 0.001 | tap | 0.9 | State | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| sign | 0.001 | tap | 0.99 | inputs | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sign | 0.001 | tap | 0.99 | State | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| sign | 0.01 | none | - | inputs | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sign | 0.01 | tap | 0.9 | inputs | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sign | 0.01 | tap | 0.9 | State | 0.9 | 0.9 | 0.7 | 0.3 | 0.5 | 0.0 | 0.0 | 0.0 |
| sign | 0.01 | tap | 0.99 | inputs | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sign | 0.01 | tap | 0.99 | State | 1.0 | 0.9 | 0.7 | 0.1 | 0.4 | 0.8 | 0.9 | 0.9 |
| trace | 0.01 | none | - | inputs | 1.0 | 1.0 | 1.0 | 0.9 | 0.6 | 0.1 | 0.0 | 0.0 |
| trace | 0.01 | tap | 0.9 | inputs | 1.0 | 1.0 | 1.0 | 0.5 | 0.4 | 0.4 | 0.0 | 0.0 |
| trace | 0.01 | tap | 0.9 | State | 1.0 | 1.0 | 1.0 | 0.4 | 0.3 | 0.9 | 0.3 | 0.0 |
| trace | 0.01 | tap | 0.99 | inputs | 1.0 | 1.0 | 0.6 | 0.6 | 0.9 | 0.7 | 0.0 | 0.0 |
| trace | 0.01 | tap | 0.99 | State | 1.0 | 1.0 | 1.0 | 0.4 | 0.8 | 0.7 | 0.4 | 0.1 |

Learned response difference, rewarded cue minus punished cue, at the cue tick (test, no learning; mean ± sd over seeds; 0 = the cues are not told apart):

| rule | lr | state | relay recovery | d=0 | d=1 | d=2 | d=4 | d=8 | d=16 | d=32 | d=64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sign | 0.001 | none | - | 0.4 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| sign | 0.001 | tap | 0.9 | 1.1 ± 0.1 | 0.5 ± 0.1 | 0.6 ± 0.1 | 0.9 ± 0.2 | 1.1 ± 0.1 | -0.0 ± 0.0 | -0.0 ± 0.0 | -0.0 ± 0.0 |
| sign | 0.001 | tap | 0.99 | 1.1 ± 0.1 | 0.5 ± 0.1 | 0.7 ± 0.1 | 1.2 ± 0.2 | 1.9 ± 0.3 | 3.2 ± 0.4 | 3.3 ± 0.5 | 2.6 ± 0.4 |
| sign | 0.01 | none | - | 10.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| sign | 0.01 | tap | 0.9 | 20.0 ± 0.0 | 8.4 ± 0.1 | 8.2 ± 0.2 | 8.2 ± 0.3 | 8.1 ± 0.4 | -0.0 ± 0.0 | 0.0 ± 0.0 | -0.0 ± 0.0 |
| sign | 0.01 | tap | 0.99 | 20.0 ± 0.0 | 8.0 ± 0.1 | 7.8 ± 0.2 | 7.8 ± 0.3 | 7.8 ± 0.3 | 7.6 ± 0.5 | 7.6 ± 0.5 | 7.6 ± 0.5 |
| trace | 0.01 | none | - | 3.7 ± 0.2 | 3.0 ± 0.2 | 3.0 ± 0.2 | 2.6 ± 0.5 | 1.5 ± 0.5 | 0.9 ± 0.4 | 0.1 ± 0.1 | 0.0 ± 0.0 |
| trace | 0.01 | tap | 0.9 | 6.4 ± 0.3 | 8.8 ± 1.6 | 12.7 ± 1.1 | 19.3 ± 0.9 | 20.0 ± 0.0 | 19.4 ± 1.1 | 7.2 ± 1.7 | 0.3 ± 0.2 |
| trace | 0.01 | tap | 0.99 | 6.9 ± 0.3 | 9.8 ± 1.5 | 12.7 ± 0.6 | 20.0 ± 0.0 | 20.0 ± 0.0 | 20.0 ± 0.0 | 16.2 ± 0.5 | 7.7 ± 0.9 |

### video_memory_learned: test accuracy by blank frames (mean ± sd over seeds; chance 0.50)

**recovery=0.9** (10 seeds)

| readout | inputs | gap 1 | gap 4 | gap 16 |
|---|---|---|---|---|
| ridge | outputs | 0.55 ± 0.02 | 0.49 ± 0.01 | 0.49 ± 0.01 |
| ridge | outputs + State | 0.67 ± 0.01 | 0.59 ± 0.04 | 0.50 ± 0.02 |
| online, trace 0 | outputs | 0.49 ± 0.01 | 0.50 ± 0.02 | 0.50 ± 0.01 |
| online, trace 0 | outputs + State | 0.54 ± 0.03 | 0.53 ± 0.01 | 0.50 ± 0.01 |
| online, trace 0.99 | outputs | 0.50 ± 0.01 | 0.50 ± 0.02 | 0.50 ± 0.01 |
| online, trace 0.99 | outputs + State | 0.51 ± 0.01 | 0.50 ± 0.00 | 0.50 ± 0.00 |

**recovery=0.99** (10 seeds)

| readout | inputs | gap 1 | gap 4 | gap 16 |
|---|---|---|---|---|
| ridge | outputs | 0.72 ± 0.02 | 0.72 ± 0.03 | 0.71 ± 0.02 |
| ridge | outputs + State | 0.90 ± 0.01 | 0.88 ± 0.02 | 0.75 ± 0.02 |
| online, trace 0 | outputs | 0.52 ± 0.02 | 0.51 ± 0.03 | 0.50 ± 0.01 |
| online, trace 0 | outputs + State | 0.87 ± 0.02 | 0.83 ± 0.02 | 0.67 ± 0.03 |
| online, trace 0.99 | outputs | 0.51 ± 0.02 | 0.51 ± 0.02 | 0.51 ± 0.03 |
| online, trace 0.99 | outputs + State | 0.68 ± 0.03 | 0.66 ± 0.04 | 0.49 ± 0.03 |

Score: test accuracy (t1-t3), 1 - NMSE (t4).

### nl_temporal: test score at the learning rate with the best validation (mean ± sd over seeds)

| task | habituation | readout reads | lr | test score | spikes |
|---|---|---|---|---|---|
| t1 | false | outputs | 0.03 | 0.994 ± 0.016 | 39.9 ± 0.6 |
| t1 | false | outputs + State | 0.03 | 0.997 ± 0.008 | 38.6 ± 1.3 |
| t1 | true | outputs | 0.001 | 0.496 ± 0.000 | 9.9 ± 0.7 |
| t1 | true | outputs + State | 0.001 | 1.000 ± 0.000 | 16.1 ± 0.0 |
| t2 | false | outputs | 0.01 | 0.743 ± 0.000 | 36.2 ± 0.9 |
| t2 | false | outputs + State | 0.01 | 0.744 ± 0.001 | 40.9 ± 3.3 |
| t2 | true | outputs | 0.001 | 0.748 ± 0.000 | 11.8 ± 0.6 |
| t2 | true | outputs + State | 0.01 | 0.615 ± 0.056 | 9.9 ± 0.5 |
| t3 | false | outputs | 0.001 | 0.745 ± 0.122 | 34.2 ± 2.4 |
| t3 | false | outputs + State | 0.03 | 0.514 ± 0.072 | 38.2 ± 1.0 |
| t3 | true | outputs | 0.001 | 0.495 ± 0.000 | 12.9 ± 0.8 |
| t3 | true | outputs + State | 0.01 | 0.495 ± 0.000 | 13.9 ± 1.4 |
| t4 | false | outputs | 0.1 | -1092.267 ± 0.362 | 39.2 ± 1.1 |
| t4 | false | outputs + State | 0.1 | -1092.477 ± 0.339 | 39.3 ± 0.7 |
| t4 | true | outputs | 0.03 | -0.000 ± 0.000 | 15.2 ± 0.0 |
| t4 | true | outputs + State | 0.1 | -1087.594 ± 6.787 | 9.8 ± 3.4 |

Score: accuracy (dir, change), R² (vel), catch rate (catch).

### dyn_ladder: test score at the learning rate with the best validation (mean ± sd over seeds)

| task | readout reads | lr | test score | spikes |
|---|---|---|---|---|
| catch | outputs | 0.003 | 0.215 ± 0.026 | 22.6 ± 1.7 |
| catch | outputs + State | 0.003 | 0.253 ± 0.043 | 21.3 ± 1.8 |
| change | outputs | 0.001 | 0.905 ± 0.051 | 74.9 ± 2.0 |
| change | outputs + State | 0.003 | 0.962 ± 0.023 | 55.8 ± 4.2 |
| dir | outputs | 0.01 | 0.568 ± 0.037 | 32.2 ± 2.2 |
| dir | outputs + State | 0.01 | 0.531 ± 0.010 | 31.7 ± 1.4 |
| vel | outputs | 0.003 | -0.042 ± 0.074 | 37.6 ± 2.5 |
| vel | outputs + State | 0.003 | 0.065 ± 0.033 | 35.5 ± 2.6 |

