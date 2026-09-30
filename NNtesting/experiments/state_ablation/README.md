# state_ablation

Which components of the hidden neurons' state carry the direction of an
occluded object, and do they add to each other? An ablation of the State
layer ([doc/state_output.md](../../../doc/state_output.md)): six
configurations read different columns of the same recorded state, under
otherwise identical conditions. Results: research log §27,
[`results/state-ablation/`](../../../results/state-ablation/).

## Task

video_memory's `side` task, unchanged ([clips.py](../video_memory/clips.py)):
a square moves left or right for 3–6 frames, the screen is blank (exactly
zero) for `gap` frames, then the square reappears at the centre for 2
frames. The reappearance is drawn the same way for both classes, so the
last frames carry no information about the direction; only the state the
motion left behind does. LEFT and RIGHT are exactly 50% of every split
(checked and recorded). The gap keeps video_memory's definition: blank
frames, 4 ticks each; gap 0 means the square reappears right after the
motion.

## Network

    clip (300 pixels) -> h: 128 E-R neurons with habituation, frozen
                      -> State layer (output, threshold - rest, habituation streak per neuron)
                      -> 2 readouts, learned online

The hidden layer is video_memory's E0 (normalised sums, resting threshold
0.2, weights uniform with variance 1 / fan-in) plus habituation with the
library defaults (cut after 100 exact repeats), so that the habituation
column is the neurons' real streak. E-R recovery 0.9 (video_memory's
default) and 0.99 (longer memory) are separate runs.

## Configurations

| | reads | columns |
|---|---|---|
| A | output | 128 |
| B | threshold − rest | 128 |
| C | habituation streak | 128 |
| D | output, threshold − rest | 256 |
| E | threshold − rest, habituation streak | 256 |
| F | output, threshold − rest, habituation streak | 384 |

## What is identical

For one trial (seed): the clips (train 1000, validation 500, test 1000),
the hidden weights, the recorded state (the hidden layer runs once per
clip; every configuration reads the same numbers), the readout's initial
weights (zero), learning rule (`apply_error`, feedback alignment on an
output layer = the delta rule, with a bias), learning rate, 10 epochs, clip
order, the error timing (once per clip, on the mean readout output over the
8 readout ticks), and the readout-side input gain (each column divided by
its standard deviation over the training readout ticks). The learning rate
is tuned once for all configurations (`sweep.sh`, tuning seeds 1000–1002,
validation clips only).

## Measures

Per configuration: accuracy, balanced accuracy, the confusion counts, and
the balanced accuracy of the best linear readout of the same columns (ridge,
offline). Per state component, independently of any readout, at the end of
the blank and at the last readout tick: population means for LEFT and RIGHT,
per-neuron |d|, neurons with |d| > 0.5, and a ridge probe on that component
alone.

    ./NNtesting/experiments/state_ablation/sweep.sh results/state-ablation
    python3 NNtesting/tools/state_ablation_summary.py results/state-ablation --md results/state-ablation/summary.md
