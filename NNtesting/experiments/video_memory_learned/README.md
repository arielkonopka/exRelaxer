# video_memory_learned

Can exRelaxer learn, online, to use a memory trace that its E-R state
already keeps? The [video_memory](../video_memory/README.md) experiment
showed that E-R thresholds keep the direction of an occluded object across
up to 16 blank frames, and that a ridge readout fitted offline reads part of
it. Here the readout is learned online by the library's own error rule,
with and without an eligibility trace, and the network state is probed
separately. Results: research log §22 and
[doc/video_memory_learned.md](../../../doc/video_memory_learned.md).

```bash
./build.sh --python
export PYTHONPATH=$PWD/build/EXrelaxer.py/package
NNtesting/nntest.py run video_memory_learned --trials 20 --set model=E0 --set gap=1,2,4,8,16 --out e0.jsonl
python3 NNtesting/tools/video_memory_learned_summary.py e0.jsonl --md summary.md --png curves.png
NNtesting/experiments/video_memory_learned/sweep.sh results/video-memory-learned   # every run behind §22
```

`--set` splits on commas (a grid); lists inside one parameter (`traces`,
`lrs`) take `;`, e.g. `--set 'traces=0;0.9'`.

## What is shared with video_memory

The clips (`clips.py`: side task, balanced LEFT/RIGHT in every split, the
same clips for every gap, model and trace at a given seed), the hidden
networks (`model=E0`, `C0`, `C1`, ... built by video_memory's `build`, from
the same seed, so the same weights), the E-R settings (normalised sum, no
habituation, linear threshold growth, recovery 0.9, resting threshold 0.2)
and the readout decision (the larger mean readout output over the 8 ticks
of the two reappearance frames). The ridge readout of video_memory is
recomputed in every trial and reproduces its accuracy exactly.

## What is new

**Online readouts.** Two library readout neurons (`LayerSpec.dense(2)`,
plain weighted sum plus bias) with the `feedback_alignment(trace)` rule and
a bias. On an output layer that rule is the delta rule, driven by
`Network.apply_error`:

```
w_ij += lr * error_j * X_i        X_i <- trace * X_i + x_i   (every tick; reset at the start of each clip)
```

Clips are presented one at a time, in random order, for `epochs` passes.
The whole clip runs through the readouts (so the trace sees the visible
frames, the blank and the reappearance). With `error_at=end` (default)
there is one prediction per clip, as in the test: the mean readout output
over the readout ticks. Its error, target (+1 / −1) minus that mean, is
applied once, after the last readout tick, to whatever eligibility the rule
holds then:

- `trace = 0` (immediate): the hidden output of the last tick;
- `trace = d`: the decaying sum of the hidden output over the whole clip,
  per tick (a frame is 4 ticks; at gap 4 and d = 0.9, the last visible
  frame is 16–24 ticks back, weight 0.9^16 = 0.19).

`error_at=tick` applies the error of every readout tick at that tick
instead (the classic online delta rule).

The forward pass does not change with the trace: the readouts compute w · x
+ b from the current hidden output. The trace only decides which inputs the
delayed error is credited to.

Each trial records the hidden network once and trains every trace value
from the same recordings and initial weights, each with its learning rate
chosen from `lrs` on the validation clips (as the ridge penalty is).

**Input gain.** `input_gain=sd` (default) scales each hidden output, on its
way to the readouts, by 1 / its standard deviation over the training
readout ticks: a fixed gain per synapse. Without it (`input_gain=none`)
the delta rule does not learn at all: E-R outputs are sparse and small
(mean |x| ≈ 0.004 at the reappearance), the bias dominates, and the input
covariance has a condition number of about 25,000. The ridge readout and
the probes standardise their features, so they do not change.

**Condition D (`mode=hidden`).** The hidden layer learns too: feedback
alignment from the same end-of-clip error, with its own input trace over the
pixels (`hidden_trace`), starting from the frozen network's weights. It is
the only way a delayed error can change what the E-R state holds at the
reappearance. The learned hidden layer is then frozen, recorded and read out
like the others.

**Library change.** `Network.reset_traces(layer)` (C++
`neuron_layer::clearTraces()`) zeroes a layer's learning traces without
touching weights, bias, baselines or the feedback matrix, so each clip
starts with an empty eligibility trace.

## Metrics

Per readout (`ridge`, `t0`, `t0.5`, ... `t0.99`, and `h` for none):
`_accuracy`, `_left_accuracy`, `_right_accuracy`, `_balanced_accuracy`,
`_cm_LL/LR/RL/RR` (test confusion matrix, true/predicted), and for the
online readouts `_train_accuracy`, `_val_accuracy`, `_lr`, `_mse_epochN`
(½‖target − prediction‖² per clip, during training) and the credit
diagnostics below. `constant_accuracy` is the trivial predictor's.

Class counts per split (`train_left`, ...), checked (an unbalanced split
fails the trial) and printed before any training.

Probes (ridge classifiers, network unchanged, penalty on validation):
`probe_before_reappearance` (E-R thresholds and outputs after the last blank
frame), `probe_prediction_state` (thresholds after the readout interval and
the mean readout-tick outputs), `probe_prediction_outputs` (the mean outputs
alone: what a linear readout can see), `probe_eligibility_tD` (the
eligibility trace the error meets).

Credit (training clips; `Δw` = the change of w_RIGHT − w_LEFT per hidden
neuron; selectivity = d′ between the classes): `_corr_dw_visible` (Δw
against each neuron's direction selectivity while the object is visible),
`_corr_dw_state` (against the selectivity of its threshold before the
reappearance), `_corr_dw_readout` (of its output at the prediction),
`_corr_dw_eligibility`, `_corr_dw_ridge` (against the ridge readout's
weights), `_eligibility_pre_share` (the share of the eligibility left by
activity before the readout interval), `_dw_share_top_visible` (the share of
|Δw| on the 10% most selective neurons before the blank), and
`neurons_selective_*` (neurons with |d′| > 0.5). `dump=DIR` saves the
per-neuron arrays.
