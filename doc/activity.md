# Activity economy and path selection

Four experiments ask whether excitation–relaxation (E-R) makes a network
use less activity, prefer some routes over others, or respond differently
depending on what it did before, **without any penalty on activity**:
nothing in any loss, reward or rule counts spikes or active neurons. The
only difference between the networks compared is how their neurons
respond. Results are reported as several measurements side by side, never
as one score, and no model is declared a winner. The measured numbers are
in the [research log, §13](research.md#13-activity-economy-and-path-selection).

## The network

```
                    ┌── path A (12 neurons) ──┐
input (32) ─────────┼── path B (20 neurons) ──┼──► readouts (one per class, linear)
stim_A, stim_B, ... └── path C (35 neurons) ──┘
```

`NNtesting/tasks/activity.hpp` builds it with named input sources. The
task input feeds three parallel dense paths of different sizes; every path
also has its own stimulation source (`stim_A`, ...), used only to fatigue
one path on purpose. The readouts are linear. The classification task is
four ±1 prototypes of 32 values with uniform noise (±1 by default), each
held for 4 ticks; the decision sums the readouts' evidence over the hold.

Three models share the seed, weights, task and learning:

| Model | Hidden neurons |
|-------|----------------|
| `er` | production E-R (habituation off by default) |
| `gate` | no E-R; a **fixed** firing threshold (`LayerSpec::gate`): output = sum if \|sum\| > gate, otherwise 0 |
| `linear` | no E-R: output = sum |

`gate` is the control that separates *thresholding* from *adaptation*. With
`gate=match` (the default) its threshold is calibrated before each trial so
that it is active as often as E-R on the same inputs (a quantile of the
linear network's |sums|), so any difference between `er` and `gate` comes
from E-R's changing threshold, not from being sparser.

`learning=fa` trains the paths with feedback alignment and the readouts
with the delta rule; `learning=readout` freezes the paths and trains only
the readouts.

A neuron counts as **active** on a tick when |output| exceeds
`firing_epsilon`; a spike is one active neuron on one tick. Spontaneous
E-R firings (±0.01) are counted separately.

## Experiments

| Experiment | Question (spec) | Main measurements |
|------------|-----------------|-------------------|
| `er_economy` | Exp 1 and 6: does E-R use less activity for the same accuracy, with no activity cost anywhere? | accuracy; active fraction, spikes per tick and per inference; unique and never-active neurons; mean active run; activity at the start and the end of training; accuracy per 100 spikes |
| `er_paths` | Exp 2 and 7: does the network prefer the cheaper path, and does it switch paths over long runs? | drive per neuron per path early and late; leading path, switch rate (all and within one input), dwell under one constant input |
| `er_fatigue` | Exp 3 and 5: after fatiguing a path, does activity move to the others, and how fast does it come back? | accuracy on the first and later probe samples; the fatigued path's share of activity relative to rest, the other paths' share; thresholds; ticks to recover |
| `er_history` | Exp 4: does the same input give a different response after different histories? | change in activity pattern, evidence, decision and spike count relative to rest; latency; accuracy; thresholds and leading path per condition |
| `er_silence` | Does a trained network stay active when its inputs are zeroed (spontaneous firing, optional recurrence)? How does it answer afterwards? | active fraction in tick windows 1–10, 11–100, 101–1000, later; ticks to quiet; first spontaneous firing, rate and burst period; readout mass and class switches; thresholds at the end; accuracy and spikes after the silence vs without it |
| `er_habituation` | Is habituation an activity-saving mechanism? Samples held for `test_holds` ticks (4, 50, 200, 500), optional per-tick flicker `hold_noise` | spikes per sample, late active fraction; accuracy from the first `hold` ticks (onset), the whole sample (sum) and the last tick (end); accuracy per 100 spikes |

History conditions (`er_fatigue`, `er_history`): `rest` (no input), `busy`
(ordinary task inputs), `all` (every path stimulated), `A`, `B`, `C` (one
path stimulated). Stimulation lasts `condition_ticks` (20) at `amplitude`
(10). Every condition starts from the same trained network (saved and
restored), so conditions differ only in what came just before.

```bash
./build/NNtesting/nntest run er_economy --set learning=fa,readout --trials 10
./build/NNtesting/nntest run er_paths --set online=false,true
./build/NNtesting/nntest run er_fatigue --set recovery=0.9,0.97
./build/NNtesting/nntest run er_history --set recovery=0.9,0.97
./build/NNtesting/nntest run er_silence --set recovery=0.9,0.97 --set recurrent=false,true
./build/NNtesting/nntest run er_habituation --set habituation=false,true \
    --set habituation_steps=5,100 --set habituation_decay=0,0.9 --set hold_noise=0,0.001
```

The habituation rule is set with `habituation_steps` (identical ticks
before it acts, default 100), `habituation_tolerance` (relative difference
still counted as identical, default 0: exact repeats only) and
`habituation_decay` (0 cuts the input; a value in (0, 1) fades it by that
factor per further identical tick). See `neuron::Habituation`.

`trace=FILE` in `er_silence` writes the tick-by-tick silence time course.

`trace=FILE` (economy, paths, fatigue) writes a per-tick CSV of activity per path.

## Reading the results

- **Thresholding vs adaptation.** Compare `er` with `gate`, not only with
  `linear`: a fixed threshold halves activity too.
- **Path choice.** `drive_per_size` is a path's share of the total drive
  divided by its share of the neurons: 1.0 means no preference.
- **Switching.** `switch_rate` counts every change of the leading path;
  `internal_switch_rate` only those while the input stays the same, which a
  network whose output is a function of its input cannot do.
- **History.** For `gate` and `linear`, every `*_change` is exactly 0 by
  construction; any nonzero value for `er` is state carried in thresholds.
- **Spontaneous firing.** Only E-R neurons fire without input (|output| ≤
  0.01 once a threshold has decayed below 1e-10); `spontaneous_*` is
  counted for `er` only, since a small output of a linear neuron is decaying
  activity.
- **Habituation's cost.** Compare `accuracy_end` with `accuracy_sum`: a
  network that stops responding to a held stimulus saves spikes but no
  longer represents it at the end.
