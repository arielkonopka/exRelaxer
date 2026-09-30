# memory_readability

A LEFT or RIGHT event, a blank period, then a query input that is the
same for both classes. Is the direction (1) stored in the hidden layer's
E-R state, (2) readable from the hidden outputs (all a downstream neuron
sees), (3) usable by a readout that learns online with the library's own
rule? Results: research log §25. The larger, visual version of the same
question is [video_memory](../video_memory/README.md).

```
event  1 tick        L = 1 or R = 1
blank  `blank` ticks  all inputs 0
query  1 tick        Q = 1            (identical for LEFT and RIGHT)
+ Gaussian noise of sd `noise` on every input, every tick

inputs [L, R, Q] -> hidden (64 neurons, frozen random weights in [-1, 1]) -> readouts at the query tick
```

Readouts, fitted on the same episodes: `output` (hidden outputs at the
query), `state` (thresholds after the query tick), `output_state` (both),
`eligibility` (the Sign rule's eligibility), `state_before` (thresholds at
the end of the blank). Each by ridge regression (best linear readout,
offline) and by an online library readout (two plain neurons with a bias,
`apply_error`, feedback alignment on an output layer = the delta rule) on
standardised features. The state readouts use the library's read-only
probe (`Network.state_probe`); downstream neurons in a network cannot read
thresholds.

Controls: `model=reset` (E-R reset to rest right after the event: nothing
can be stored, every readout must be at chance), `model=relu` (stateless).
Classes are exactly balanced in every split; metrics include accuracy,
balanced accuracy and the confusion counts (`*_tp`, `*_fn`, `*_fp`,
`*_tn`, LEFT positive) for each readout, and `chance`.

```bash
./build.sh --python
export PYTHONPATH=$PWD/build/EXrelaxer.py/package
NNtesting/nntest.py run memory_readability --trials 20 --set model=er,reset,relu --set blank=0,4,32
NNtesting/experiments/memory_readability/sweep.sh results/memory-readability   # every run behind §25
python3 NNtesting/tools/temporal_summary.py results/memory-readability
```

## State layer

The thresholds (B, C, D) and habituation streaks (H, D) are read from a
State layer on the hidden layer, as a downstream neuron would read them
([doc/state_output.md](../../../doc/state_output.md)); `habituation=1`
gives the hidden neurons habituation (`hab_steps`, `hab_tolerance`).
Results: research log §26.
