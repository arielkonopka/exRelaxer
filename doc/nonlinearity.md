# Dynamic nonlinearity substitution

Can E-R neuron dynamics stand in for network topology? These experiments
ask how large a network must be to reach a **fixed** error on a task,
depending on the source of its nonlinearity. They are designed so that "no"
is as clear an answer as "yes". Milestone 1 covers static functions
(`nl_static`); the second step adds temporal tasks and a memoryless E-R
control (`nl_temporal`). Parameter- and neuron-matched comparisons and
robustness tests follow. Results are in the
[research log, §14](research.md#14-dynamic-nonlinearity-substitution-static-tasks)
and §16.

## Models

Every model is the same stack: `x → hidden 1 → … → hidden depth → output`,
with equal-width dense hidden layers and one linear output neuron. Only the
hidden neurons differ:

| Model | Hidden neuron | How it is built |
|-------|---------------|-----------------|
| `relu` | conventional: `max(0, sum)` | `LayerSpec::rectify` (added for this baseline; off by default) |
| `er` | production E-R, habituation off | `LayerSpec::Dense(width, false, true)`; nothing changed in E-R |
| `er_memoryless` | the same E-R neurons, state reset to rest before every presentation | a test-only wrapper in the experiment (`Mlp::resetHiddenState`, a weights-only round trip through the neuron's own format); production E-R is unchanged |
| `gate` | static threshold: `sum` if \|sum\| > 0.2, else 0 | `LayerSpec::gate` = E-R's resting threshold (`baseline_threshold`) |
| `clamp` | the library's plain neuron: `sum`, clamped to ±10 | the default; nearly linear for these inputs (`linear` is an alias) |

`clamp` is kept as the "revert to clamping" option and as a control that
cannot represent products or sines. E-R's odd, sign-preserving response is
shared with `gate`: both need their biases to represent even functions
such as `x1·x2`.

**Everything else is the same for every model**:

- data: `data_seed`;
- weights: the library's uniform draw from the trial seed, scaled to
  variance 1 / fan-in, so each seed gives every model the same initial
  weights for a given architecture;
- learning: feedback alignment with a learned bias per neuron, the delta
  rule on the output;
- training budget and stopping rule;
- learning-rate grid.

**Presentation.** Each sample is held for `depth + 1 + settle` ticks (the
input needs `depth + 1` ticks to reach the output) and the output is read
on the last one. While training, that output's error is applied once. E-R
state carries on from one sample to the next, as it would in use.

**E-R configuration** (recorded in every result line): alpha 1.2, resting
threshold 0.2, threshold after any firing ≥ 0.4, growth
`threshold + alpha·ln(|s|/threshold)` (the log rule, the default when
milestone 1 and §16 were run; linear growth is the default since, so pass
`growth=log` to reproduce them), recovery 0.9 per silent tick,
habituation off, learning gain 2. Spontaneous firing is part of production
E-R and cannot be switched off without changing it. It happens only after
about 200 silent ticks (threshold ≤ 1e-10), far longer than a sample.

**Options for E-R studies** (off by default, so the comparisons above are
unchanged): `growth` / `growth_amount` pick the threshold growth rule
(`log`, `linear`, `fixed`, `multiplicative`; see
[neuron](neuron.md#excitationrelaxation-e-r)), and `learn_ticks=all`
learns on every tick of a presentation at `lr / ticks` instead of once from
the last tick.

## Tasks (`nl_static`)

`x` is uniform in [-1, 1]^d.

| Task | Target | d |
|------|--------|---|
| `l0` | x1 + x2 | 2 |
| `l1` | x1 · x2 | 2 |
| `l2` | sin(x1 · x2) | 2 |
| `l3` | sin(x1 · x2) + exp(−x3²) | 3 |
| `l4`, K ∈ {1, 2, 4, 8, 16} | K^−½ Σₖ sin(aₖ·x + bₖ), aₖ ∈ U[−2, 2]^d, bₖ ∈ U[−π, π] | 2 |

The `l4` coefficients come from `task_seed` (12345), are the same for every
model and seed, and are written into every result line (`task_a<k>_<j>`,
`task_b<k>`). The K^−½ keeps the target's variance near 0.5 for every K.

## Temporal tasks (`nl_temporal`)

The input is a stream. Each step `x(t)` is held for `depth + 1 + settle`
ticks and the output is read on the last one, so the feed-forward pipeline
has flushed the previous step: a model without neuron state sees only
`x(t)`. With `window=W`, every model also gets `x(t−1) … x(t−W)` as extra
inputs (a tapped delay line), which turns the task into a static one.

| Task | Target | x(t) |
|------|--------|------|
| `t1` | x(t) XOR x(t−1) | {0, 1}, shown as ∓1 |
| `t2` | x(t) AND NOT x(t−3) | {0, 1} |
| `t3`, n ∈ {2, 4, 8, 16} | parity of x(t−n+1) … x(t) | {0, 1} |
| `t4` | sin(x(t) · x(t−2)) | U[−1, 1] |

Training is one continuous stream (`data_seed`); every step's error is
applied once. Validation (`data_seed + 1`) and test (`data_seed + 2`) are
fresh streams run continuously; the first `warmup` steps (20) are not
scored. **Success**, fixed in advance: accuracy ≥ 0.95 on `t1`–`t3`, MSE ≤
1e-3 on `t4`. Each binary result also records `input_ceiling_accuracy`
(the best any function of the shown inputs can do on the test stream: the
limit for models without state) and `chance_accuracy` (always the majority
answer).

```bash
nntest run nl_temporal --threads 1 --trials 5 --set task=t1 \
    --set model=relu,er,er_memoryless,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
    --set depth=1,2,3,4,6,8 --set width=4,8,16,32,64,128 --out t1.jsonl
```

## Protocol and success

- Training: up to `train` samples (20 000) from the training stream. Every
  `eval_every` samples (1000) the validation MSE (500 samples) is measured,
  and training stops once it is ≤ `target_mse` (`early_stop=false` keeps
  training for the whole budget, for learning curves).
- Test: 1000 samples, presented once.
- **Success**: test MSE ≤ 1e-3, fixed before any comparison.
- **Solved**: an architecture solves a task when ≥ 80 % of its seeds
  succeed.
- **Learning rate**: a fixed grid {0.001, 0.003, 0.01, 0.03} is run for
  every model. For each model and architecture, the rate with the lowest
  median *validation* MSE is used. The rule is the same for every model and
  never looks at the test set. The pilot showed why one rate for all would
  not be fair: 0.01 is ReLU's good rate, and there E-R, gate and clamp
  networks diverge.
- **Grid**: depth {1, 2, 3, 4, 6, 8} × width {4, 8, 16, 32, 64, 128}.
  Exploratory runs use 5 seeds; final runs use 10.

## Measurements

Each trial writes one nntest JSONL line with the parameters, the seed, and
the machine, compiler, build and git commit.

| Group | Metrics |
|-------|---------|
| Error | `test_mse`, `test_nmse` (MSE / target variance), `validation_mse`, `initial_validation_mse`, `success`, `training_samples` |
| Size | `hidden_neurons`, `neurons` (+ output), `parameters` (weights + biases), `hold_ticks` |
| Activity (test set) | `active_neurons_mean` (per tick), `active_fraction`, `spikes_per_sample`, `unique_neurons_per_sample`, `fraction_of_neurons_used`, `never_active` |
| Cost proxies | `dense_synops_per_sample` (every connection, every tick), `event_synops_per_sample` (active sources × fan-out), `inference_us` (wall time). These are proxies only, not energy. |
| E-R state | `threshold_mean`, `threshold_p10/p50/p90` (end of each test sample) |
| Learning curve | `curve_<n>`: validation MSE after n training samples, `best_validation_mse`, `best_validation_at` |
| State test (`state_probes` > 0) | each of `state_probes` test inputs is presented after `state_histories` (20) different random histories of `state_history_len` (10) samples: `state_variance` (output variance across histories), `state_bias2`, `state_share` (variance / MSE: the part of the error that comes from state), `state_output_range`, `state_spike_sd`. Always 0 for models without state. |
| Configuration | E-R and neuron constants, `task_complexity`, the l4 coefficients |

A neuron is **active** on a tick when |output| > `firing_epsilon`. For
`relu`, that means a positive sum; for `clamp`, practically always.

**Trace mode**: `--set trace=FILE` appends every hidden neuron's output and
threshold on every tick of the first `trace_samples` test samples. The
output is deterministic.

## Running and analysing

```bash
nntest run nl_static --threads 1 --trials 10 --set task=l4 --set k=1,2,4,8,16 \
    --set model=relu,er,gate,clamp --set lr=0.001,0.003,0.01,0.03 \
    --set depth=1,2,3,4,6,8 --set width=4,8,16,32,64,128 --out nl.jsonl
python3 NNtesting/tools/capacity.py nl.jsonl --out report/
```

`capacity.py` finds, for each task and model and each on its own:

- minimum depth, width, neuron count and parameter count;
- at the smallest solving network: its error, spikes, active neurons,
  synaptic operations and inference time;
- the best error on any architecture.

It writes `capacity.md`, `architectures.csv` (every architecture: the
chosen rate, success fraction and medians) and the l4 capacity curves
(`capacity_neurons.svg`, `capacity_params.svg`). `--target` and
`--fraction` re-analyse the same runs at another threshold, and `--lr`
fixes the rate. No measurement is folded into a single score, and no model
is declared the winner.
