# Research protocol

How new mechanisms and experiments are introduced, so that each result has
one explanation. The definitions the rules refer to are in
[model](model.md).

## 1. Controlled first, Doom last

Doom is an end-to-end validation environment, not the place to debug a
learning rule. A mechanism (a learning rule, a trace, a state readout, a
neuron option) goes into a Doom run only after a minimal synthetic
benchmark has shown what it does:

| Question | Controlled benchmark |
|---|---|
| Is the timing right (t vs t−1, feedback, recurrence)? | `tests/temporal.cpp` |
| Does an eligibility trace behave as documented? | `tests/traces.cpp` |
| Can a reward reach a synapse active d ticks earlier? | `delayed_credit` |
| Is past input stored in the state, readable from outputs, usable by a learned readout? | `memory_readability`, `video_memory`, `video_memory_learned` |
| Does E-R state help on time-varying input? | `dyn_ladder`, `nl_temporal` |

For Doom runs themselves:

- keep the perception (retina, pooling, cochlea) and the action set fixed
  between runs that are compared; a change to either starts a new series;
- keep the task definition (scenario, reward table, episode length) fixed
  within a series, and record it in the run's `config.json`;
- balance the action distribution where the method allows it (e.g. by
  reporting per-action counts) so a constant policy is not mistaken for
  learning;
- record every seed: network seed, evolution seed, game seeds for
  training, validation and test;
- separate training from evaluation: select on validation episodes, report
  on fresh test episodes with learning off;
- report the untrained network and a stateless baseline with the same
  interface next to every trained result;
- keep learning during an episode (online rules, `lr > 0` in play) apart
  from post-training performance (weights frozen), and say which one a
  number is.

[doom](doom.md) lists the setup and every Doom experiment.

## 2. Determinism

- Every experiment takes an explicit seed and records it: nntest's runners
  seed each trial (`exr.reseed(seed)`, and `t.rng` for NumPy) and write the
  seed into every JSONL line together with the git commit and the build
  (`build_info()`: compiler, build type, SIMD variant, OpenMP).
- The SIMD and scalar kernels give bit-identical sums; results do not
  depend on the thread count ([kernels](kernels.md#determinism)). A new
  kernel must keep that (and `KernelsTest` checks it), or state and test the
  difference.
- Update order is deterministic (topological, ties by creation order, or
  explicit). Keep it that way; a parallel update that changes results with
  the thread count needs a documented reason and a test that shows the
  difference.
- Do not trade determinism for a small speed-up.

## 3. No mechanism without a hypothesis

When an experiment performs poorly, first find out which of these causes
it, with a control for each:

1. temporal semantics (is the neuron learning from the value it used?
   `tests/temporal.cpp`, [model](model.md#one-tick));
2. representation (can any readout separate the classes from the hidden
   activity? a ridge probe);
3. state readability (is the information in the state but not in the
   outputs? `memory_readability`);
4. eligibility (does the trace or threshold still hold the event when the
   reward comes? `delayed_credit`, `tests/traces.cpp`);
5. credit assignment (does the update reach the right synapse, and not the
   distractors? `delayed_credit`'s relevant vs irrelevant weight changes);
6. learning rate and scaling (a learning-rate sweep; sums normalised or
   not);
7. topology (depth, width, recurrence, feedback).

A new neuron mechanism, activation, normalisation, adaptive parameter or
learning rule is added only with a written hypothesis ("X fails because
of cause k; mechanism M changes it because ...") and a controlled
experiment that tests it, run before and after, with balanced classes and
several seeds. Mechanisms added so far and their experiments are in the
[research log](research.md).

## 4. Reporting

- Balanced classes or actions, exactly, in every split, so a constant
  prediction scores chance; report balanced accuracy and the confusion
  counts next to accuracy, and the chance level.
- Several seeds (10 or more for claims), mean and spread.
- Name the controls: a reset ablation for anything attributed to state, a
  stateless model with an explicit history window for anything attributed
  to memory.
- Raw results go in `results/<topic>/` (gzipped nntest output) with the
  command that produced them (`sweep.sh` or `run.sh`); the research log
  links there.
