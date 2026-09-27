# Appendix 1: semantics and evidence audit

An audit of the current implementation and research state (2026-09-27),
answering twelve questions about execution semantics, topology, and what
the experiments in the [research log](research.md) (§13–§16, and the dynamic
ladder in [dynamic](dynamic.md)) actually show. Result files referred to
below are the experiment outputs kept with the project (`reports/`
directories named after each experiment).

Everything below was checked against the code on `main` (commit 178a3b3, PRs #1–#11 merged), the unmerged habituation/spontaneous-firing branch, the experiment result files, and small probe programs built against the library. Negative findings are reported as plainly as positive ones.

Audited 2026-09-27 · library code unchanged · probes: 1 vs 8 OpenMP threads, reseed, update-order swap, mid-run growth, sign-rule timing

Evidence tags: **[CODE]** read in the source · **[RAN]** verified with a probe or result file · **[INFERRED]** reasoned, not measured · **[UNTESTED]** no experiment or test covers it.

## Q1. Execution semantics of one `network::step()`

> **Answer.**
>
> `step()` is a sequential (Gauss–Seidel) sweep at layer granularity and a synchronous (Jacobi) update inside each wiring group. Layers run once each, in update order (`core/network.cpp:321`). A reader sees whatever value its source's output buffer holds at the moment the reader runs. Timing is therefore a property of the update order, not of the edge type: a "feedback" edge has no built-in delay.

### The rules, precisely

1. [CODE] **When a neuron reads another neuron's output.** A dense group first copies its whole input pool into a private buffer (`Group::gather`, `core/layers/dense.cpp:28`), then computes every row's weighted sum and calls `neuron::activate`. So a neuron reads the values its sources hold at the moment its own group starts.
2. [CODE] **Between layers:** a source that ran earlier in this tick is seen at *t*; a source that runs later (or the layer itself) is seen at *t−1*. The default order is a topological sort of `connect` edges with lowest-id-first tie breaking (`network.cpp:238`), so forward edges are same-tick and a feedback edge from a later layer to an earlier one is one tick late.
3. [CODE] **When a feedback connection becomes visible:** `addFeedback(from, to, w)` only adds `w` neurons to `to` reading `from`. It does not constrain the order. If `from` runs before `to` (e.g. feedback from an earlier layer to a later one, or any custom order that puts it first) the value arrives in the same tick.
4. [CODE] **Inside one layer:** groups run one after another (`dense.cpp:161`). A later group of a layer sees the *current* outputs of earlier groups of the same layer, while an earlier group sees the later group's previous-tick values. Within one group everything is simultaneous: the copy in `gather` makes a self-reading group see its own pre-tick outputs, and OpenMP chunking cannot change this.
5. [CODE] **Inside one neuron** (`core/neuron.cpp:84`): clamp the sum to ±10, apply habituation (compared with the previous raw sum), then E-R: fire if `|sum| > threshold`, output the signed sum, and grow the threshold; otherwise output 0, multiply the threshold by `recovery`, and fire spontaneously if it fell to the floor. The threshold used for this tick's decision is the one left by the previous tick.
6. [CODE] **Learning happens after the whole sweep.** `applyReward`/`applyError` run after `step()` and use each neuron's post-tick threshold. See the finding below for the timing mismatch this creates.

### Does update order change the result?

[RAN] Yes, whenever a loop (feedback, self-connection or a custom order) exists. In acyclic forward graphs the default order is the only sensible one and parallel branches cannot affect each other. Swapping the order of two mutually connected layers changes every value (table below). This is intentional and documented: `doc/network.md` "Update order" says the order defines the timing and uses it to build delay lines, and `doc/dense.md` says a later group sees earlier groups' updated outputs. What the docs do not state is the general rule that an `addFeedback` edge is same-tick whenever its source runs first.

### Can a neuron observe a partially updated state within one tick?

[RAN] Yes, in two documented situations. (a) Across layers, whenever some of a neuron's sources have run this tick and others have not: a neuron reading both an earlier and a later layer mixes *t* and *t−1* values in one weighted sum. (b) Inside one layer that has several wiring groups (constructor neurons plus `addFeedback(X, X, …)` neurons): the feedback group sees the first group at *t* and itself at *t−1*, while the first group (if it also reads X) sees the feedback group at *t−1*. There is no torn read inside a group.

### Worked example 1: three linear neurons, order A→B (default) vs B→A

Neurons without E-R or habituation so values are exact. `a` (layer A) reads the sensor x with weight 1. `b` (layer B) reads a with weight 1 (`connect(A,B)`). `f` is a feedback neuron added to A by `addFeedback(B, A, 1)`, reading b with weight 1. Input: a single pulse x=1 at t, then 0.

| tick | x | a (order A, B) | b (A, B) | f reads b (A, B) | a (order B, A) | b (B, A) | f (B, A) |
|---|---|---|---|---|---|---|---|
| t | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| t+1 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| t+2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

With the default order the pulse crosses the forward edge in the same tick (b=1 at t) and comes back through feedback one tick later (f=1 at t+1). With B first, b lags one tick and f receives b's value in the same tick b produced it, so f=1 at t+1 happens for a different reason: the forward edge became the delay. Same graph, same weights, different state at t.

### Worked example 2: mixed timing inside one layer

Layer X: neuron p reads x (weight 1) and, in the second variant, X itself through `connect(X,X)` (weight 1 on q, 0 elsewhere). `addFeedback(X, X, 1)` adds q reading [p, q] with weights [1, 0.5].

| tick | x | p | q = p + 0.5·q | p (also reads q) | q |
|---|---|---|---|---|---|
| t | 1 | 1 | 1.000 | 1.000 | 1.000 |
| t+1 | 0 | 0 | 0.500 | 1.000 | 1.500 |
| t+2 | 0 | 0 | 0.250 | 1.500 | 2.250 |

q sees p's value from this tick (q=1 at t, same as p) but its own value from the previous tick. In the right-hand variant p sees q from the previous tick. Both neurons are in one layer, joined by a "self" edge, yet the two directions have different delays.

### Worked example 3: one E-R neuron (linear growth, recovery 0.9)

| tick | input | threshold before | output | threshold after |
|---|---|---|---|---|
| t | 1.00 | 0.200 | 1.000 | 0.600 |
| t+1 | 1.00 | 0.600 | 1.000 | 0.800 |
| t+2 | 1.00 | 0.800 | 1.000 | 0.900 |
| t+3 | 0.50 | 0.900 | 0 | 0.810 |
| t+4 | 0.00 | 0.810 | 0 | 0.729 |
| t+5 | 1.00 | 0.729 | 1.000 | 0.865 |

Under a constant input the threshold climbs towards the input's magnitude but the neuron keeps firing (the output is the full sum, not sum minus threshold). The state that carries history is the threshold, which encodes roughly how strongly the neuron fired recently. A weaker input after a strong one is suppressed; that is the whole memory mechanism.

> **Finding: the default Sign rule learns from values the neuron never saw**
>
> [RAN] Sign-rule learning re-gathers each group's pool *at learning time* (`dense.cpp:201`), after the whole sweep. For a group that reads a layer running later in the order (feedback) or its own layer, those values are from tick *t*, while the forward pass used *t−1*. In a probe, feedback neuron f fired +1 on input +1, received reward +1, and its weight *fell* from 1.000 to 0.610 because the source had meanwhile turned −1. The trace-based rules (Trace, FA, Perturbation, Oja, BCM) use the inputs recorded during forward and are not affected. The recent suites (§13–§17) are feed-forward and do not hit this; older recurrent sign-rule experiments with learned feedback groups would have.

## Q2. Topology invariants

Every structural rule the framework enforces, with the reason it exists. "Semantics" means the configuration has no meaning; "execution model" means the tick model cannot express it; "implementation" means the model could express it but the code cannot do it safely; "design" means a deliberate choice.

| Invariant | Where | Why |
|---|---|---|
| A neuron belongs to at most one wiring group; groups are contiguous runs of neurons | `dense::addGroup` | Implementation (see below) |
| Each neuron has exactly one weight per entry of its group's pool; loaded weight counts must match | dense, `neuron_layer::deserialize` | Implementation: the weight matrix is laid out per group |
| Forward (`connect`) edges must be acyclic, self-edges excepted, unless a custom order is set | `topologicalOrder` | Execution model: the default order is derived by topological sort, and a cycle has none. A cycle is perfectly meaningful with a custom order (it just acquires a one-tick lag somewhere) |
| A custom order lists every layer exactly once; adding a layer makes `step()` throw until it is set again | `setUpdateOrder`, `updateOrder` | Design: one tick = one forward per layer. Running a layer twice per tick would be meaningful (sub-steps) but is not allowed |
| A custom order is never checked against the edges | (absence) | Design: needed for delay lines, but it means a later `connect` can silently acquire a lag |
| Each (from, to) forward pair is connected once | `network::connect` | Design: a second connect would add nothing (one weight per input already) and would make the edge list misleading |
| `connect` wires only neurons that exist now; feedback neurons added later do not read earlier sources | `dense::join` | Design, documented; it is how feedback neurons get a separate input set |
| Layer names unique and non-empty; input-source names unique; a source feeds a given layer once; sensor count > 0 | network | Design (names are the lookup and save key) |
| Learning rule, gate, habituation, growth, rectify only on layers made of neurons | `addLayer` | Semantics: other layers have no neurons |
| Gate and ReLU only on layers without E-R | `setGate`, `setRectified` | Design: E-R has its own threshold. A gate on top of E-R would be well defined |
| Parameter ranges: recovery [0.01, 0.999], habituation steps ≥ 1 and tolerance in [0,1), growth amount in [0, 1e6], gate finite ≥ 0 | neuron, neuron_layer | Semantics / numeric safety |
| Weights clamped to ±10 by learning (not by `setWeights`); sums clamped to ±10 before habituation | kernels, `activate` | Implementation: keeps recurrent loops finite |
| A layer's neuron count can change on load only if it is unwired and unread | `neuron_layer::deserialize` | Implementation: readers hold offsets into its output |
| No removal: no disconnect, remove-neuron or remove-layer operation exists | (absence) | Implementation: readers keep references and offsets; removal would need re-indexing |
| Networks cannot be copied or moved; a network is used from one thread | `network.hpp` | Implementation: layers reference each other's buffers |
| Input sizes: `setInputs` and `applyError` must match exactly | network | Semantics |

### Why a neuron cannot belong to two groups, and what would break

[CODE] The rule does not come from the mathematics. "A neuron sums inputs from several sources" is fully supported: `join` appends a new source to the neuron's existing group so all inputs go into one weighted sum. The restriction exists because `forward()` calls `activate()` once per group row, with that group's partial sum. If a neuron sat in two groups:

- `activate()` would run twice per tick on two partial sums: the E-R threshold would be tested and grown twice, habituation would compare the two partials with each other and count two "repeats" per tick, and the output slot would end with whichever partial came last. Nothing would ever compute the true total.
- Spontaneous firing and the neuron's random stream would advance twice per tick.
- Serialization stores one weight row per neuron, so one row would be lost on save, and the load-time weight-count check would reject the file.
- Learning would update both rows with one eligibility, which is harmless, but the input traces would be split.

Removing it safely would need a two-phase forward (accumulate all partial sums, then activate each neuron once), which is also what arbitrary per-neuron connectivity would need. The real limitation the rule creates is different: neurons created together share one input set, so sparse or per-neuron connectivity inside a layer can only be approximated with zero weights, and **learning does not preserve zeros** (the Sign rule moves every weight of an eligible neuron). There are no connectivity masks. [INFERRED]

## Q3. Arbitrary and pathological networks

| Configuration | Representable? | Explicitly tested? |
|---|---|---|
| Cycles | Yes. Feedback cycles by default; forward cycles only with a custom order | `ForwardCycleThrowsUnlessOrderIsSet`, `RecursiveTopologyAndCascadePropagation`, `StressTestFourLargeLayersWithCrossFeedback` (4×1000 neurons, cross feedback, checks sizes and no NaN) |
| Self-connections | At layer level (`connect(x,x)`, `addFeedback(x,x,n)`). A single neuron's private self-loop needs zero weights to all other neurons of its layer | `SelfConnectionIsIgnoredByOrdering`, `SelfConnectedLayerGrowthIsAppendedOnce`, `SelfFeedbackDoesNotExtendSensorOnlyNeurons`. Wiring only; no test of the resulting dynamics |
| Multiple feedback paths | Yes, any number of `addFeedback` calls between any layers | `MultipleAddFeedbackUpdatesListenersWeightsCorrectly`, the stress test. Wiring and no-NaN only |
| Zero-input recurrent subnetworks | Yes, a layer fed only by feedback or itself. Caveat: neurons in no group are never activated, so they never relax or fire spontaneously; they stay at output 0 forever | No unit test. `er_silence` ran recurrent nets with inputs set to zero (not absent): E-R activity died in 3–10 ticks |
| Disconnected components | Yes | Used in §7 (unconnected neurons as memory); no structural test |
| Strongly asymmetric update orders | Any permutation of layers; the order is not checked against edges | `CustomOrderBuildsDelayLine` only. No test compares results across orders |
| Dynamic topology changes | Growth only: `addLayer`, `connect`, `addFeedback`, `addInputs` at any time. No removal | Growth after running is exercised in two dense tests (sizes, no NaN). State preservation and behaviour after growth are not tested (checked by probe, see Q8) |
| No conventional input or output path | Yes: no sensors, or no output layers (then `outputs()` is empty and `applyError` takes an empty span) | [UNTESTED] |
| Multiple edges between the same pair, per-neuron sparse wiring, removal, weight sharing across dense layers | No | n/a |

In short, the structural tests prove the wiring bookkeeping is right in pathological graphs. None of them checks the *dynamics* of such graphs (order sensitivity, stability, E-R under recurrence), and none of the research suites since §13 uses recurrence at all.

## Q4. Strict causal decomposition

For each observed behaviour: which ingredients are present, which were removed by a control, and what the result therefore licenses. "Recurrence" and "feedback" are absent from every network in §13–§17 (the activity, nonlinearity and ladder networks are strictly feed-forward), so nothing recent can be attributed to them.

| Behaviour | Attributable to | Control that isolates it | Not yet separated from |
|---|---|---|---|
| ~50% sparsity without a penalty (§13) | **Thresholding**, not E-R state | Fixed gate matched to the same activity gives the same sparsity; activity does not fall with training | Nothing; this one is settled |
| Fatigue "rerouting" (§13) | **E-R state of the stimulated path only**, plus a normalized metric | See Q6: the other paths are bit-identical to rest | Settled by this audit: not a network effect |
| Leading path switches under a constant input (§13) | E-R state (independent per-neuron threshold oscillations) read through an argmax of per-size drive | Gate and linear never switch | Any per-neuron oscillator would do the same; paths do not interact |
| Same input, different response after different recent history (`er_history`) | E-R state | Stateless gate and linear (trivially 0) | Any stateful neuron (leaky integrator, adaptation current) |
| Extra ticks per sample reduce static error 2–4× (§14) | E-R state evolving within a presentation | Other models do not change with ticks (checked) | More compute and spikes per answer |
| 30–57% of static-task error is history variance (§15) | E-R state carried across samples | Static models have exactly 0 variance | Nothing; it is a cost, not a benefit |
| Delayed XOR solved, t2 and parity-4 partly (§16) | E-R state, specifically its magnitude-tracking part | `er_memoryless` (same neurons reset each step) stays at the ceiling; fixed and multiplicative growth fail | Memory per se: ReLU with one past frame also solves it. Whether hidden-layer learning is needed was not tested |
| Change detection 0.98 without a window (§17 ladder) | E-R state | Reset ablation drops to the 0.70 ceiling | Same as above: ReLU + last frame reaches 1.00 |
| E-R hidden layers learn under FA where a matched gate does not (0.995 vs 0.76–0.80) | Interaction of E-R and the learning rule | None | E-R forward dynamics vs E-R's wider eligibility window (see Q7, Q10) |
| Memory in unconnected E-R neurons 0.72 vs 0.50 (§7) | E-R state | E-R off | —; frozen **recurrence** does better (0.94), and E-R on top of recurrence slightly hurts: a negative interaction |
| Recurrent activity dies in 3–10 ticks with E-R, rings 600–1300 ticks linear (§15) | Interaction: E-R thresholds damp recurrence | Linear recurrent net | Gate or ReLU recurrent controls not run |
| Spikes saved up to 90× on held stimuli | **Habituation**, not E-R | Works on linear neurons too | — |
| Deep E-R fails (depth ≥ 4 at chance), 2-layer E-R diverges | Interaction of E-R and FA learning (cause unknown) | None | Hold length, eligibility gating, sign-symmetric transfer |

**Bottom line:** every E-R-specific effect found so far is a *single-neuron* effect of the threshold state. No result depends on recurrence or feedback connections, and the only effect that involves learning (FA through a thresholded layer) has not been decomposed. No emergent network-level phenomenon has been demonstrated.

## Q5. Strongest evidence of history dependence

> **Answer.**
>
> **Delayed XOR (`nl_temporal` t1) with the `er_memoryless` ablation.** On a ±1 stream the current input x(t) alone predicts the target at chance, so any correct answer requires the same x(t) to produce different outputs depending on internal state. The ablation keeps the same neurons, architecture, learning rule and budget and only resets the state before each step.

| Model (1 hidden layer) | Growth rule | Test accuracy per seed (5 seeds, lr chosen on validation) |
|---|---|---|
| er, 64 neurons | log | 0.951 0.952 0.957 0.962 0.973 |
| er, 8 neurons | linear (current default) | 1.000 1.000 1.000 1.000 1.000 |
| er_memoryless, 64 neurons | log | 0.496 in all 5 seeds |
| relu / gate / clamp, 64 neurons | — | 0.49–0.50 in all seeds |
| Memoryless ceiling |  | 0.51 |

[RAN] Recomputed from `reports/nonlinearity/temporal/t1.jsonl` and `growth/temporal.jsonl`. The effect survives every seed, and the separation is huge (0.95–1.0 vs 0.50). Metric: test accuracy on 2000 held-out steps.

**Limits.** Five seeds only. The memoryless ablation was run only with the log rule, so the headline "8 neurons with linear growth" has no matched ablation. The ablation is trained separately, not the same trained weights with state reset at test time. And this shows history dependence exists, not that it is special: ReLU given one past frame solves the same task with 8 neurons.

### Runner-up evidence

- `er_history`, 10 seeds, same trained network, 20 identical probes after each condition: after stimulating every path, at least one decision changed in 9/10 seeds (recovery 0.9) and 10/10 (0.97); mean decision change 0.11 and 0.20. After ordinary task use ("busy"), decisions changed in 0/10 seeds, although firing patterns changed by about 0.27. The control is stateless, so it only rules out "no state".
- `nl_static` state probes, 10 seeds: variance across 20 histories is 6–14% (1 extra tick) and 30–57% (7 ticks) of error; exactly 0 for static models.
- Ladder change detection, 5 seeds: 0.98 vs 0.70 for the reset ablation.

## Q6. The minimal mechanism behind fatigue-driven redistribution

> **Finding: there is no redistribution**
>
> [RAN] In `er_fatigue` the three paths never interact: each reads only the task input and its own stimulation sensors, and the readouts are frozen during the probe. I compared the non-stimulated paths' per-sample active fraction and mean threshold between the "rest" and "A/B/C fatigued" conditions in `er_fatigue_trace.csv`: **all 2,400 comparisons are identical**. The unfatigued paths do exactly what they do after rest. The reported `others_share > 1` is computed as `(1 − fatigued share) / (1 − rest share)` (`er_fatigue.cpp:161`), so it rises by construction whenever the fatigued path's share falls.

So the minimal mechanism for what was measured is:

- **Required:** an activity-dependent state that suppresses a recently driven neuron and decays (E-R's threshold; any refractory or adaptation variable would do), plus a readout that sums several redundant paths so accuracy survives the loss of one. The redundancy explains why the answer survives; it is not redistribution.
- **Not required:** recurrence, learning (the probe runs at lr 0), competition between paths, or magnitude-tracking growth.
- **Not sufficient:** heterogeneous fixed thresholds. They have no state, so pre-stimulation leaves no trace.

### Smallest discriminating experiment

Report absolute drive and activity per path (not shares), then add the one thing redistribution needs: a coupling through which A's silence can change B. Three conditions, same trained weights, same fatigue protocol:

1. Independent paths (today): predicted ΔB = 0 exactly.
2. Paths coupled through a shared inhibitory or normalizing layer (a small layer reading all paths with negative feedback into each).
3. Paths coupled through recurrent feedback from the readouts to the paths (`addFeedback(readout, path, k)`), trained.

Run each with E-R, with gate plus a fixed refractory period, and with the Fixed growth rule. Genuine redistribution means B's absolute activity rises after A is fatigued; if it appears with the refractory gate too, it is generic adaptation, not E-R.

## Q7. Learning and E-R

### What depends directly on the threshold

| Rule | Uses the E-R threshold? | How |
|---|---|---|
| Sign (default) | Yes, twice | Eligible if threshold > 0.2 (fired recently); step size × (threshold / 0.2 − 1) (`neuron.cpp:161–176`) |
| Feedback alignment | Yes, as a gate | Surrogate derivative 1 while eligible (threshold > 0.2), else 0; the step size does not depend on the threshold (`neuron_layer.cpp:423`) |
| Trace, Perturbation, Oja, BCM | No | Only through the neuron's output (post-synaptic trace) |

[INFERRED] After any real firing the threshold is at least 0.4 and decays by 0.9 per quiet tick, so a neuron stays eligible for at least 7 ticks (longer after a strong firing). E-R eligibility is therefore a decaying trace of recent firing magnitude. A gate neuron is eligible only on ticks it fires.

### Could a static neuron reproduce the learning results?

For the **learning signal**, yes by construction. A static neuron with a "shadow threshold" (the same growth and decay equations, driven by its output, used only for eligibility) would get identical Sign and FA updates for the same firing pattern. For the **learning results**:

- Static tasks: ReLU beats E-R everywhere, so nothing needs reproducing.
- Temporal and change tasks: the gain comes from the forward state (the reset ablation keeps E-R eligibility within a presentation and still fails). A static neuron with any eligibility rule cannot reproduce it without a window.
- FA through a thresholded layer (er_economy 0.995 vs 0.76–0.80): this is the one result a static surrogate might reproduce, and no experiment distinguishes the two explanations.

### Which result distinguishes dynamic state from a static surrogate?

Only the temporal results (delayed XOR, change detection), and they distinguish forward state, not learning. No existing result distinguishes E-R's role in learning from a static neuron with an eligibility trace. The clean test is a 2×2 on er_economy and t1: {E-R forward, gate forward} × {E-R eligibility via shadow threshold, instantaneous eligibility}. [UNTESTED]

## Q8. What happens when topology changes during execution

Checked in the code and with a probe that grew a running E-R network (`addFeedback` after 5 ticks).

| Item | After a mutation |
|---|---|
| Existing neuron state | [RAN] Kept exactly (threshold, habituation streak, output, own random stream) |
| New neurons | Start at rest (threshold 0.2, output 0). Recovery, gain and alpha drawn from the layer's jitter; spontaneous seed from the global stream. Layer settings (gate, ReLU, habituation, growth, learning rule, frozen) inherited |
| Weights | [RAN] Existing rows kept. **Every reader of a grown layer gets random weights (uniform ±1) for the new outputs**, and `connect` gives every existing neuron random weights for the new source. A trained function is perturbed as soon as the new neurons fire; nothing starts at zero |
| Groups | `addFeedback` creates a new group; `connect` appends to all existing groups, including earlier feedback groups. One-group-per-neuron still holds |
| Feedback timing | Determined by the update order, which `addFeedback` does not touch |
| Update order | Default: recomputed after `addLayer`/`connect`. Custom: stays in force after `connect`/`addFeedback` without any check, so a new forward edge can silently become a one-tick delay; `addLayer` makes `step()` throw |
| RNG | New weights come from the process-wide Initial/Growth/Sensor streams, so any mutation shifts every later draw in the process |
| Learning state | Per-neuron vectors (bias, traces, baselines, BCM θ) extended with initial values; input traces get zero columns; FA feedback rows for new neurons are drawn lazily. **If the number of network outputs changes, every FA layer's feedback matrix is redrawn from scratch** (`neuron_layer.cpp:332`). Perturbation noise shifts for all neurons after the new ones |
| Serialization | Every mutation is appended to the build log and replayed on load, so topology round-trips. Settings applied directly on a layer object (not through the network) are not recorded and are lost on save/load (found by the learning-rules thread) |

### Guaranteed after a mutation

- Every neuron is in at most one group, and each group has one weight column per pool entry.
- Existing neuron indices and output positions are stable: growth only appends.
- Every reader of a grown layer, in registration order, reads the new outputs.
- Sensors never grow; History layers give new channels an all-zero past.

### Not guaranteed

That the network computes the same function; that a custom order still matches the edges; that learning state (FA matrices) survives an output change; that direct layer-level settings survive save/load. Mid-run growth has no behavioural test. [UNTESTED]

## Q9. Reproducibility

> **Answer.**
>
> Yes for the conditions you list, including thread count, **provided** `exr::reseed(seed)` is called immediately before construction and the same binary and standard library are used. [RAN] A network of about 4,200 E-R neurons with feedback, stepped and trained for 30 ticks, gave the same output hash with 1 and 8 OpenMP threads, and again after reseed. Sums always accumulate from 0 in input order, reductions (conv2d kernel learning) sum fixed chunks in order, and the build disables FMA contraction.

### Every remaining source of nondeterminism

1. **Process-global random streams.** Weights, growth, sensors, spontaneous seeds, jitter and FA matrices come from six process-wide generators. [RAN] Building a second identical network without reseeding gives different weights. Loading a saved network consumes the streams too, so a network built after a load differs (known, disabled test `DISABLED_LoadingDoesNotChangeNetworksBuiltAfterIt`). Two networks built in an interleaved way depend on the interleaving.
2. **Standard-library distributions.** `std::uniform_real_distribution` and `std::normal_distribution` are implementation-defined; the same seed gives different weights with libstdc++, libc++ (macOS) and MSVC. [INFERRED]
3. **libm.** `log` (log growth), `pow` (habituation fade), `sin/exp` (task targets) may differ in the last bit across libm versions.
4. **Compiler flags outside the core.** `-ffp-contract=off` is applied to the core library only, not to experiment code, and not on MSVC. Task generation compiled with `-march=native` may fuse multiply-adds.
5. **Save/load gaps.** Direct layer setters are not saved. `WeightsOnly` loading keeps each neuron's current spontaneous generator instead of the file's. Files are native-endian and use native `size_t`, so they are not portable.
6. **Python and game experiments.** ViZDoom and Python-side random generators are outside the library's streams (not audited in detail). Wall-time metrics (`inference_us`, `seconds`) are inherently variable.

E-R amplifies any of these: firing is a discontinuous comparison, so a last-bit difference can flip a firing decision and the trajectories then diverge. Bit-exact cross-platform reproduction should not be expected; statistical reproduction across seeds is what the suites rely on.

## Q10. Experiment contamination

Cases where more than one relevant mechanism differs between condition and control.

| Experiment and claim | What else changed | Severity |
|---|---|---|
| er_economy: "E-R learns with FA, matched gate does not" (0.995 vs 0.76; rerun 0.999 vs 0.80) | (1) Eligibility window: E-R neurons take FA steps for ≥ 7 ticks after firing, gated neurons only while firing, a **learning-rule difference**. (2) The gate is calibrated on the *untrained linear* network's activity against the *trained* E-R network's. (3) Training accuracy learns from the true label on every tick of the sample before deciding, which leaks the label: the gate scores 0.97–0.98 in training and 0.76–0.80 on test, so its training curve is not evidence | High: the only E-R learning advantage is unattributed |
| er_fatigue: "other paths take over" | A normalized share metric, not a mechanism (Q6) | High: the claim should be withdrawn |
| er_history, path switching: "only E-R" | Controls are stateless, so this is "stateful vs stateless", not E-R vs alternatives | Medium |
| nl_temporal: "only E-R beats the memoryless ceiling" | Memory vs no memory. Resolved by the ladder's window baseline (ReLU + 1 frame solves t1 and beats E-R on 3 of 4 tasks). `er_memoryless` correctly separates state from transfer function | Resolved, but §16's framing overstates it |
| Growth rules: "linear solves t1 with 8 neurons" | The rule changes threshold magnitudes and hence FA eligibility (learning) as well as the memory; no memoryless control under linear | Medium |
| nl_static: "7-tick E-R beats the gate on x1·x2 (0.029 vs 0.088)" | 7 extra ticks of computation and more spikes; the gate is fixed at 0.2, not tuned or matched; E-R's transfer is sign-symmetric while ReLU is one-sided | Medium |
| Ladder: "E-R + window is worse than ReLU + window" | E-R vs ReLU also differ in rectification (E-R passes negative sums); a gate + window or E-R-with-rectification control would separate them | Low (the direction is negative anyway) |
| §7: E-R on vs off in unconnected neurons | E-R off also removes thresholding and changes eligibility. Linear neurons have no memory at all, so the memory claim holds, but the size (0.72) mixes thresholding and state | Low |
| Parameter count | Equal trainable parameters in every comparison, but E-R carries 2 hidden state variables per neuron that the baselines lack; "state per neuron" is never equalized | Medium: central to Q11 |
| Training time | Same sample budget and per-model learning-rate choice. The learning-rules thread found 80k samples lifts E-R on t2 (0.74 → 0.84); baselines were not given the same extension (they are at their ceiling, so the comparison survives) | Low |
| Rerun with habituation off / cut after 5 / fade after 2 | Habituation variants are a second mechanism layered on E-R; any "E-R" result from a habituation-on run is E-R plus habituation | Keep the "off" rows as the E-R reference |

## Q11. Complexity transfer from topology to neuron dynamics

### Evidence for

- A one-layer E-R network with no delay line solves delayed XOR (8 neurons, linear growth) and change detection (0.98), which need a one-step delay line in a stateless network. That is one tap of topology replaced by neuron state.
- Unconnected E-R neurons carry sequence memory (0.72 vs 0.50) without recurrence.
- With more ticks per sample E-R represents x1·x2 better than a fixed threshold at the same size: time traded for nonlinearity.

### Evidence against

- Static functions: no E-R network of any size reaches MSE 1e-3; ReLU needs 5–129 neurons. E-R buys some nonlinearity only by spending ticks and spikes.
- The memory is about one step: x(t−3) partly, parity beyond 4 not at all, continuous history (t4, velocity) not at all.
- When topology supplies the memory cheaply (one past frame, 16 extra inputs), ReLU matches or beats E-R on 3 of 4 ladder tasks, and adding E-R to that network makes it worse (catch 0.60 vs 1.00).
- Frozen random recurrence is a better memory than E-R (0.94 vs 0.72), and E-R on top of recurrence hurts.
- Depth does not compose: E-R networks of depth ≥ 4 stay at chance and 2-layer ones often diverge.
- The one network-level effect claimed (rerouting) turns out to be a metric artifact.

**Reading.** The complexity transferred so far is quantifiable and small: roughly one delay-line tap per neuron, used as a magnitude-coded novelty filter. Delayed XOR on a ±1 stream and change detection are the same computation (did the input change?), so the two positive temporal results are one capability, not two.

### Strongest next test: an exchange-rate experiment with state-matched baselines

Tasks with a controlled lag L = 1, 2, 4, 8 (x(t) XOR x(t−L), change detection over L, and a graded version). For each model, find the smallest network that solves it:

- ReLU with a window of W frames (topology memory), W = 0…L;
- E-R with W = 0;
- two state-matched controls with the same number of state variables per neuron as E-R: a leaky-integrator ReLU and a ReLU with a subtractive adaptation variable.

The output is an exchange rate (window taps and neurons saved per unit of neuron state) as a function of L. If E-R's rate beats the state-matched controls, its specific dynamics matter; if the controls match it, the transfer is generic to "neurons with state"; if the rate collapses beyond L = 1, the answer to the research question is "about one tap".

## Q12. Five things we do not understand

1. **Is anything about E-R specific, or would any stateful neuron do?** Every positive result was against stateless controls or a reset ablation.
   *Experiment:* add a leaky-integrator ReLU and an adaptive-threshold ReLU (one state variable each) to `nl_temporal` t1/t2 and the ladder's change and direction tasks. One run of the existing grid with two extra models.
2. **Why E-R fails in depth and diverges.** Depth ≥ 4 stays at chance on t1; two layers diverge under FA even at lr 0.001. Candidates: hold length (depth + 2 ticks may not let upper layers settle), eligibility gating starving upper layers, the sign-symmetric transfer.
   *Experiment:* t1 at depth 1–4 with settle 0/2/6, logging per-layer eligible fraction and weight norm per 1k samples, plus one variant with frozen random lower layers. The logs should say which one breaks first.
3. **What information the E-R state actually holds, and whether readout or state is the bottleneck.** We know t1 works and parity-8 does not, but not whether x(t−3) is absent from the state or merely not learned.
   *Experiment:* a memory-capacity curve. Drive an untrained and a trained E-R layer with a random stream, fit linear regressions from thresholds and outputs to x(t−k) for k = 1…10. No training loop needed; minutes to run.
4. **Why E-R helps FA learning in a thresholded layer.** Forward dynamics or the wider eligibility window?
   *Experiment:* the 2×2 from Q7 on er_economy: gate or E-R forward, crossed with E-R-style (shadow threshold) or instantaneous eligibility. Report test accuracy only, since training accuracy leaks the label.
5. **Whether any genuine network-level E-R phenomenon exists.** All confirmed effects are single-neuron; recurrence has only ever been shown to die out under E-R, and update-order effects have never been measured on a trained network.
   *Experiment:* the coupled-paths fatigue test from Q6 (independent vs shared inhibition vs trained feedback), reporting absolute activity of unfatigued paths, run with both update orders to measure how much results depend on the order.
