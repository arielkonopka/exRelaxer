# exrelaxer documentation

Detailed reference for each class. For building, a quick start and the list
of known limitations, see the [project README](../README.md).

| Page | Covers |
|------|--------|
| [neuron](neuron.md) | `neuron`, `DeserializeMode`, the tunable constants: E-R, habituation, learning rule, serialization format |
| [layer](layer.md) | `layer` (abstract interface), `LayerType`: the contract every layer type implements |
| [dense](dense.md) | `dense`: wiring groups, growth propagation, parallel forward pass |
| [layer_factory](layer_factory.md) | `layer_factory`, `LayerSpec`: creating layers by type, registering new types |
| [network](network.md) | `network`: layer graph, inputs/outputs, update order, freezing, save/load |
| [pattern_benchmark](pattern_benchmark.md) | Test support (`tests/pattern_benchmark.hpp`): gapped-pattern benchmark, frozen value detectors, delay window |
| [research](research.md) | Research log: parameter changes, mechanisms, topologies, jitter and what each showed |

## How the pieces fit together

```
network ──owns──> layer (via layer_factory, by LayerType)
                    └── dense ──owns──> neuron × N
                                          (weights, threshold, habituation state)
```

- A **neuron** turns a vector of input values into one output value and keeps
  its own adaptive state (E-R threshold, habituation counter).
- A **layer** is a set of neurons plus the wiring that decides which values
  each neuron reads. `dense` is the only layer type so far.
- A **network** owns layers, records how they are connected, owns the input
  sensors, decides the order in which layers run, and saves/loads everything.
- **layer_factory** lets the network create layers from a `LayerSpec`
  without knowing their concrete class.

### Composition

What owns what (filled diamond: owns; open diamond / dashed: refers to).
Note the shared `float` slots: a neuron owns its output slot, and every
reader holds a `shared_ptr` to it.

![Object composition](diagrams/composition.svg)

### Class diagram

![Class diagram](diagrams/classes.svg)

### Interactions

| Diagram | Shows | Details in |
|---------|-------|------------|
| [One tick](diagrams/sequence_step.svg) | `setInputs` → `step` → `outputs` → `applyReward` | [network](network.md#running), [dense](dense.md#forward) |
| [Feedback growth](diagrams/sequence_growth.svg) | how `addFeedback` adds neurons and extends every reader | [dense](dense.md#growth-propagation) |
| [Save and load](diagrams/sequence_save_load.svg) | recording and replaying the construction history | [network](network.md#serialization) |

### Diagram sources

The diagrams are PlantUML sources in [diagrams/](diagrams/) (`*.puml`),
rendered to SVG. After editing a source, regenerate with:

```bash
cd doc/diagrams && plantuml -tsvg *.puml
```

## Core concepts

### Values are shared, not copied

Every neuron's output lives in a `std::shared_ptr<float>`. A layer that
reads another layer holds copies of those pointers, so it always sees the
current value without being notified. Input sensors are the same kind of
pointer, owned by the network (or by the caller when using layers directly).

### A tick and its timing

One tick is one call to `forward()` on each layer, in some order (in a
network: one `network::step()`). A layer reads whatever values its sources
hold **at the moment its own `forward()` runs**:

- a source that already ran this tick delivers this tick's value;
- a source that runs later delivers last tick's value.

So the update order *is* the timing model. `network` defaults to the
topological order of its forward edges (a value crosses the whole forward
path in one tick; feedback arrives one tick late), and `setUpdateOrder`
lets you build delay lines by running a chain of copy layers oldest-first.

### Wiring groups

Inside a `dense` layer, neurons are organised in **wiring groups**: a set of
neurons plus one input pool they all read. `join` creates a group of all
current neurons; `addFeedback` adds new neurons in a group of their own;
`attachInputs` connects external sensors. When a source layer grows, every
group reading it grows too, and its neurons get matching new weights. See
[dense](dense.md#wiring-groups).

### Learning

Learning is reward-modulated and local: `applyReward(reward, learningRate)`
moves each *eligible* neuron's weights by `learningRate × reward ×
eligibility × sign(input)`. There is no backpropagation and no per-neuron
error signal: every eligible neuron receives the same reward. See
[neuron](neuron.md#learning-updateweights) for eligibility and the exact
rule, and [pattern_benchmark](pattern_benchmark.md#reward-modes) for why an
error-driven reward (reward only when wrong) matters in practice.

### Reproducibility

Initial weights and each neuron's spontaneous-firing generator come from
process-wide random streams. Call `neuron::reseed(seed)` before building a
network to make it independent of anything created earlier in the process.
Results never depend on the number of threads.
