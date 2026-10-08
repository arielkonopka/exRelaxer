# Structural development: growing and pruning a learning network

**Experimental.** This page describes mechanisms that let a network change its own size while it
keeps learning: E-R populations grow wider when they saturate, lose neurons that are demonstrably
unnecessary, and a deeper layer is added when the score has improved and then stopped improving.
It is loosely inspired by how nervous systems develop, starting from very little (one cell, if you
want). It does **not** claim to reproduce biological neurodevelopment: there is no cell division,
migration, guidance or apoptosis here, only counted rules on top of the existing E-R state.

The design keeps the library's existing primitives and adds as little as possible:

| Need | Mechanism | Where |
|---|---|---|
| grow a population | `growLayer` (existing; fresh neurons, zero outgoing weights by default) | `network` |
| freeze old structure | `freezeNeurons`, `freeze`, `freezeInputs` (existing) | `network` |
| a protected core | `setMinimumSize` / `LayerSpec::minimumSize` | `network` |
| tell base from grown neurons | `neuron_layer::growthOrder`, `network::grownNeurons` | `neuron_layer` |
| pruning candidates | `network::pruneCandidates` (nothing is removed) | `network` |
| LIFO pruning | `network::pruneNewest` | `network` |
| saturation | `activity_monitor` reading each neuron's own E-R state | `development.hpp` |
| when to grow or prune | `WidthGrowth`, `Pruning`, `Plateau`, `grow_depth` | Python `exrelaxer.development` |
| communication | buses (`addBus`, `subscribe`; existing) | `network` |

There is no "organism" object. The policies are small Python classes whose every number is a
parameter, so the architecture can come out of experiments rather than be designed in advance.

## 1. Width growth

Width growth adds neurons to an existing Dense E-R population when the population no longer has
enough *active* capacity: its neurons keep receiving input but are kept silent by thresholds their
own earlier firing raised.

```
input -> E-R population (4)          saturated              input -> E-R population (6)
         [1][2][3][4]                ─────────>                      [1][2][3][4] [5][6]
                                                                      frozen       new, plastic
```

`net.grow_layer(layer, count, outgoing=WeightInit.Zero, freeze_existing=True)`:

- the new neurons read what the layer's first wiring group reads (sensors, other layers, buses,
  the layer itself if it is recurrent), with fresh random weights;
- they start in a fresh state: resting threshold, no habituation streak, no traces. They do not
  inherit the population's fatigue;
- readers get the new outputs with zero weights, so the network plays exactly as before until the
  readers learn them;
- `freeze_existing` freezes the current neurons (they still run but do not learn), so only the new
  neurons learn;
- the growth is recorded in the network's history and saved.

`WidthGrowth(increment=2, max_size=16)` decides when: on persistent saturation (section 3). The
increment is a parameter of the policy, not a constant in the library.

## 2. Depth growth

Depth growth answers a different question: has the current representation learned something useful
but reached its limit? It is not a reaction to a bad score.

```
baseline 0.40 -> 0.42 0.48 0.53 0.57 0.59 0.60 0.60 0.60 0.60
                                                    plateau  -> add a layer
```

`Plateau(baseline, margin, min_delta, patience)` takes one score per evaluation window (the mean
over many episodes, never a single episode) and fires only when

    best score > baseline + margin      the network has demonstrably learned, AND
    no new best for `patience` windows  it has stopped improving.

A network that never beats its baseline never grows deeper, so a bad architecture cannot grow
forever. After growing, `plateau.restart()` makes the score reached so far the new baseline.

`grow_depth(net, after, readers, spec, buses=())`:

```
input -> L1 -> output            input -> L1 ------------> output
                         ──>               frozen \         ^
                                                   -> L2 ---/  (zero weights at first)
                                                     trainable
```

- the new layer reads `after` (and subscribes to `buses`), and each reader reads the new layer
  with zero weights: behaviour is unchanged at the moment of growth;
- the neurons `after` has are frozen, neuron by neuron. The layer itself is not frozen, so it
  can still grow new plastic neurons later. Nothing unfreezes it because a layer was added;
- the readers keep their connection to `after`. The library has no "disconnect", and keeping it
  means the network does not lose what it can already do. With `freeze_reader_inputs=True` those
  weights stop learning too;
- the new layer is tagged `LayerSpec.grown` (saved).

## 3. E-R saturation

An E-R neuron fires when its input exceeds its threshold, and firing raises the threshold. A
population is saturated when the thresholds have risen above the input the population receives:

```
neuron 1: input < threshold -> 0
neuron 2: input < threshold -> 0
...
neuron N: input < threshold -> 0          population output = 0
```

This is emergent E-R behaviour, and the detector reads only the neurons' own state. Every neuron
records, for its last tick, the clamped raw sum, the threshold the tick started with, and whether it
fired on its input, fired spontaneously, or stayed silent (`neuron::lastSum`, `lastThreshold`,
`lastFiring`; read-only probes, also in `state_probe` as `last_sum`, `last_threshold`,
`spontaneous`). `ActivityMonitor` turns that into one state per neuron per tick:

| State | Meaning |
|---|---|
| `Fired` | fired on its input (\|sum\| > `input_epsilon`) |
| `Spontaneous` | fired on its own (E-R spontaneous firing); counts as silent |
| `Fatigued` | silent, input present, below a threshold raised above `fatigue_ratio` × resting |
| `Habituated` | silent, the raw input would have fired it but habituation suppressed it |
| `Weak` | silent, input present, threshold not raised: the input is simply too weak |
| `NoInput` | \|sum\| ≤ `input_epsilon` (also a firing on a vanishing input after E-R relaxed) |

A **saturated tick** is one where no neuron fired on its input and at least `blocked_share` of the
neurons were `Fatigued`. The population is **saturated** when, over a sliding window of `window`
ticks, at least `saturated_share` of the ticks were saturated. So:

- a single silent tick is never saturation;
- fatigue that recovers inside the window is not saturation;
- silence without input is not saturation (nothing relevant is coming in);
- silence because the input is weak relative to a resting threshold is not saturation either: that
  is a property of the weights, not of fatigue.

Thresholds still recover and spontaneous firing still happens: the monitor only watches. It starts
again whenever the layer's size changes, so new neurons are judged from their first tick.

**What it looks like on real E-R state** (Snake, the experiment's untrained network, 30 games
with 30% random moves; `sweep.py --saturation`): in a first probe each neuron was fatigued about
two thirds of the ticks and fired about a third of the time, whatever the population size. The share of ticks on
which the whole population was saturated falls with its size:

| recovery | ER(1) | ER(2) | ER(4) | ER(8) | ER(16) |
|---|---|---|---|---|---|
| 0.9 (default) | 75% | 38% | 19% | 0% | 0% |
| 0.99 | 90% | 76% | 65% | 39% | 11% |

So saturation is a real signal, and width growth limits itself: growing makes saturation rarer.
The trigger (`saturated_share`) decides where growth stops; with the default 0.9 nothing grows at
recovery 0.9.

## 4. Minimum size: the protected core

`net.set_minimum_size(layer, n)` (or `LayerSpec.minimum_size`) is a count: pruning never leaves
the layer with fewer than `n` neurons. `prune_neurons` refuses (and changes nothing) when it would
go below; `prune_newest` and the `Pruning` policy stop at it. The floor can be 1: a population can
start from a single neuron, grow, and be pruned back to that one neuron. 0 (the default) means no
floor, so existing networks behave as before. The minimum is saved, as the value it has now.

## 5. Base and grown neurons, and LIFO pruning

Every neuron of a layer has a growth order: 0 for neurons built with the layer (at construction or
by feedback wiring), and 1, 2, 3... for neurons `grow_layer` added, in the order they were added,
counting on across pruning. `net.growth_order(layer)` and `net.grown_neurons(layer)` read it. It is
not stored in the file: replaying the saved history (grow and prune operations in order) rebuilds
it exactly.

`net.prune_newest(layer, count)` removes up to `count` grown neurons, the most recently grown first,
never base neurons and never below the minimum:

```
[1][2][3][4][5][6][7][8]  ->  [1][2][3][4][5][6][7]  ->  ...  ->  [1][2][3][4]   (stops)
```

It is the fallback that preserves older learned structure. In the Snake experiment it undoes a
width growth that did not raise the score after a few evaluation windows; the neurons that growth
froze learn again.

## 6. Invalid and useless neurons

`net.prune_candidates(layer, monitor=None, inactive_after=1000)` lists `(index, [reasons])` and
removes nothing:

| Reason | Meaning |
|---|---|
| `invalid` | a weight, the bias, the threshold or the output is NaN or infinite |
| `disconnected` | the neuron reads nothing |
| `zero_incoming` | every weight it reads with is 0 |
| `unread` | nothing downstream reads it with a non-zero weight (not an output layer, not a critic or curiosity feature) |
| `inactive` | the monitor saw it go `inactive_after` ticks without firing on its input; fatigued and habituated ticks are not counted |

`unread` is true of every freshly grown neuron (zero outgoing weights) until its readers learn, so
the `Pruning` policy ignores it by default. `inactive` never counts fatigue: a neuron is not pruned
because its threshold is high. An E-R neuron that receives any input fires again once its threshold
has relaxed below it, so in practice `inactive` means "receives no input". The `Pruning` policy
removes the chosen candidates newest grown first, within the minimum size (`only_grown=True` keeps
the base population whole).

`Network.read_strength(layer)` gives, per neuron, the largest |weight| any Dense reader gives it (0:
unread; infinity: the layer is read as a whole). The `Pruning` policy's optional `weak` reason uses
it: a neuron whose read strength is below `weak_fraction` x the layer's mean is one its readers
barely use. A layer is judged only after `weak_after` ticks of `update` calls with a tick, so readers
that start at zero (depth growth) can learn first. On Snake, wide depth growth plus `weak` pruning
shrinks each new layer from 16 towards its minimum and plays as well as twice as many neurons
(research log §31).

## 7. Buses

Growth does not rewire: a new population subscribes to the buses that already exist.

```
sensor -> vision bus -+-> population A
                      +-> population B
                      +-> newly created population C      net.subscribe(C, vision)
```

`grow_depth(..., buses=[vision])` subscribes the new layer. Layers grown in width keep reading
whatever their wiring group reads, buses included. Buses stay frozen unless unfrozen. Bus structure
(writers, readers) is saved with the network.

## 8–10. The Snake example, running it, inspecting growth

The practical guide, which is also a tour of the Python API, is
[snake_growth_guide.md](snake_growth_guide.md). In short:

```
python NNtesting/experiments/snake_growth/experiment.py --start 1 --games 1500 --save grown.exr
NNtesting/nntest.py run snake_growth --set start=1,4 --set grow=true,false
```

Inspect a network with `net.describe()`, which ends with one `growth` line per layer that has a
minimum, grown neurons or the grown tag:

```
  growth er: 5 neurons, 4 grown, minimum 1
  growth er2: 4 neurons, 0 grown, minimum 4, a grown layer
```

and with `growth_order`, `grown_neurons`, `frozen_neurons`, `minimum_size`, `layer_spec(id).grown`,
`neuron_state` and the `ActivityMonitor` (`saturation`, `states()`, `inactive_ticks()`).

## 11. What successful development looks like

Compare against the same run with growth off, on the same games:

- the score after growth beats the starting architecture's, and ideally matches a fixed network
  of the final size (growth found the size, rather than costing performance);
- width growth stops on its own (saturation falls as the population grows), not at `max_width`;
- each depth growth comes after a real improvement and is followed by a further one;
- few undo events: growth that had to be undone was growth the task did not need;
- old structure keeps working: the frozen neurons' weights are unchanged.

Results for the Snake example are in the guide.

## 12. Limitations and failure modes

- **Thresholds of the trigger.** Whether a population grows depends on `saturated_share`, `window`
  and the neurons' recovery. With the default recovery 0.9 and the default share 0.9, a Snake
  population never grows. These are experimental knobs, not derived quantities.
- **Saturation measures silence, not usefulness.** A population can be busy and still too small for
  the task. Width growth only answers "is it silent while input arrives?".
- **Frozen is not safe from drift.** Frozen neurons keep their weights, but readers of them still
  learn, and the readers' weights on old neurons can drift (`freeze_inputs` stops that, at the cost
  of plasticity).
- **Depth keeps the old shortcut.** Readers still read the frozen layer directly; there is no
  disconnect.
- **One growing layer at a time in the example.** Only the newest adaptive layer grows in width;
  older ones are frozen.
- **Noisy scores.** A plateau rule on noisy evaluation windows can fire on noise. Use long windows
  (many games), a margin, and patience.
- **Monitors are not saved.** An `ActivityMonitor` starts empty after loading a network.
- **Only Dense layers** have a minimum size, grown neurons and pruning (as before, pruning needs Dense
  readers).
- **Determinism.** Everything is deterministic for a seed: growth draws its weights from the
  library's weight stream, so the same run grows the same way, and a saved network replays its
  growth and pruning exactly.
- **Learning rule.** In the Snake example the hidden layers learn with feedback alignment; the
  reward-driven rules (sign, eligibility) made the hidden layer play worse there.

## API summary

C++ (`network.hpp`, `development.hpp`, `neuron_layer.hpp`):

```cpp
net.setMinimumSize(id, n);  net.minimumSize(id);
net.growLayer(id, count, WeightInit::Zero, /*freezeExisting=*/true);
net.grownNeurons(id);       layer.growthOrder(i);  layer.neuronGrown(i);
net.pruneNewest(id, count);
net.pruneCandidates(id, &monitor, inactiveAfter);  // PruneCandidate{index, reasons}
activity_monitor m({.window = 200, .saturatedShare = 0.5f});
m.observe(net.layerAs<neuron_layer>(id));  m.saturated();  m.inactiveTicks(i);
```

Python (`exrelaxer`, `exrelaxer.development`):

```python
net.set_minimum_size(h, 1); net.grow_layer(h, 2, freeze_existing=True)
net.growth_order(h); net.grown_neurons(h); net.prune_newest(h, 2); net.prune_candidates(h, monitor)
monitor = exr.ActivityMonitor(exr.ActivitySpec(window=200, saturated_share=0.5))
WidthGrowth(increment=2, max_size=16).update(net, h, monitor)
Pruning(reasons=("invalid", "inactive")).update(net, h, monitor)
net.read_strength(h); Pruning(reasons=("weak",), weak_fraction=0.2, weak_after=20000).update(net, h, tick=t)
Plateau(baseline=b, patience=4).update(score); grow_depth(net, h, readers=[out])
```

Network files are format 19 (minimum size and the grown tag per layer); older files load as before.
