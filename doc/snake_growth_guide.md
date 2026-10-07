# Snake with a growing network: a guide to the Python API

This guide builds, step by step, a network that plays Snake, learns while it plays, and changes
its own structure as it goes: it starts small, grows wider when its E-R population saturates,
prunes what is not needed, and grows deeper when its score stops improving. On the way it walks
through most of the `exrelaxer` Python API: layers, inputs, buses, stepping, the learning calls,
the critic, state probes, freezing, growth, pruning, and saving.

The aim is to demonstrate the developmental mechanism ([development.md](development.md)), not
to play the best Snake. The finished experiment is
[`NNtesting/experiments/snake_growth`](../NNtesting/experiments/snake_growth/experiment.py); its
results are at the end (raw runs in [results/snake-growth](../results/snake-growth/summary.md)).

Every code block below runs, in order, as one script: `EXrelaxer.py/tests/test_guide.py` runs
them all, so the guide stays correct as the library changes. The game counts are kept small so
that it runs in seconds; the experiment uses real ones.

## 0. Setup

```
./build.sh --python          # builds the library and the Python package, runs the tests
export PYTHONPATH=$PWD/build/EXrelaxer.py/package
```

## 1. The pieces, kept apart

| | In this example |
|---|---|
| **environment state** | the snake's cells, heading, the apple: `game.Game` |
| **network input** | 23 numbers in [−1, 1] computed from that state, in the snake's own frame |
| **network output** | 3 numbers, one per action |
| **environment action** | the action with the largest output: turn left, go straight, turn right |
| **reward** | +1 for an apple, −1 for a crash, ±0.1 for getting closer to / farther from the apple |

The game is the Python copy of the C++ task, `NNtesting/experiments/snake/game.py` (run the
guide's code from the repository root):

```python
import importlib.util
from pathlib import Path

import numpy as np
import exrelaxer as exr

GAME = Path("NNtesting/experiments/snake/game.py")
spec = importlib.util.spec_from_file_location("snake_game", GAME)
snake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snake)

g = snake.Game(10, 10, seed=1)
state = g.state()             # 23 floats
print(len(state), g.score, g.over)
```

## 2. The input

The 23 values (see `game.py` for details) are in the snake's frame, so "left" always means the same
thing whichever way the snake faces:

| values | meaning |
|---|---|
| 0–7 | the 8 cells around the head: +1 wall or body, −1 free (immediate danger) |
| 8–9 | the apple's direction, a unit vector (forward, right) |
| 10–17 | the first wall or body cell seen forward, left, right and back, as map positions |
| 18–21 | how close those 4 cells are: `2 / d − 1` |
| 22 | 1 (a bias) |

No image and no convolution: the developmental mechanism is the subject, not vision.

## 3. Building the network

```
state (23 sensors) -> eye (pass-through) -> er (E-R, 4 neurons) -> left, straight, right
```

```python
exr.reseed(1)                         # every random stream: the same seed builds the same network
net = exr.Network()

shape = exr.Shape(1, 1, snake.STATE_SIZE)
eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(shape), False, False))
net.add_inputs(eye, shape, "state")   # a named input source: set it with set_inputs("state", ...)

er = net.add_layer("er", exr.LayerSpec.dense(
    4, False, True,                   # 4 neurons, no habituation, E-R on
    learning_rule=exr.LearningRule.feedback_alignment()))
net.connect(eye, er)                  # er reads eye

readouts = []
for name in ("left", "straight", "right"):
    out = net.add_layer(name, exr.LayerSpec.dense(1, False, False))   # plain linear readout
    net.connect(er, out)
    net.add_output(out)               # outputs() concatenates output layers in this order
    readouts.append(out)

net.set_minimum_size(er, 4)           # the protected core: pruning never goes below 4
print(net.describe())
```

`LayerSpec.dense(size, habituation, er, ...)` takes, as keywords, `frozen`, `learning_rule`,
`recovery_jitter`, `learning_jitter` and `alpha_jitter`. The spec's fields can be set before
`add_layer` (for example `spec.normalize`, `spec.resting_threshold`, `spec.threshold_growth`,
`spec.habituation_rule`, `spec.spontaneous`, `spec.minimum_size`). `net.layer_spec(id)` reads
them back.

**Starting from one cell.** An organism starts as one cell. The same network can start with a
single E-R neuron that is its own protected core: `LayerSpec.dense(1, ...)` and
`set_minimum_size(er, 1)`. It can't learn Snake at that size, so everything it ends up able to do
comes from growth (section 12).

### The same input through a bus

Buses are shared groups of neurons that layers subscribe to, so a new population can join without
rewiring. A bus is a Dense layer with its own neuron type, frozen by default:

```python
bus_net = exr.Network()
senses = bus_net.add_bus("senses", exr.LayerSpec.dense(snake.STATE_SIZE, False, False))
bus_net.add_inputs(senses, snake.STATE_SIZE, "state")
pop_a = bus_net.add_layer("a", exr.LayerSpec.dense(4, False, True))
bus_net.subscribe(pop_a, senses)                   # a reads the bus
print(bus_net.bus_readers(senses), bus_net.is_bus(senses))
```

Growth in depth can subscribe the new layer to buses (section 10).

## 4. One tick: input, output, action

```python
def choose(net, state):
    """One tick: the action with the largest output (ties go to straight, then left)."""
    net.set_inputs("state", np.asarray(state, np.float32))
    net.step()                         # every layer's forward(), in update order
    y = net.outputs()                  # 3 floats
    best = snake.STRAIGHT
    for a in (snake.LEFT, snake.RIGHT):
        if y[a] > y[best]:
            best = a
    return best, y

action, y = choose(net, g.state())
print(action, y, net.layer_output(er))
probe = net.state_probe(er)            # read-only: output, threshold, last_sum, ...
print(probe["threshold"], probe["last_sum"])
```

`net.update_order` shows the order layers run in; by default it follows the connections, so a
value travels from the sensors to the readouts in one step.

## 5. Learning while playing

There is no separate training phase: the network learns after every move.

```python
def reward_for(result, before, after):
    if result == snake.ATE:
        return 1.0
    if result == snake.DIED:
        return -1.0
    return 0.1 if after < before else -0.1

def learn(net, action, y, reward, lr=0.03):
    """Error-driven: only the chosen action's readout gets an error, and only when its sign
    disagrees with the reward. The readouts learn from their own error; the E-R layer learns from
    the same errors through a fixed random feedback matrix (feedback alignment)."""
    if y[action] * reward > 0:
        return
    errors = np.zeros(3, np.float32)
    errors[action] = reward
    net.apply_error(errors, lr)
```

Other ways to learn, all on the same network:

- `net.apply_reward(r, lr)`: every unfrozen layer, with its own rule (Sign, Trace, Eligibility...);
- `net.apply_reward_to(layer, r, lr)`: one layer only (the original snake experiment's readouts);
- `net.apply_modulators_to(layer, m, lr)`: one modulator per neuron;
- an actor-critic: `net.set_critic(layers=[er])`, then `net.apply_reward_td(r, lr, terminal)` after
  each step (the layers learn from the TD error) and `net.reset_critic()` between games;
- curiosity: `net.set_curiosity(predict_layers=[er], from_layers=[er])` and add
  `net.curiosity_reward()` to the reward.

`net.set_learning_rule(layer, rule)` changes a layer's rule at any time (see [learning](learning.md)).
In this task the reward-driven rules made the hidden layer play worse, and feedback alignment
learned, so the experiment uses it.

## 6. Playing games, and the baseline

```python
def play(net, games, seed, learning, explore=0.05, on_tick=None):
    """Plays `games` games; returns per game (apples, steps, died)."""
    policy = snake.Rng(seed ^ 0x5EED)
    results = []
    for i in range(games):
        g = snake.Game(10, 10, (seed * 1000003 + i) & snake.MASK)
        last = snake.MOVED
        while not g.over:
            action, y = choose(net, g.state())
            if learning and policy.uniform() < explore:
                action = policy.below(3)       # a little exploration while learning
            before = g.apple_distance()
            last = g.step(action)
            if learning:
                learn(net, action, y, reward_for(last, before, g.apple_distance()))
            if on_tick:
                on_tick()
        results.append((g.score, g.steps, last == snake.DIED))
    return np.array(results, dtype=float)

baseline = play(net, 10, seed=50, learning=False)
print("baseline apples/game", baseline[:, 0].mean())
```

A developmental system cannot be judged from one game. The experiment records, per run:
the baseline (the untrained network over a window of games), the score of every evaluation window
(mean apples per game over e.g. 50 games), the final score (the last 3 windows), the best window
and best game, the steps played, the network size, and how many width-growth, pruning and
depth-growth events happened.

## 7. Watching saturation

An `ActivityMonitor` watches one layer and reads, every tick, what each neuron did from its own
E-R state: fired, fired spontaneously, silent because fatigued, habituated, weak input, or no
input. Call `observe` after every step:

```python
monitor = exr.ActivityMonitor(exr.ActivitySpec(window=200, saturated_share=0.5))
play(net, 3, seed=7, learning=True, on_tick=lambda: monitor.observe(net, er))
print(monitor.ticks, monitor.saturation, monitor.saturated)
print(monitor.states())                # TickState per neuron, last tick
print(monitor.inactive_ticks())        # ticks without firing on input (fatigue not counted)
```

`saturation` is the share of the window's ticks on which no neuron fired, input was present, and
at least `blocked_share` of the neurons were fatigued. `saturated` is True once the window is full
and that share reaches `saturated_share`. A single silent tick, or fatigue that recovers within the
window, is not saturation.

## 8. Width growth

By hand:

```python
before = net.layer_size(er)
net.grow_layer(er, 2, exr.WeightInit.Zero, freeze_existing=True)
print(net.layer_size(er), net.grown_neurons(er), net.frozen_neurons(er), net.growth_order(er))
```

The new neurons read what the old ones read, start with a fresh threshold and learn, while the old
ones are frozen. The readouts read the new neurons with zero weights, so nothing changes until they
learn. With a policy, growth happens when the monitor reports saturation:

```python
from exrelaxer.development import WidthGrowth, Pruning, Plateau, grow_depth

width = WidthGrowth(increment=2, max_size=16)   # freeze_existing=True, outgoing=Zero by default
monitor = exr.ActivityMonitor(exr.ActivitySpec(window=200, saturated_share=0.5))

def tick():
    monitor.observe(net, er)
    width.update(net, er, monitor)              # grows only when saturated

play(net, 3, seed=8, learning=True, on_tick=tick)
print(width.events)                             # (tick, old size, new size) per growth
```

## 9. Pruning

Nothing is removed by asking:

```python
print(net.prune_candidates(er, monitor, inactive_after=1000))   # [(index, [reasons])]
```

The reasons are `invalid` (NaN or infinite state), `disconnected`, `zero_incoming`, `unread` (no
downstream weight; true of freshly grown neurons until the readouts learn them) and `inactive` (no
firing on input for `inactive_after` ticks, fatigue not counted). `Pruning` removes the chosen ones,
newest grown first, never below the minimum:

```python
pruning = Pruning(reasons=("invalid", "inactive"), inactive_after=5000)
pruning.update(net, er, monitor)
```

LIFO pruning removes the most recently grown neurons, and never base neurons:

```python
net.grow_layer(er, 1)
newest = net.layer_size(er) - 1
print(net.prune_newest(er, 1) == [newest])
print(net.layer_size(er), net.minimum_size(er))
```

## 10. Depth growth

Depth growth waits for a plateau: one score per evaluation window, never one game.

```python
plateau = Plateau(baseline=0.4, margin=0.0, patience=3)
print([plateau.update(s) for s in [0.42, 0.48, 0.53, 0.57, 0.59, 0.60, 0.60, 0.60, 0.60]])
```

It fires only when the best score beats the baseline and has not improved for `patience` windows.
Then:

```python
er2 = grow_depth(net, er, readers=readouts,
                 spec=exr.LayerSpec.dense(4, False, True,
                                          learning_rule=exr.LearningRule.feedback_alignment()),
                 name="er2")
plateau.restart()                     # the score reached so far is the new baseline
print(net.layer_spec(er2).grown, net.frozen_neurons(er), net.update_order)
```

`er` is frozen neuron by neuron, `er2` reads it and learns, and the readouts read `er2` with zero
weights (they keep reading `er` too). With `buses=[senses]` the new layer also subscribes to
existing buses. From here on, width growth watches the newest layer.

## 11. Saving and loading

```python
data = net.to_bytes()                  # or net.save("grown.exr")
copy = exr.Network.from_bytes(data)    # or exr.Network.load("grown.exr")
print(copy.layer_size(er), copy.minimum_size(er), copy.growth_order(er) == net.growth_order(er))
copy.set_inputs("state", np.asarray(g.state(), np.float32)); copy.step()
net.set_inputs("state", np.asarray(g.state(), np.float32)); net.step()
print(np.array_equal(copy.outputs(), net.outputs()))   # the same state, the same behaviour
```

A saved network keeps its layers, wiring, buses, weights, E-R and habituation state, frozen neurons
and inputs, learning rules and their traces, the minimum sizes, the grown tags and its whole growth
and pruning history (which rebuilds the growth order). Activity monitors and policies are plain
Python objects and are not saved.

## 12. The whole experiment

[`NNtesting/experiments/snake_growth/experiment.py`](../NNtesting/experiments/snake_growth/experiment.py)
puts the pieces together: an `Agent` with the network and policies, `play` with learning always
on, evaluation windows of 50 games, width growth on the newest layer, pruning, undoing a width
growth that did not raise the score within `undo` windows (LIFO), and depth growth on a plateau.

```
python NNtesting/experiments/snake_growth/experiment.py --start 1 --games 1500 --save grown.exr
python NNtesting/experiments/snake_growth/experiment.py --start 4 --grow false     # the control
NNtesting/nntest.py run snake_growth --set start=1,4 --set grow=true,false          # 3 seeds each
```

The script prints the development as it happens:

```
baseline 0.06 apples/game, network [1]
  tick 277: saturated, width 1 -> 3
evaluation 0: 0.82 apples/game, network [3]
evaluation 1: 1.96 apples/game, network [3]
  tick 4798: saturated, width 3 -> 5
...
evaluation 12: 3.26 apples/game, network [5]
  plateau at 3.96 (baseline 0.06): depth [5] -> [5, 4]
...
```

Every parameter is a flag: `--start`, `--minimum`, `--recovery`, `--increment`, `--max-width`,
`--window`, `--saturated-share`, `--freeze-old`, `--eval-games`, `--games`, `--patience`,
`--margin`, `--max-depth`, `--deep-size`, `--undo`, `--lr`, `--explore`, the rewards, and `--grow`.

### Results

3 seeds each, 1,500 games with learning always on, a 10 × 10 board. "Final" is the mean apples per
game over the last 150 games.

| setup | final | network at the end |
|---|---|---|
| ER(1), no growth | 0.2 (0.0–0.4) | [1] |
| ER(1), growth (old neurons frozen) | 4.5 (3.6–5.4) | [5] + [4]: 2 width growths, 1 depth growth every seed |
| ER(4), no growth | 5.1 (4.9–5.5) | [4] |
| ER(4), growth | 5.8 (5.0–6.6) | [6], [6] + [4] or [4] + [6] |
| ER(8), fixed from the start | 7.0 (6.8–7.2) | [8] |
| ER(1), growth, recovery 0.99 | 2.5 (1.4–3.1) | grew to [11–16] + [10–16] |
| ER(1), growth, old neurons stay plastic | 4.0 (2.5–5.5) | [3–7] + [4] |
| ER(4), growth, old neurons stay plastic | 5.8 (5.4–6.3) | [4–6] + [4]; one growth undone (LIFO) |

What it shows:

- growth rescues a population that cannot learn the task (one neuron: 0.2 → 4.5 apples per game),
  and width growth stops by itself (saturation falls as the population grows);
- a grown network ends below a network built at the right size from the start (7.0), and
  keeping the old neurons plastic does not close the gap (4.0 and 5.8 against 4.5 and 5.8 with
  freezing), so freezing is not what costs the score. A likelier cause (untested) is time: a
  grown network spends much of the run small, and its late neurons have less of it to learn;
- slowly recovering neurons (0.99) saturate more, grow much larger and play worse: saturation
  measures silence, not usefulness;
- runs are deterministic for a seed (the single-seed run and the matrix gave identical numbers).

## 13. Inspecting a grown network

```python
print(net.describe().splitlines()[-4:])   # ends with one "growth" line per developing layer
for layer in (er, er2):
    print(net.layer_name(layer), net.layer_size(layer), net.minimum_size(layer),
          net.grown_neurons(layer), net.frozen_neurons(layer), net.layer_spec(layer).grown)
```

`neuron_state(layer)` gives thresholds, recovery and gains; `state_probe(layer)` the last tick;
`weights(layer, i)` a neuron's weights; `edges` and `update_order` the wiring.

Signs of successful development: the grown network beats its starting architecture on the same
games, width growth stops on its own rather than at `max_width`, each depth growth follows a real
improvement and is followed by another, few growths need undoing, and frozen neurons keep their
weights.

## API at a glance

| Task | Call |
|---|---|
| build | `Network()`, `add_layer(name, LayerSpec)`, `connect(a, b, init)`, `add_inputs(layer, n_or_shape, name)`, `connect_inputs(name, layer)`, `add_output(layer)`, `add_feedback(a, b, width)` |
| buses | `add_bus(name, spec)`, `subscribe(reader, bus)`, `write_bus(writer, bus)`, `bus_readers`, `bus_writers` |
| run | `set_inputs(name, values)`, `step()`, `outputs()`, `run(inputs)`, `layer_output(layer)` |
| learn | `apply_error`, `apply_reward`, `apply_reward_to`, `apply_modulators_to`, `set_learning_rule`, `reset_traces` |
| critic, curiosity | `set_critic`, `apply_reward_td`, `temporal_difference`, `reset_critic`, `set_curiosity`, `curiosity_reward` |
| freeze | `freeze(layer)`, `freeze_neurons(layer, first, count)`, `freeze_inputs(layer, source)` |
| grow | `grow_layer`, `grown_neurons`, `growth_order`, `development.WidthGrowth`, `development.grow_depth` |
| prune | `set_minimum_size`, `prune_candidates`, `prune_neurons`, `prune_newest`, `development.Pruning` |
| watch | `ActivityMonitor(ActivitySpec(...))`, `observe`, `saturated`, `states()`, `inactive_ticks()` |
| decide depth | `development.Plateau(baseline, margin, patience)` |
| inspect | `describe()`, `state_probe`, `neuron_state`, `weights`, `layer_spec`, `edges`, `update_order` |
| save | `save(path)`, `Network.load(path)`, `to_bytes()`, `Network.from_bytes(data)` |
| reproduce | `reseed(seed)` before building |
