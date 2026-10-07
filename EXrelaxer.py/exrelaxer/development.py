"""Structural development policies: when a population grows, loses neurons, or gets a deeper
layer. See doc/development.md.

The mechanisms are network calls (grow_layer, prune_newest, prune_candidates, freeze_neurons,
connect, subscribe) and the ActivityMonitor, which reads the neurons' own E-R state. The policies
here only decide when to use them, and every number they use is a parameter, so different growth
increments, windows and plateau rules can be compared in experiments:

    monitor = exr.ActivityMonitor(exr.ActivitySpec(window=200))
    width = WidthGrowth(increment=2, max_size=16)
    prune = Pruning(reasons=("invalid", "inactive"), inactive_after=2000)
    for t in range(ticks):
        net.set_inputs(x); net.step(); monitor.observe(net, hidden)
        ...learn...
        width.update(net, hidden, monitor)   # grows when the population is saturated
        prune.update(net, hidden, monitor)   # removes unnecessary neurons, newest first

    plateau = Plateau(baseline=initial_score, patience=5)
    if plateau.update(evaluation_score):     # better than the baseline AND no longer improving
        deeper = grow_depth(net, hidden, readers=[out], spec=exr.LayerSpec.dense(4, False, True))
        plateau.restart()                     # the new layer starts from the score it inherits

Width answers "does the population have enough active capacity?" and depth "has the
representation learned something useful but stopped improving?": the two are separate objects,
each usable without the other.
"""
from dataclasses import dataclass, field

from ._core import LayerSpec, WeightInit


@dataclass
class WidthGrowth:
    """Grows a Dense E-R layer when its ActivityMonitor reports persistent saturation.

    increment        neurons added per growth event
    max_size         never grows past this (None: no limit)
    freeze_existing  the layer's current neurons stop learning; only the new ones learn
    outgoing         weights with which readers read the new neurons (Zero: nothing changes until
                     the readers learn them)

    The monitor starts again after growth (the layer's size changed), so the next growth needs a
    whole new window of saturation. `events` lists (tick, old size, new size) per growth.
    """

    increment: int = 2
    max_size: int | None = None
    freeze_existing: bool = True
    outgoing: WeightInit = WeightInit.Zero
    events: list = field(default_factory=list)

    def update(self, net, layer, monitor, tick=None):
        """Grows if the monitor (watching `layer`) is saturated; returns the neurons added."""
        if not monitor.saturated:
            return 0
        size = net.layer_size(layer)
        count = self.increment
        if self.max_size is not None:
            count = min(count, self.max_size - size)
        if count <= 0:
            return 0
        net.grow_layer(layer, count, self.outgoing, freeze_existing=self.freeze_existing)
        monitor.reset()
        self.events.append((tick, size, size + count))
        return count


@dataclass
class Pruning:
    """Removes neurons that prune_candidates flags, newest grown first, never below the layer's
    minimum size.

    reasons         which candidate reasons count: any of "invalid", "disconnected",
                    "zero_incoming", "unread", "inactive". "unread" is off by default: neurons grown
                    with zero outgoing weights are unread until their readers learn.
    inactive_after  ticks without firing on input (fatigue not counted) before "inactive"
    only_grown      only grown neurons are removed (the base population is kept whole)

    `events` lists (tick, [removed indices], size after) per pruning.
    """

    reasons: tuple = ("invalid", "inactive")
    inactive_after: int = 2000
    only_grown: bool = False
    events: list = field(default_factory=list)

    def candidates(self, net, layer, monitor=None):
        """The indices that would be removed now, newest grown first."""
        order = net.growth_order(layer)
        found = [i for i, why in net.prune_candidates(layer, monitor, self.inactive_after)
                 if any(r in self.reasons for r in why) and (order[i] > 0 or not self.only_grown)]
        found.sort(key=lambda i: (order[i], i), reverse=True)
        room = net.layer_size(layer) - net.minimum_size(layer)
        return found[:max(room, 0)]

    def update(self, net, layer, monitor=None, tick=None):
        """Removes the candidates; returns their indices (as they were before)."""
        chosen = self.candidates(net, layer, monitor)
        if chosen:
            net.prune_neurons(layer, chosen)
            if monitor is not None:
                monitor.reset()
            self.events.append((tick, sorted(chosen), net.layer_size(layer)))
        return sorted(chosen)


@dataclass
class Plateau:
    """Depth-growth trigger: the score has improved on the baseline and then stopped improving.

    Feed it one score per evaluation window (e.g. the mean over many episodes, never one
    episode). It fires when
        best score > baseline + margin                (the network learned something), AND
        the best has not risen by more than min_delta for `patience` evaluations (a plateau).
    A score that never beats the baseline never fires, so a bad architecture does not keep
    growing deeper.

    baseline   the score before learning; None: the first score fed in
    """

    baseline: float | None = None
    margin: float = 0.0
    min_delta: float = 0.0
    patience: int = 5
    best: float | None = None
    stale: int = 0
    history: list = field(default_factory=list)

    def update(self, score):
        """Adds a score; returns True when depth growth is due."""
        self.history.append(score)
        if self.baseline is None:
            self.baseline = score
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.stale = 0
        else:
            self.stale += 1
        return self.improved and self.stale >= self.patience

    @property
    def improved(self):
        return self.best is not None and self.best > self.baseline + self.margin

    def restart(self, baseline=None):
        """After growing: the new structure starts from `baseline` (default: the best so far)."""
        self.baseline = self.best if baseline is None else baseline
        self.best, self.stale = None, 0


def grow_depth(net, after, readers, spec=None, name=None, buses=(), freeze_previous=True,
               freeze_reader_inputs=False):
    """Adds a trainable layer behind `after`: input -> after -> new -> readers.

    The new layer reads `after` (and every bus in `buses`, by subscribing), and each reader in
    `readers` reads the new layer with zero weights, so the network plays exactly as before until
    the readers learn them. The readers keep reading `after` too (the library has no disconnect;
    with freeze_reader_inputs those weights stop learning). freeze_previous freezes the neurons
    `after` has now (neuron by neuron, so it can still grow plastic neurons later); nothing
    unfreezes it. The new layer is tagged LayerSpec.grown. Returns its id.
    """
    if spec is None:
        spec = LayerSpec.dense(4, False, True)
    spec.grown = True
    if name is None:
        base, k = net.layer_name(after) + "_deep", 1
        names = {net.layer_name(i) for i in range(net.layer_count)}
        while f"{base}{k}" in names:
            k += 1
        name = f"{base}{k}"
    new = net.add_layer(name, spec)
    net.connect(after, new)
    for bus in buses:
        net.subscribe(new, bus)
    for reader in readers:
        net.connect(new, reader, WeightInit.Zero)
        if freeze_reader_inputs:
            net.freeze_inputs(reader, after)
    if freeze_previous:
        net.freeze_neurons(after, 0, net.layer_size(after))
    return new
