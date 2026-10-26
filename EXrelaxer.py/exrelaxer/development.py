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

import numpy as np

from ._core import LayerSpec, WeightInit


@dataclass
class WidthGrowth:
    """Grows a Dense E-R layer when its ActivityMonitor reports persistent saturation.

    increment        neurons added per growth event
    max_size         never grows past this (None: no limit)
    freeze_existing  the layer's current neurons stop learning; only the new ones learn
    outgoing         weights with which readers read the new neurons (Zero: nothing changes until
                     the readers learn them); with init="copy" the split below decides instead
    init             "random": new incoming weights drawn as for any new wiring; "copy": each new
                     neuron copies an existing neuron (Net2Net-style widening, Network.copy_neuron)
    source           init="copy": which neurons are copied, the top `increment` by
                     "fatigued" (most ticks held back by its own raised threshold since the
                     monitor's reset: the overloaded ones), "active" (most firings on input),
                     "read" (largest read strength: the most used) or "random"
    noise            init="copy": each copied weight gets noise x the source's weight RMS x U[-1, 1]
    split            init="copy": share of the source's outgoing weights handed to the copy (0.5:
                     the pair together reads like the source alone did; 0: the copy starts unread)

    The monitor starts again after growth (the layer's size changed), so the next growth needs a
    whole new window of saturation. `events` lists (tick, old size, new size) per growth, and
    `copies` (tick, [(source, new neuron)]) per copy-init growth.
    """

    increment: int = 2
    max_size: int | None = None
    freeze_existing: bool = True
    outgoing: WeightInit = WeightInit.Zero
    init: str = "random"
    source: str = "fatigued"
    noise: float = 0.05
    split: float = 0.5
    seed: int = 0
    events: list = field(default_factory=list)
    copies: list = field(default_factory=list)

    def __post_init__(self):
        if self.init not in ("random", "copy"):
            raise ValueError("init must be 'random' or 'copy'")
        if self.source not in ("fatigued", "active", "read", "random"):
            raise ValueError("source must be 'fatigued', 'active', 'read' or 'random'")

    def sources(self, net, layer, monitor, count):
        """The `count` neurons copy-init copies, by `source` (ties: lower index first); a layer
        smaller than count is copied round-robin."""
        size = net.layer_size(layer)
        if self.source == "random":
            rng = np.random.default_rng([self.seed, len(self.events)])
            order = list(rng.permutation(size))
        else:
            if self.source == "fatigued":
                score = monitor.fatigued_ticks()
            elif self.source == "active":
                score = monitor.firings()
            else:
                score = net.read_strength(layer)
            score = np.asarray(score, dtype=np.float64)
            if score.size != size:   # the monitor has not watched this size yet
                score = np.zeros(size)
            score = np.where(np.isfinite(score), score, np.finfo(np.float64).max)
            order = sorted(range(size), key=lambda i: (-score[i], i))
        return [int(order[k % size]) for k in range(count)]

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
        copied = self.sources(net, layer, monitor, count) if self.init == "copy" else []
        outgoing = WeightInit.Zero if copied else self.outgoing
        net.grow_layer(layer, count, outgoing, freeze_existing=self.freeze_existing)
        if copied:
            pairs = [(src, size + k) for k, src in enumerate(copied)]
            for src, new in pairs:
                net.copy_neuron(layer, src, new, self.noise, self.split)
            self.copies.append((tick, pairs))
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
    weak_fraction   the "weak" reason (off unless listed in reasons): a neuron its readers barely
                    read, whose read strength (Network.read_strength: largest reader |weight|) is
                    below weak_fraction x the layer's mean read strength
    weak_after      ticks a layer is watched (update calls with a tick) before "weak" applies to it,
                    so readers that start at zero (grow_depth) can learn first

    `events` lists (tick, [removed indices], size after) per pruning.
    """

    reasons: tuple = ("invalid", "inactive")
    inactive_after: int = 2000
    only_grown: bool = False
    weak_fraction: float = 0.2
    weak_after: int = 20000
    events: list = field(default_factory=list)
    first_seen: dict = field(default_factory=dict)

    def weak(self, net, layer, tick=None):
        """Indices whose read strength is below weak_fraction x the layer mean (none before the layer
        has been watched for weak_after ticks, or when the layer is read as a whole)."""
        if tick is not None:
            start = self.first_seen.setdefault(layer, tick)
            if tick - start < self.weak_after:
                return []
        strength = np.asarray(net.read_strength(layer), dtype=np.float64)
        if strength.size == 0 or not np.all(np.isfinite(strength)):
            return []
        limit = self.weak_fraction * strength.mean()
        return [int(i) for i in np.flatnonzero(strength < limit)]

    def candidates(self, net, layer, monitor=None, tick=None):
        """The indices that would be removed now, newest grown first."""
        order = net.growth_order(layer)
        found = {i for i, why in net.prune_candidates(layer, monitor, self.inactive_after)
                 if any(r in self.reasons for r in why)}
        if "weak" in self.reasons:
            found.update(self.weak(net, layer, tick))
        found = [i for i in found if order[i] > 0 or not self.only_grown]
        found.sort(key=lambda i: (order[i], i), reverse=True)
        room = net.layer_size(layer) - net.minimum_size(layer)
        return found[:max(room, 0)]

    def update(self, net, layer, monitor=None, tick=None):
        """Removes the candidates; returns their indices (as they were before)."""
        chosen = self.candidates(net, layer, monitor, tick)
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
