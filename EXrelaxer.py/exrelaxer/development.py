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
    resting_scale    the new neurons' E-R resting threshold is the layer's times this (1: the same;
                     below 1 they fire more easily for good, Network.set_resting_threshold; not
                     saved with the network)
    start_threshold  the new neurons' threshold right after growth (None: their resting threshold;
                     e.g. 0: the minimum, they fire on any input until it relaxes back)

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
    resting_scale: float = 1.0
    start_threshold: float | None = None
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
        return self.grow(net, layer, monitor, tick)

    def grow(self, net, layer, monitor, tick=None):
        """Grows now, whatever the monitor says (another trigger decided); returns the neurons added."""
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
        if self.resting_scale != 1.0 or self.start_threshold is not None:
            resting = net.state_probe(layer)["resting_threshold"]
            for i in range(size, size + count):
                if self.resting_scale != 1.0:
                    net.set_resting_threshold(layer, i, float(resting[i]) * self.resting_scale)
                if self.start_threshold is not None:
                    net.set_threshold(layer, i, self.start_threshold)
        monitor.reset()
        self.events.append((tick, size, size + count))
        return count


@dataclass
class Maturation:
    """A critical period for grown neurons: newborn neurons are hyperexcitable and more plastic,
    then mature into neurons like the rest (as adult-born hippocampal granule cells do over a few
    weeks; doc/development.md, research log §39).

    ticks           the critical period: ticks from growth to maturity
    resting_scale   a newborn's E-R resting threshold, x the layer's (below 1: fires more easily)
    gain_scale      a newborn's learning gain, x its gain at birth (above 1: learns faster)
    mature_resting  the resting threshold at maturity, x the layer's
    mature_gain     the learning gain at maturity, x its gain at birth (0: it stops learning)
    freeze          at maturity the neuron is frozen (the critical period closes for good)
    every           ticks between updates (the values move in steps of every / ticks)

    Both values move linearly with age, from the newborn's to the mature value. A neuron's birth
    is the first update that sees it (call update every tick, or right after growth); neurons are
    told apart by their growth order, so pruning and undoing growth do not confuse ages.
    `matured` counts neurons that reached maturity. The per-neuron resting threshold is not saved
    with the network (see Network.set_resting_threshold).
    """

    ticks: int = 20000
    resting_scale: float = 0.5
    gain_scale: float = 1.0
    mature_resting: float = 1.0
    mature_gain: float = 1.0
    freeze: bool = False
    every: int = 20
    births: dict = field(default_factory=dict)
    matured: int = 0

    def __post_init__(self):
        if self.ticks <= 0 or self.every <= 0:
            raise ValueError("ticks and every must be positive")
        if min(self.resting_scale, self.mature_resting) <= 0 or min(self.gain_scale, self.mature_gain) < 0:
            raise ValueError("resting scales must be > 0 and gain scales >= 0")

    def update(self, net, layer, tick):
        """Registers newborn neurons and, every `every` ticks (and at each birth), moves every
        maturing neuron's values; returns how many neurons are still maturing."""
        order = [int(o) for o in net.growth_order(layer)]
        born = [i for i, o in enumerate(order) if o and (layer, o) not in self.births]
        if not born and tick % self.every:
            return sum(1 for o in order if o and self.births[(layer, o)][0] is not None)
        if born:
            resting = net.state_probe(layer)["resting_threshold"]
            gains = net.neuron_state(layer)["learning_gain"]
            base = [resting[i] for i, o in enumerate(order) if o == 0]
            for i in born:   # [birth tick, the layer's resting threshold, the gain at birth]
                self.births[(layer, order[i])] = [tick, float(base[0] if base else resting[i]), float(gains[i])]
        maturing = 0
        for i, o in enumerate(order):
            entry = self.births.get((layer, o)) if o else None
            if entry is None or entry[0] is None:
                continue
            birth, resting, gain = entry
            f = min(1.0, (tick - birth) / self.ticks)
            net.set_resting_threshold(layer, i, resting * (self.resting_scale + (self.mature_resting - self.resting_scale) * f))
            net.set_learning_gain(layer, i, gain * (self.gain_scale + (self.mature_gain - self.gain_scale) * f))
            if f < 1.0:
                maturing += 1
                continue
            entry[0] = None   # mature: left alone from now on
            self.matured += 1
            if self.freeze:
                net.freeze_neurons(layer, i, 1)
        return maturing


@dataclass
class PatternDiversity:
    """How diverse a layer's firing is: the distribution of its binary firing patterns (which
    neurons have a non-zero output) over the ticks observed since the last reset.

    summary() gives
        entropy     Shannon entropy of the pattern distribution, bits (at most log2(ticks) and the
                    layer size)
        distinct    how many different patterns occurred
        neuron_entropy  mean over neurons of each neuron's own firing entropy (bits, at most 1):
                    how evenly each fires and stays silent
        redundancy  1 - entropy / sum of the neurons' entropies (0: independent neurons; near 1:
                    they fire together, the layer acts like fewer neurons)
        active      mean share of neurons firing per tick
    A change of the layer's size resets the counts.
    """

    counts: dict = field(default_factory=dict)
    fired: np.ndarray | None = None
    ticks: int = 0

    def observe(self, net, layer):
        on = np.asarray(net.layer_output(layer)).ravel() != 0
        if self.fired is None or self.fired.size != on.size:
            self.reset()
            self.fired = np.zeros(on.size)
        key = np.packbits(on).tobytes()
        self.counts[key] = self.counts.get(key, 0) + 1
        self.fired += on
        self.ticks += 1

    def reset(self):
        self.counts, self.fired, self.ticks = {}, None, 0

    def summary(self):
        if not self.ticks:
            return {"entropy": 0.0, "distinct": 0, "neuron_entropy": 0.0, "redundancy": 0.0, "active": 0.0}
        p = np.array(list(self.counts.values()), dtype=np.float64) / self.ticks
        entropy = float(-(p * np.log2(p)).sum())
        q = np.clip(self.fired / self.ticks, 1e-12, 1 - 1e-12)
        each = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        each[(self.fired == 0) | (self.fired == self.ticks)] = 0.0
        total = float(each.sum())
        return {"entropy": entropy, "distinct": len(self.counts), "neuron_entropy": float(each.mean()),
                "redundancy": 1.0 - entropy / total if total > 0 else 0.0,
                "active": float(self.fired.mean() / self.ticks)}


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
