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
    skip_muted      muted neurons (Network.mute_neurons) are never removed: whoever muted them (a
                    ShadowPool, significance pruning) manages them

    `events` lists (tick, [removed indices], size after) per pruning.
    """

    reasons: tuple = ("invalid", "inactive")
    inactive_after: int = 2000
    only_grown: bool = False
    weak_fraction: float = 0.2
    weak_after: int = 20000
    skip_muted: bool = True
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
        if self.skip_muted:
            muted = set(int(i) for i in net.muted_neurons(layer))
            found = [i for i in found if i not in muted]
        found.sort(key=lambda i: (order[i], i), reverse=True)
        room = net.layer_size(layer) - net.minimum_size(layer)
        if self.skip_muted:
            room -= len(net.muted_neurons(layer))
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


# --- Error-driven growth, shadow candidates, synaptic scaling and significance (research §69) ---

@dataclass
class ErrorPlateau:
    """Width-growth trigger from a TD critic's error: the critic is still learning hard, yet its
    error has stopped falling (Dynamic Node Creation, Ash 1989; Cascade-Correlation, Fahlman &
    Lebiere 1990, on a critic's TD error).

    Feed it every tick: update(delta, target, effort), where target is the TD target the critic
    was moved toward (r + gamma V(s'), r at a terminal step) and effort how much the critic's
    weights moved this tick (e.g. mean |dw|). Per window of `window` ticks it computes
        error     mean delta^2 / variance of the target: the share of the target the critic
                  does not explain (1: no better than a constant, 0: perfect). Unlike mean |delta|
                  it does not rise just because the agent got better and meets larger rewards
                  and values.
        effort    mean effort
    and returns "grow" when, for `patience` windows in a row,
        the error fell by less than min_progress (relative to the best window so far),
        effort >= min_effort x the first window's effort (it is still being trained, not merely
                  stopped: a rate decayed to nothing is not a reason to grow), and
        error > goal (not solved).
    After a growth it judges the growth `probation` windows later: "keep" if the error is below
    (1 - min_gain) x its value at the growth, else "undo" (the caller removes the neurons) and the
    patience before the next growth doubles (up to max_patience). Returns None otherwise.
    judge=False skips the probation (the caller judges growth another way, e.g. by the score;
    in RL the critic's error is a moving target: a better policy meets new states and the error
    can rise just because the agent improved, research §69); call backoff() after an undo.

    `history` holds (window error, effort) per window and `events` (tick, verdict).
    """

    window: int = 2000
    patience: int = 3
    min_progress: float = 0.02
    min_effort: float = 0.25
    goal: float = 0.0
    probation: int = 3
    min_gain: float = 0.0
    max_patience: int = 24
    judge: bool = True
    history: list = field(default_factory=list)
    events: list = field(default_factory=list)
    _ticks: int = 0
    _sq_delta: float = 0.0
    _sum_target: float = 0.0
    _sq_target: float = 0.0
    _effort: float = 0.0
    _first_effort: float | None = None
    _best: float | None = None
    _stale: int = 0
    _pending: tuple | None = None   # (windows since growth, error at growth)
    _wait: int = 0

    def __post_init__(self):
        self._wait = self.patience

    def update(self, delta, target, effort=0.0, tick=None):
        self._ticks += 1
        self._sq_delta += float(delta) ** 2
        self._sum_target += float(target)
        self._sq_target += float(target) ** 2
        self._effort += float(effort)
        if self._ticks < self.window:
            return None
        n = self._ticks
        var = max(self._sq_target / n - (self._sum_target / n) ** 2, 1e-12)
        error = (self._sq_delta / n) / var
        effort = self._effort / n
        self._ticks, self._sq_delta, self._sum_target, self._sq_target, self._effort = 0, 0.0, 0.0, 0.0, 0.0
        self.history.append((error, effort))
        if self._first_effort is None:
            self._first_effort = effort
        if self._pending is not None:
            age, before = self._pending
            age += 1
            if age < self.probation:
                self._pending = (age, before)
                return None
            self._pending = None
            self._best, self._stale = error, 0
            if error < (1.0 - self.min_gain) * before:
                self._wait = self.patience
                self.events.append((tick, "keep"))
                return "keep"
            self._wait = min(self._wait * 2, self.max_patience)
            self.events.append((tick, "undo"))
            return "undo"
        if self._best is None or error < self._best * (1.0 - self.min_progress):
            self._best, self._stale = error, 0
            return None
        self._stale += 1
        busy = effort >= self.min_effort * self._first_effort
        if self._stale >= self._wait and busy and error > self.goal:
            self._pending = (0, error) if self.judge else None
            self._best, self._stale = None, 0
            self.events.append((tick, "grow"))
            return "grow"
        return None

    def backoff(self, kept=False):
        """A growth judged by the caller: kept resets the patience, undone doubles it."""
        self._wait = self.patience if kept else min(self._wait * 2, self.max_patience)

    @property
    def judging(self):
        """A growth is on probation."""
        return self._pending is not None

    @property
    def last_error(self):
        return self.history[-1][0] if self.history else None


@dataclass
class ShadowPool:
    """Shadow candidates (Cascade-Correlation, run online): `size` extra neurons of a Dense E-R
    layer that see its input but are muted (nobody reads them) and frozen for the library's own
    rule. This object trains them instead, each to follow a target signal (a critic's TD error):

        dw = rate * sign(corr) * (target - mean) * gate * x_prev / (1 + |x_prev|^2)

    where corr is the neuron's running correlation with the target, x_prev the layer's input one
    tick earlier (the TD error at t is about the state the critic valued at t-1), and gate 1 when
    the neuron fired at t-1, else `leak` (so the E-R step makes the features non-linear and the
    candidates do not all converge to the same linear direction).

    promote(n) unmutes (and unfreezes, unless keep_frozen) the n candidates with the largest
    |corr| and refills the pool with new random ones; `revive(indices)` puts muted neurons (from
    significance pruning) back in the pool, so growth first reuses them. Candidates are told
    apart by Network.growth_order, so pruning elsewhere in the layer does not confuse them.
    `events` lists (tick, promoted growth orders, their correlations).
    """

    size: int = 8
    rate: float = 0.02
    leak: float = 0.1
    decay: float = 0.999
    keep_frozen: bool = False
    train: bool = True
    members: dict = field(default_factory=dict)  # growth order -> running stats [m_y, m_yy, m_yt]
    events: list = field(default_factory=list)
    _mean_t: float = 0.0
    _var_t: float = 1e-6
    _x_prev: np.ndarray | None = None
    _y_prev: dict | None = None
    _source: int | None = None

    def indices(self, net, layer):
        order = [int(o) for o in net.growth_order(layer)]
        return [i for i, o in enumerate(order) if o in self.members]

    def fill(self, net, layer, source=None):
        """Grows the pool back to `size` (new neurons: random incoming, zero outgoing, muted, frozen).
        `source`: the layer whose output is the pool's input (for training)."""
        if source is not None:
            self._source = source
        missing = self.size - len(self.members)
        if missing <= 0:
            return 0
        first = net.layer_size(layer)
        net.grow_layer(layer, missing, WeightInit.Zero, freeze_existing=False)
        net.mute_neurons(layer, first, missing)
        net.freeze_neurons(layer, first, missing)
        order = net.growth_order(layer)
        for i in range(first, first + missing):
            self.members[int(order[i])] = [0.0, 1e-6, 0.0]
        return missing

    def revive(self, net, layer, indices):
        """Muted neurons join the pool (kept muted and frozen)."""
        order = net.growth_order(layer)
        for i in indices:
            net.mute_neurons(layer, int(i), 1)
            net.freeze_neurons(layer, int(i), 1)
            key = int(order[int(i)])
            if key == 0:   # a base neuron has no unique growth order: it cannot join
                continue
            self.members.setdefault(key, [0.0, 1e-6, 0.0])

    def correlations(self, net, layer):
        """{index: running correlation with the target} for the pool's neurons."""
        out = {}
        order = net.growth_order(layer)
        for i in self.indices(net, layer):
            m_y, m_yy, m_yt = self.members[int(order[i])]
            var_y = max(m_yy - m_y * m_y, 1e-9)
            out[i] = m_yt / np.sqrt(var_y * self._var_t)
        return out

    def observe(self, net, layer, target):
        """One tick, after net.step() and once the target (TD error) for the previous tick is known."""
        idx = self.indices(net, layer)
        if not idx:
            return
        order = net.growth_order(layer)
        y_now = np.asarray(net.state_probe(layer)["output"], dtype=np.float64)
        x_now = np.asarray(net.layer_output(self._source), dtype=np.float64).ravel() if self._source is not None else None
        if self._y_prev is not None:
            t = float(target) - self._mean_t
            a = 1.0 - self.decay
            self._mean_t += a * t
            self._var_t += a * (t * t - self._var_t)
            for i in idx:
                key = int(order[i])
                if key not in self._y_prev:
                    continue
                y = self._y_prev[key]
                st = self.members[key]
                st[0] += a * (y - st[0])
                st[1] += a * (y * y - st[1])
                st[2] += a * ((y - st[0]) * t - st[2])
                if self.train and self._x_prev is not None and t != 0.0:
                    gate = 1.0 if y != 0.0 else self.leak
                    sign = 1.0 if st[2] >= 0.0 else -1.0
                    x = self._x_prev
                    w = np.asarray(net.weights(layer, i), dtype=np.float64)
                    if w.size == x.size:
                        w += self.rate * sign * t * gate * x / (1.0 + float(x @ x))
                        net.set_weights(layer, i, np.clip(w, -10.0, 10.0).astype(np.float32))
        self._y_prev = {int(order[i]): float(y_now[i]) for i in idx}
        self._x_prev = x_now

    def promote(self, net, layer, count, tick=None, rng=None):
        """Unmutes the `count` best candidates (largest |corr|, or random when not training);
        returns their indices."""
        corr = self.correlations(net, layer)
        if not corr:
            return []
        if self.train:
            ranked = sorted(corr, key=lambda i: (-abs(corr[i]), i))
        else:
            ranked = list(corr)
            (rng or np.random.default_rng(len(self.events))).shuffle(ranked)
        chosen = sorted(ranked[:count])
        order = net.growth_order(layer)
        for i in chosen:
            net.mute_neurons(layer, i, 1, False)
            if not self.keep_frozen:
                net.freeze_neurons(layer, i, 1, False)
            del self.members[int(order[i])]
        self.events.append((tick, [int(order[i]) for i in chosen], [float(corr[i]) for i in chosen]))
        return chosen

    def forget(self, net, layer, removed_orders):
        """Pool neurons removed from the layer elsewhere."""
        for o in removed_orders:
            self.members.pop(int(o), None)
            if self._y_prev:
                self._y_prev.pop(int(o), None)


@dataclass
class SynapticScaling:
    """Weight saturation per neuron, as the user proposed (research §69): the first time a neuron's
    incoming weights press against the limit, all of them are scaled down together (synaptic
    scaling, Turrigiano 1998); if the same neuron saturates again within `repeat_window` ticks it
    is split in two (Network.copy_neuron with opposite noise, the copy as a newborn).

    limit       |w| at or above which a weight counts as saturated (the clamp, or a lower synapse
                maximum)
    share       a neuron is saturated when at least this share of its weights is saturated
    target      after scaling, the neuron's largest |w| is target x limit
    every       ticks between checks (reading weights costs a copy per neuron)
    `events` lists (tick, index, "scale"|"split").
    """

    limit: float = 10.0
    share: float = 0.1
    target: float = 0.5
    repeat_window: int = 20000
    every: int = 200
    noise: float = 0.05
    split: float = 0.5
    max_size: int | None = None
    events: list = field(default_factory=list)
    _last: dict = field(default_factory=dict)   # growth order or ("base", index) -> tick of last scaling

    def check(self, net, layer, tick, skip=()):
        """Scales or splits saturated neurons; returns the indices of new neurons (splits)."""
        if tick % self.every:
            return []
        new = []
        order = net.growth_order(layer)
        size = net.layer_size(layer)
        skip = set(skip)
        for i in range(size):
            if i in skip:
                continue
            w = np.asarray(net.weights(layer, i), dtype=np.float64)
            if w.size == 0 or np.mean(np.abs(w) >= self.limit * (1 - 1e-6)) < self.share:
                continue
            key = int(order[i]) if order[i] else ("base", i)
            last = self._last.get(key)
            room = self.max_size is None or net.layer_size(layer) < self.max_size
            if last is not None and tick - last <= self.repeat_window and room:
                net.grow_layer(layer, 1, WeightInit.Zero, freeze_existing=False)
                target = net.layer_size(layer) - 1
                net.copy_neuron(layer, i, target, self.noise, self.split)
                # halve both and push them apart: they start as one neuron, but not as twins
                for k, sign in ((i, 1.0), (target, -1.0)):
                    v = np.asarray(net.weights(layer, k), dtype=np.float64) * self.target
                    v += sign * self.noise * self.limit * np.random.default_rng([tick, k]).uniform(-1, 1, v.size)
                    net.set_weights(layer, k, np.clip(v, -self.limit, self.limit).astype(np.float32))
                self._last.pop(key, None)
                self.events.append((tick, i, "split"))
                new.append(target)
            else:
                scale = self.target * self.limit / max(np.abs(w).max(), 1e-9)
                net.set_weights(layer, i, (w * scale).astype(np.float32))
                self._last[key] = tick
                self.events.append((tick, i, "scale"))
        return new


@dataclass
class Significance:
    """How much each neuron of a layer contributes to its readers, as a running mean of
        sum over readers r of |w_r,i * y_i| / ||w_r||
    (for linear readers this is exactly how much muting neuron i moves their sums). Feed it every
    tick with observe(net, layer, readers[, extra]), where `extra` adds per-neuron contributions of
    readers outside the network (e.g. a Python critic: |w_i * y_i|).

    prune(...) mutes, every `every` ticks after `after` ticks, the active neuron whose significance
    is lowest, if it is below fraction x the active mean and the layer keeps `minimum` active
    neurons. A muted neuron goes on probation: `judge(error)` (call once per evaluation window
    with the critic's error) unmutes it if the error rose by more than `tolerance` (relative)
    within `probation` windows, else hands it back as finally pruned.
    """

    decay: float = 0.999
    after: int = 20000
    every: int = 2000
    fraction: float = 0.2
    minimum: int = 1
    probation: int = 3
    tolerance: float = 0.05
    scores: dict = field(default_factory=dict)   # growth order or ("base", index) -> running value
    events: list = field(default_factory=list)
    trial: tuple | None = None   # (index key, windows, error before)

    @staticmethod
    def key(order, i):
        return int(order[i]) if order[i] else ("base", i)

    def observe(self, net, layer, readers, extra=None):
        y = np.abs(np.asarray(net.state_probe(layer)["output"], dtype=np.float64))
        c = np.zeros_like(y)
        for r in readers:
            for k in range(net.layer_size(r)):
                w = np.asarray(net.weights(r, k), dtype=np.float64)
                norm = float(np.linalg.norm(w)) or 1.0
                c += np.abs(w[:y.size]) * y / norm
        if extra is not None:
            c += np.asarray(extra, dtype=np.float64)[:y.size]
        order = net.growth_order(layer)
        a = 1.0 - self.decay
        for i in range(y.size):
            k = self.key(order, i)
            self.scores[k] = self.scores.get(k, c[i]) + a * (c[i] - self.scores.get(k, c[i]))

    def values(self, net, layer):
        order = net.growth_order(layer)
        return np.array([self.scores.get(self.key(order, i), 0.0) for i in range(net.layer_size(layer))])

    def prune(self, net, layer, tick, active, error=None):
        """Mutes the least significant of `active` (indices); returns it (or None)."""
        if self.trial is not None or tick < self.after or tick % self.every or len(active) <= self.minimum:
            return None
        v = self.values(net, layer)
        vals = v[list(active)]
        j = int(np.argmin(vals))
        if vals[j] >= self.fraction * vals.mean():
            return None
        i = int(active[j])
        net.mute_neurons(layer, i, 1)
        order = net.growth_order(layer)
        self.trial = (self.key(order, i), 0, error)
        self.events.append((tick, i, "mute"))
        return i

    def judge(self, net, layer, error, tick=None):
        """Once per window: returns ("restore", index), ("pruned", index) or None."""
        if self.trial is None:
            return None
        key, age, before = self.trial
        order = net.growth_order(layer)
        idx = [i for i in range(net.layer_size(layer)) if self.key(order, i) == key]
        if not idx:
            self.trial = None
            return None
        i = idx[0]
        age += 1
        if before is not None and error is not None and error > (1.0 + self.tolerance) * before:
            net.mute_neurons(layer, i, 1, False)
            self.trial = None
            self.events.append((tick, i, "restore"))
            return ("restore", i)
        if age >= self.probation:
            self.trial = None
            self.events.append((tick, i, "pruned"))
            return ("pruned", i)
        self.trial = (key, age, before)
        return None
