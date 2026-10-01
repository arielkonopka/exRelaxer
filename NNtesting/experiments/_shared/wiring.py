"""Network building that remembers what every weight reads.

The library appends a dense neuron's weights for each new input at the end
of its weight vector: connecting a layer adds that layer's outputs to every
neuron of the target that does not read it yet, and a layer that gains
neurons (add_feedback) extends every neuron reading it. Which input a
weight belongs to therefore follows from the order of construction, and
`Wiring` mirrors those rules as it builds: `sources(layer, i)` lists, for
neuron i, the input each of its weights reads, as (layer name, index) or
(input source name, index). `check()` compares the mirror with the
library's weight counts.

Uses: scaling initial weights by kind (a layer reading itself), and
carrying evolved weights over to a grown network by the input they read
(`carry`), so that a network built one layer deeper starts by playing as
the shallower one did.

Experiments import it with sys.path pointing at this folder (the harness
skips folders whose name starts with "_").
"""
import math

import numpy as np


class Wiring:
    def __init__(self, net):
        self.net = net
        self.names = {}     # layer id -> name
        self._size = {}     # dense layer id -> neurons so far
        self._groups = {}   # dense layer id -> [{"neurons", "src", "reads"}], as the library's groups
        self._unwired = {}  # dense layer id -> neurons that read nothing yet

    # --- construction -------------------------------------------------
    def dense(self, name, spec, size):
        layer = self.net.add_layer(name, spec)
        self.names[layer] = name
        self._size[layer], self._groups[layer], self._unwired[layer] = size, [], list(range(size))
        return layer

    def other(self, name, spec):
        """A layer that is not dense (e.g. a cochlea): tracked as a source only."""
        layer = self.net.add_layer(name, spec)
        self.names[layer] = name
        return layer

    def inputs(self, layer, count, name):
        self.net.add_inputs(layer, count, name)
        if layer in self._groups:
            self._append(layer, None, [(name, j) for j in range(count)])

    def connect(self, source, target):
        self.net.connect(source, target)
        n = self._size.get(source) or self.net.layer_size(source)
        self._append(target, source, [(self.names[source], j) for j in range(n)])

    def feedback(self, source, target, n):
        """add_feedback: n new neurons in `target` that read `source`."""
        first = self._size[target]
        self.net.add_feedback(source, target, n)
        self._groups[target].append({"neurons": list(range(first, first + n)), "reads": {source},
                                     "src": [(self.names[source], j) for j in range(self._size[source])]})
        self._size[target] += n
        for groups in self._groups.values():  # every group reading `target` reads its new neurons too
            for g in groups:
                if target in g["reads"]:
                    g["src"] += [(self.names[target], j) for j in range(first, first + n)]

    def _append(self, layer, source, block):
        for g in self._groups[layer]:
            if source is not None and source in g["reads"]:
                continue
            g["src"] += block
            if source is not None:
                g["reads"].add(source)
        if self._unwired[layer]:
            self._groups[layer].append({"neurons": self._unwired[layer], "src": list(block),
                                        "reads": {source} if source is not None else set()})
            self._unwired[layer] = []

    # --- what the weights read ------------------------------------------
    def sources(self, layer, i):
        for g in self._groups[layer]:
            if i in g["neurons"]:
                return g["src"]
        return []

    def check(self, layers):
        for layer in layers:
            for i in range(self.net.layer_size(layer)):
                n = len(self.net.weights(layer, i))
                if len(self.sources(layer, i)) != n:
                    raise RuntimeError(f"wiring: {self.names[layer]}[{i}] has {n} weights, "
                                       f"the mirror {len(self.sources(layer, i))}")

    def init_weights(self, layers, recurrent_scale):
        """Uniform weights with variance 1 / fan-in, and recurrent_scale / sqrt(n)
        on the n weights by which a neuron reads its own layer (below 1 the echo fades)."""
        for layer in layers:
            name = self.names[layer]
            for i in range(self.net.layer_size(layer)):
                w = np.asarray(self.net.weights(layer, i), dtype=np.float32)
                own = np.array([s[0] == name for s in self.sources(layer, i)], dtype=bool)
                if own.any():
                    if (~own).any():
                        w[~own] *= math.sqrt(3.0 / (~own).sum())
                    w[own] *= recurrent_scale * math.sqrt(3.0 / own.sum())
                else:
                    w *= math.sqrt(3.0 / len(w))
                self.net.set_weights(layer, i, w)


def carry(old, new, arrays, fresh):
    """Maps flat weight vectors of one network onto another built from it.

    old, new: lists of (key, sources) in flat order, one per evolving neuron,
    key = (layer name, neuron index); arrays: flat vectors over `old` (the
    weights, and e.g. Adam's moments); fresh: the new network's own initial
    weights over `new`. A neuron that exists in both keeps its values by the
    input each weight reads, and weights from new inputs start at zero; a new
    neuron starts at `fresh` in the first array and at zero in the others."""
    out = []
    for k, a in enumerate(arrays):
        before, at = {}, 0
        for key, src in old:
            before[key] = dict(zip(src, a[at:at + len(src)]))
            at += len(src)
        parts, at = [], 0
        for key, src in new:
            if key in before:
                parts.append(np.array([before[key].get(s, 0.0) for s in src]))
            else:
                parts.append(fresh[at:at + len(src)] if k == 0 else np.zeros(len(src)))
            at += len(src)
        out.append(np.concatenate(parts))
    return out
