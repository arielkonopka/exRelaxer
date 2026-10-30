"""Shared pieces of the TD actor-critic experiments: the frozen random E-R
reservoir and how its activity becomes the critic's and the actor's features."""
import math

import numpy as np

import exrelaxer as exr


def recovery_jitter(r, spread=0.0):
    """Every neuron gets recovery r exactly (0.5 is the library default), or one drawn
    uniformly from r +- spread: a spectrum of time constants, so the thresholds decay at
    different rates after an event and together tell how long ago it was."""
    if spread > 0:
        return exr.Jitter.uniform(spread).around(r)
    return (exr.Jitter.uniform(1e-7) if r != 0.5 else exr.Jitter.none()).around(r)


class Reservoir:
    """inputs (named sources) -> frozen random E-R layer that also reads itself one tick late.

    features: "output" (the layer's outputs), "threshold" (each neuron's E-R threshold, the
    history its firing leaves), "both", or "none" (only the `extra` raw inputs: a control).
    """

    def __init__(self, sources, width=128, recurrent_scale=0.5, recovery=0.9, spread=0.0, habituation=False, model="er",
                 features="both", ticks=1, extra=None, vision=None, with_pool=False):
        self.net = net = exr.Network()
        front = None
        if vision is not None:
            front = self._vision(net, sources, **vision)
        er = model == "er"
        spec = exr.LayerSpec.dense(width, habituation, er, frozen=True)
        if er:
            spec.recovery_jitter = recovery_jitter(recovery, spread)
        else:
            spec.rectify = True
        self.h = h = net.add_layer("reservoir", spec)
        self.sources = dict(sources)
        if front is not None:
            net.connect(front, h)
        else:
            for name, size in self.sources.items():
                net.add_inputs(h, size, name)
        recurrent = recurrent_scale > 0
        if recurrent:
            net.connect(h, h)
        for i in range(width):
            w = np.asarray(net.weights(h, i), dtype=np.float32)
            if recurrent:
                w[:-width] *= math.sqrt(3.0 / (len(w) - width))
                w[-width:] *= recurrent_scale * math.sqrt(3.0 / width)
            else:
                w *= math.sqrt(3.0 / len(w))
            net.set_weights(h, i, w)
        net.add_output(h)
        self.features_kind, self.ticks, self.width = features, ticks, width
        self.extra = extra or 0  # raw inputs appended to the features (e.g. a bias or the state itself)
        self.with_pool = with_pool and front is not None  # the frozen vision features too
        self.pool_size = net.layer_size(front) if front is not None else 0
        self.spikes = 0

    def _vision(self, net, sources, height, width, kernel=5, orientations=4, wavelength=4.0, sigma=1.5,
                pool_y=6, pool_x=4):
        """Frozen early vision for the one source (an image `height` x `width`): a Gabor bank
        (`orientations` x 2 phases) and an on and an off centre-surround filter, rectified,
        then max-pooled over pool_y x pool_x squares. Returns the pool layer."""
        (name, size), = sources.items()
        image = exr.Shape(1, height, width)
        eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image), False, False))
        net.add_inputs(eye, image, name)
        bank = list(exr.filters.gabor_bank(kernel, orientations, wavelength, sigma)) + \
            list(exr.filters.centre_surround_bank(kernel, 0.8, 2.0))
        spec = exr.LayerSpec.conv2d(len(bank), exr.Window2D(kernel, kernel, 1, 1, kernel // 2, kernel // 2), False, False,
                                    frozen=True)
        spec.rectify = True
        self.v1 = v1 = net.add_layer("v1", spec)
        net.connect(eye, v1)
        net.load_filters(v1, bank)
        self.pool = pool = net.add_layer("pool", exr.LayerSpec.pool2d(exr.Window2D(pool_y, pool_x, pool_y, pool_x)))
        net.connect(v1, pool)
        return pool

    @property
    def size(self):
        return {"none": 0, "output": 1, "threshold": 1, "both": 2}[self.features_kind] * self.width + self.extra + \
            (self.pool_size if self.with_pool else 0)

    def step(self, inputs, raw=None):
        """Sets the named inputs, runs the ticks, returns the feature vector."""
        for name, value in inputs.items():
            self.net.set_inputs(name, np.asarray(value, dtype=np.float32))
        for _ in range(self.ticks):
            self.net.step()
        out = np.asarray(self.net.layer_output(self.h), dtype=np.float64).ravel()
        self.spikes += int(np.count_nonzero(np.abs(out) > 1e-6))
        parts = []
        if self.features_kind in ("output", "both"):
            parts.append(out)
        if self.features_kind in ("threshold", "both"):
            thr = np.asarray(self.net.neuron_state(self.h)["threshold"], dtype=np.float64)
            parts.append(thr)  # rests near 0, jumps when the neuron fires, decays back at `recovery`
        if self.with_pool:
            parts.append(np.asarray(self.net.layer_output(self.pool), dtype=np.float64).ravel())
        if self.extra:
            parts.append(np.asarray(raw, dtype=np.float64))
        return np.concatenate(parts)

    def reset(self):
        self.net.reset_state(self.h)
