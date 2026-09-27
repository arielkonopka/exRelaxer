"""exrelaxer: biologically inspired neurons for Python.

Neurons adapt through excitation-relaxation (an adaptive firing threshold)
and habituation, and learn through a reward-modulated Hebbian rule. The
networks, layers and kernels are the C++ library's (see the repository's
doc/ folder); this package binds them with nanobind.

    import numpy as np
    import exrelaxer as exr

    exr.reseed(1)
    net = exr.Network()
    inp = net.add_layer("in", exr.LayerSpec.dense(16))
    hid = net.add_layer("hid", exr.LayerSpec.dense(32))
    out = net.add_layer("out", exr.LayerSpec.dense(1, er=False))
    net.add_inputs(inp, 2)
    net.connect(inp, hid)
    net.connect(hid, out)
    net.add_output(out)

    for t in range(1000):
        net.set_inputs(np.array([0.5, -0.2], dtype=np.float32))
        net.step()
        y = net.outputs()
        net.apply_reward(1.0 if y[0] <= 0 else 0.0, 0.005)   # error-driven

    ys = net.run(np.zeros((100, 2), np.float32))   # 100 ticks in one call: 100 x 1
"""
from ._core import (  # noqa: F401
    CochleaSpec,
    Compression,
    DeserializeMode,
    DisparityMeasure,
    DisparitySpec,
    Edge,
    EdgeKind,
    FrequencyScale,
    Habituation,
    Interpolation,
    Jitter,
    LayerSpec,
    LayerType,
    LearningRule,
    LearningRuleType,
    Network,
    PoolMode,
    ResizeSpec,
    RetinaSpec,
    Sampling,
    Shape,
    Spontaneous,
    ThresholdGrowth,
    Window2D,
    build_info,
    constants,
    filters,
    reseed,
    set_threads,
    threads,
)
from . import audio, datasets  # noqa: F401,E402

__all__ = [
    "CochleaSpec", "Compression", "DeserializeMode", "DisparityMeasure", "DisparitySpec", "Edge", "EdgeKind",
    "FrequencyScale", "Habituation", "Interpolation", "Jitter", "LayerSpec", "LayerType", "LearningRule",
    "LearningRuleType", "Network", "PoolMode", "ResizeSpec",
    "RetinaSpec", "Sampling", "Shape", "Spontaneous", "ThresholdGrowth", "Window2D", "audio", "build_info", "constants", "datasets", "filters", "reseed",
    "set_threads", "threads",
]
