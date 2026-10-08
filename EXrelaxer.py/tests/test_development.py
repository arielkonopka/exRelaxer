"""Structural development from Python (doc/development.md): the protected minimum size, grown
neurons, pruning candidates, LIFO pruning and the E-R saturation monitor."""
import numpy as np
import pytest

import exrelaxer as exr


def watched(size=4, recovery=None):
    """x (one sensor) -> h (E-R, plain sums, weights 1) -> out."""
    exr.reseed(9)
    spec = exr.LayerSpec.dense(size, False, True)
    spec.normalize = False
    net = exr.Network()
    h = net.add_layer("h", spec)
    out = net.add_layer("out", exr.LayerSpec.dense(2, False, False))
    net.add_inputs(h, 1, "x")
    net.connect(h, out)
    net.add_output(out)
    if recovery is not None:
        net.set_recovery_jitter(h, exr.Jitter.uniform(1e-7).around(recovery))
    for i in range(size):
        net.set_weights(h, i, [1.0])
    return net, h, out


def tick(net, h, monitor, x):
    net.set_inputs("x", np.array([x], np.float32))
    net.step()
    monitor.observe(net, h)


def test_minimum_size_and_grown_neurons():
    net, h, _ = watched(4)
    assert net.minimum_size(h) == 0
    net.set_minimum_size(h, 4)
    with pytest.raises(ValueError):
        net.prune_neurons(h, [0])
    net.grow_layer(h, 2)
    net.grow_layer(h, 1)
    assert net.grown_neurons(h) == [4, 5, 6]
    assert net.growth_order(h) == [0, 0, 0, 0, 1, 2, 3]
    copy = exr.Network.from_bytes(net.to_bytes())
    assert copy.minimum_size(h) == 4
    assert copy.growth_order(h) == net.growth_order(h)


def test_start_from_one_neuron():
    net, h, _ = watched(1)
    net.set_minimum_size(h, 1)
    net.grow_layer(h, 2, freeze_existing=True)
    assert net.layer_size(h) == 3
    assert net.prune_newest(h, 5) == [2, 1]
    assert net.layer_size(h) == 1


def test_candidates_and_lifo():
    net, h, _ = watched(4)
    net.set_weights(h, 1, [float("nan")])
    net.set_weights(h, 2, [0.0])
    found = dict(net.prune_candidates(h))
    assert "invalid" in found[1]
    assert "zero_incoming" in found[2]
    assert 0 not in found
    net.grow_layer(h, 1)  # A
    net.grow_layer(h, 1)  # B
    net.grow_layer(h, 1)  # C
    assert "unread" in dict(net.prune_candidates(h))[6]  # zero outgoing weights
    net.set_minimum_size(h, 4)
    assert net.prune_newest(h) == [6]  # C first
    assert net.prune_newest(h) == [5]  # then B
    assert net.prune_newest(h, 9) == [4]
    assert net.prune_newest(h, 9) == []
    assert net.layer_size(h) == 4


def test_fatigue_is_not_saturation_but_persistent_silence_is():
    net, h, _ = watched(4)  # recovery 0.9: thresholds come back within ~20 ticks
    m = exr.ActivityMonitor(exr.ActivitySpec(window=100))
    for _ in range(3):
        tick(net, h, m, 5.0)
    for _ in range(300):
        tick(net, h, m, 0.5)
    assert not m.saturated

    net, h, _ = watched(4, recovery=0.999)  # thresholds stay above the input
    m = exr.ActivityMonitor(exr.ActivitySpec(window=100))
    for _ in range(3):
        tick(net, h, m, 5.0)
    for _ in range(150):
        tick(net, h, m, 0.5)
    assert m.saturated
    assert m.states()[0] == exr.TickState.Fatigued
    assert m.inactive_ticks() == [0, 0, 0, 0]  # silent, but fatigued: not inactive
    probe = net.state_probe(h)
    assert probe["last_threshold"][0] > probe["last_sum"][0] > 0
    found = dict(net.prune_candidates(h, m, inactive_after=50))
    assert all("inactive" not in reasons for reasons in found.values())
