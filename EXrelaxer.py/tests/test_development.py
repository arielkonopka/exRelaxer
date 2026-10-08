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


from exrelaxer.development import Plateau, Pruning, WidthGrowth, grow_depth  # noqa: E402


def saturate(net, h, monitor, ticks=150):
    for _ in range(3):
        tick(net, h, monitor, 5.0)
    for _ in range(ticks):
        tick(net, h, monitor, 0.5)


def test_width_growth_adds_plastic_neurons_and_keeps_old_ones():
    net, h, out = watched(4, recovery=0.999)
    m = exr.ActivityMonitor(exr.ActivitySpec(window=100))
    width = WidthGrowth(increment=2, max_size=8)
    assert width.update(net, h, m) == 0  # nothing observed: no growth
    saturate(net, h, m)
    old = [net.weights(h, i).copy() for i in range(4)]
    assert width.update(net, h, m, tick=1) == 2
    assert net.layer_size(h) == 6
    assert net.frozen_neurons(h) == [0, 1, 2, 3]
    assert net.grown_neurons(h) == [4, 5]
    assert not m.saturated  # starts again
    # New neurons start fresh, not with the population's raised thresholds.
    th = net.neuron_state(h)["threshold"]
    assert th[4] == th[5] == exr.constants.baseline_threshold
    assert th[0] > 1.0
    # Learning: old weights stay, new ones change.
    new = [net.weights(h, i).copy() for i in (4, 5)]
    for t in range(50):
        tick(net, h, m, 1.0 + 0.5 * np.sin(t))
        net.apply_reward(1.0, 0.05)
    for i in range(4):
        np.testing.assert_array_equal(net.weights(h, i), old[i])
    assert any(not np.array_equal(net.weights(h, i), w) for i, w in zip((4, 5), new))
    # The limit: one more growth to 8, then none.
    saturate(net, h, m, 300)
    width.update(net, h, m, tick=2)
    saturate(net, h, m, 300)
    width.update(net, h, m, tick=3)
    assert net.layer_size(h) <= 8
    assert width.events[0] == (1, 4, 6)


def test_pruning_policy_respects_minimum_and_prefers_newest():
    net, h, _ = watched(4)
    net.set_minimum_size(h, 4)
    net.grow_layer(h, 3)
    for i in range(7):
        net.set_weights(h, i, [float("nan")])  # every neuron invalid
    prune = Pruning(reasons=("invalid",))
    assert prune.candidates(net, h) == [6, 5, 4]
    assert prune.update(net, h, tick=7) == [4, 5, 6]
    assert net.layer_size(h) == 4
    assert prune.update(net, h) == []  # at the minimum: base neurons stay, even invalid ones


def test_read_strength_and_weak_pruning():
    net, h, out = watched(4)
    assert np.all(np.isinf(net.read_strength(out)))  # an output layer is read as a whole
    for j, w in enumerate(([0.5, 0.01, -0.6, 0.0], [0.4, 0.02, 0.1, 0.0])):
        net.set_weights(out, j, w)
    assert np.allclose(net.read_strength(h), [0.5, 0.02, 0.6, 0.0])
    assert [i for i, why in net.prune_candidates(h) if "unread" in why] == [3]
    prune = Pruning(reasons=("weak",), weak_fraction=0.2, weak_after=10)
    assert prune.candidates(net, h, tick=0) == []      # too young to judge
    assert prune.candidates(net, h, tick=9) == []
    assert prune.candidates(net, h, tick=10) == [3, 1]  # below 0.2 x mean 0.28
    net.set_minimum_size(h, 3)
    assert prune.update(net, h, tick=10) == [3]         # the minimum leaves room for one


def test_plateau_needs_improvement_then_a_plateau():
    p = Plateau(baseline=0.40, patience=3)
    assert not any(p.update(s) for s in [0.40, 0.38, 0.39, 0.40, 0.37, 0.40, 0.40])  # never better
    p = Plateau(baseline=0.40, patience=3)
    fired = [p.update(s) for s in [0.42, 0.48, 0.53, 0.57, 0.59, 0.60, 0.60, 0.60, 0.60]]
    assert fired == [False] * 8 + [True]
    p.restart()
    assert p.baseline == 0.60 and not p.update(0.60)


def test_depth_growth_freezes_old_layer_and_changes_nothing_at_first():
    a, h, out = watched(4)
    b, _, _ = watched(4)
    xs = [1.0 + np.sin(0.3 * t) for t in range(40)]
    ya, yb = [], []
    for x in xs[:20]:
        for net, ys in ((a, ya), (b, yb)):
            net.set_inputs("x", np.array([x], np.float32))
            net.step()
    deep = grow_depth(b, h, readers=[out], spec=exr.LayerSpec.dense(3, False, True))
    assert b.layer_name(deep) == "h_deep1"
    assert b.layer_spec(deep).grown
    assert b.frozen_neurons(h) == [0, 1, 2, 3]
    assert b.frozen_neurons(deep) == []
    for x in xs[20:]:
        for net, ys in ((a, ya), (b, yb)):
            net.set_inputs("x", np.array([x], np.float32))
            net.step()
            ys.append(net.outputs())
    np.testing.assert_array_equal(np.array(ya), np.array(yb))


def test_development_with_buses_survives_save_and_load():
    """Learning + width growth + pruning + depth growth on a network fed through a bus."""
    exr.reseed(3)
    net = exr.Network()
    bus = net.add_bus("senses", exr.LayerSpec.dense(4, False, False))
    spec = exr.LayerSpec.dense(1, False, True)  # one cell to start with
    spec.normalize = False
    h = net.add_layer("h", spec)
    out = net.add_layer("out", exr.LayerSpec.dense(3, False, False))
    net.add_inputs(bus, 2, "x")
    net.subscribe(h, bus)
    net.connect(h, out)
    net.add_output(out)
    net.set_minimum_size(h, 1)
    net.set_recovery_jitter(h, exr.Jitter.uniform(1e-7).around(0.995))

    m = exr.ActivityMonitor(exr.ActivitySpec(window=50, blocked_share=0.5))
    width = WidthGrowth(increment=2, max_size=5)
    prune = Pruning(reasons=("invalid",))
    for t in range(2000):
        x = 3.0 if t % 400 < 5 else 0.3  # a strong burst, then weak input
        net.set_inputs("x", np.array([x, -x], np.float32))
        net.step()
        m.observe(net, h)
        net.apply_reward(0.1, 0.01)
        width.update(net, h, m, t)
    assert width.events, "the slowly recovering population never saturated"
    net.set_weights(h, net.layer_size(h) - 1, [float("nan")] * 4)
    assert len(prune.update(net, h, m, t)) == 1
    deep = grow_depth(net, h, readers=[out], spec=exr.LayerSpec.dense(2, False, True), buses=[bus])
    for t in range(50):
        net.set_inputs("x", np.array([0.5, -0.5], np.float32))
        net.step()
        net.apply_reward(0.1, 0.01)

    copy = exr.Network.from_bytes(net.to_bytes())
    assert copy.layer_size(h) == net.layer_size(h)
    assert copy.minimum_size(h) == 1
    assert copy.growth_order(h) == net.growth_order(h)
    assert copy.frozen_neurons(h) == net.frozen_neurons(h)
    assert copy.layer_spec(deep).grown
    assert copy.bus_readers(bus) == net.bus_readers(bus) == [h, deep]
    np.testing.assert_array_equal(copy.neuron_state(h)["threshold"], net.neuron_state(h)["threshold"])
    for t in range(20):
        for n in (net, copy):
            n.set_inputs("x", np.array([np.sin(t), 1.0], np.float32))
            n.step()
        np.testing.assert_array_equal(net.outputs(), copy.outputs())
