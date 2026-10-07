"""Per-synapse learning rules, the critic and curiosity, buses, and changing a running network
from Python."""
import numpy as np
import pytest

import exrelaxer as exr


def small(seed=11):
    exr.reseed(seed)
    net = exr.Network()
    h = net.add_layer("h", exr.LayerSpec.dense(6, False, True))
    out = net.add_layer("out", exr.LayerSpec.dense(3, False, False))
    net.add_inputs(h, 4, "x")
    net.connect(h, h)
    net.connect(h, out)
    net.add_output(out)
    return net, h, out


def run(net, ticks, start=0):
    ys = []
    for t in range(start, start + ticks):
        net.set_inputs("x", np.array([np.sin(0.4 * t), np.cos(0.9 * t), 0.5, -0.3], np.float32))
        net.step()
        ys.append(net.outputs())
    return np.array(ys)


def test_rules():
    for rule in (exr.LearningRule.eligibility(0.8), exr.LearningRule.eprop(0.5), exr.LearningRule.surrogate(4)):
        assert rule.per_synapse
    assert exr.LearningRule.surrogate(4).window == 4
    assert "eprop" in repr(exr.LearningRule.eprop())
    with pytest.raises(ValueError):
        exr.LearningRule.surrogate(0).validate()


def test_eprop_and_surrogate_learn_a_readout():
    for rule in (exr.LearningRule.eprop(), exr.LearningRule.surrogate(4)):
        exr.reseed(2)
        net = exr.Network()
        spec = exr.LayerSpec.dense(1, False, False, learning_rule=rule)
        spec.normalize = False  # gradients treat 1 / |w| as a constant
        out = net.add_layer("out", spec)
        net.add_inputs(out, 2)
        net.add_output(out)
        errors = []
        for t in range(300):
            x = np.array([np.sin(0.3 * t), 0.5], np.float32)
            net.set_inputs(x)
            net.step()
            e = 0.7 * x[0] - 0.2 * x[1] - net.outputs()[0]
            errors.append(abs(e))
            net.apply_error([float(e)], 0.05)
        assert np.mean(errors[-30:]) < 0.1 * np.mean(errors[:30]), rule
        if rule.type == exr.LearningRuleType.EProp:
            assert net.synapse_trace(out, 0).shape == (2,)
            assert set(net.derivatives(out)) == {"dyds", "dydthr", "dthrdthr", "dthrds"}


def test_grow_prune_freeze_and_save():
    a, h, _ = small()
    b, _, out = small()
    run(a, 10)
    run(b, 10)
    b.grow_layer(h, 4)  # outgoing weights start at zero
    assert b.layer_size(h) == 10
    np.testing.assert_array_equal(run(a, 10, 10), run(b, 10, 10))
    b.grow_layer(h, 2, exr.WeightInit.Random, freeze_existing=True)
    assert b.frozen_neurons(h) == list(range(10))
    b.prune_neurons(h, [0, 3])
    assert b.layer_size(h) == 10
    b.freeze_inputs(out, h)
    assert b.frozen_input_count(out) == 10
    b.freeze_inputs(h, "x")
    copy = exr.Network.from_bytes(b.to_bytes())
    np.testing.assert_array_equal(run(b, 5, 30), run(copy, 5, 30))
    assert copy.frozen_input_count(out) == 10


def test_bus_of_perceptrons():
    net = exr.Network()
    a = net.add_layer("a", exr.LayerSpec.dense(4))
    bus = net.add_bus("bus", exr.LayerSpec.perceptron(5, 0.1))
    r = net.add_layer("r", exr.LayerSpec.dense(2, False, True))
    net.add_inputs(a, 3, "x")
    net.connect_inputs("x", bus)
    net.write_bus(a, bus)
    net.subscribe(r, bus, exr.WeightInit.Zero)
    assert net.is_bus(bus) and net.buses == [bus] and net.is_frozen(bus)
    assert net.bus_writers(bus) == [a] and net.bus_readers(bus) == [r]
    assert net.layer_spec(bus).binary and net.layer_spec(bus).bus
    for t in range(5):
        net.set_inputs("x", np.array([np.sin(t), 0.5, -0.5], np.float32))
        net.step()
        assert set(np.unique(net.layer_output(bus))) <= {0.0, 1.0}
    assert "bus bus" in net.describe()


def test_critic_and_curiosity():
    net, h, out = small()
    net.set_critic(layers=[h], inputs=["x"], gamma=0.9, lambda_=0.8, rate=0.05)
    net.set_curiosity(predict_inputs=["x"], from_layers=[h], from_inputs=["x"], rate=0.2)
    assert net.has_critic and net.has_curiosity
    assert net.critic_spec.layers == [h]
    deltas, curious = [], []
    for t in range(300):
        run(net, 1, t)
        curious.append(net.curiosity_reward())
        deltas.append(net.apply_reward_td(1.0 + curious[-1], 0.01))
    assert deltas[0] == 0.0 and deltas[1] != 0.0
    assert curious[0] == 0.0
    assert np.mean(curious[-20:]) < np.mean(curious[1:21])
    assert net.critic_weights().shape == (6 + 4 + 1,)
    copy = exr.Network.from_bytes(net.to_bytes())
    assert copy.critic_value() == pytest.approx(net.critic_value())
    np.testing.assert_array_equal(copy.curiosity_prediction(), net.curiosity_prediction())
    net.reset_critic()
    net.remove_curiosity()
    assert not net.has_curiosity
