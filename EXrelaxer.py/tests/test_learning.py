"""Learning rules from Python: a rule per layer, apply_error, bias, save/load."""
import numpy as np
import pytest

import exrelaxer as exr


def test_rules_and_their_options():
    assert exr.LayerSpec.dense(3).learning_rule == exr.LearningRule.sign()
    rule = exr.LearningRule.traced(0.5, 0.1).with_bias().with_decay(0.01)
    assert rule.type == exr.LearningRuleType.Trace
    assert rule.bias and rule.trace == pytest.approx(0.5) and rule.decay == pytest.approx(0.01)
    assert "trace" in repr(rule)
    assert exr.LearningRule.oja(2).unsupervised
    with pytest.raises(ValueError):
        exr.LearningRule.traced(1.5).validate()


def test_one_network_with_a_rule_per_layer():
    exr.reseed(3)
    net = exr.Network()
    hid = net.add_layer("hid", exr.LayerSpec.dense(8, False, False, learning_rule=exr.LearningRule.oja()))
    fa = net.add_layer("fa", exr.LayerSpec.dense(6, False, False,
                                                 learning_rule=exr.LearningRule.feedback_alignment()))
    out = net.add_layer("out", exr.LayerSpec.dense(2, False, False,
                                                   learning_rule=exr.LearningRule.traced().with_bias()))
    net.add_inputs(hid, 3)
    net.connect(hid, fa)
    net.connect(fa, out)
    net.add_output(out)
    before = [net.weights(layer, 0).copy() for layer in (hid, fa, out)]
    rng = np.random.default_rng(0)
    for _ in range(20):
        net.set_inputs(rng.uniform(-1, 1, 3).astype(np.float32))
        net.step()
        net.apply_error(np.array([0.5, -0.5], np.float32), 0.01)
        net.apply_reward(1.0, 0.01)
    for layer, w in zip((hid, fa, out), before):
        assert not np.array_equal(net.weights(layer, 0), w), net.layer_name(layer)
    assert net.bias(out).shape == (2,)
    assert "oja" in net.describe() and "fa" in net.describe()

    restored = exr.Network.from_bytes(net.to_bytes())
    assert restored.layer_spec(out).learning_rule == exr.LearningRule.traced().with_bias()
    np.testing.assert_array_equal(restored.bias(out), net.bias(out))

    net.set_learning_rule(hid, exr.LearningRule.bcm())
    assert net.layer_spec(hid).learning_rule == exr.LearningRule.bcm()
    with pytest.raises(ValueError):
        net.apply_error([1.0], 0.1)


def test_modulators_per_neuron():
    net = exr.Network()
    out = net.add_layer("out", exr.LayerSpec.dense(2, False, False,
                                                   learning_rule=exr.LearningRule.feedback_alignment()))
    net.add_inputs(out, 1)
    net.add_output(out)
    net.set_weights(out, 0, [0.0])
    net.set_weights(out, 1, [0.0])
    net.set_inputs([1.0])
    net.step()
    net.apply_modulators_to(out, np.array([1.0, -1.0], np.float32), 0.1)
    assert net.weights(out, 0)[0] > 0 > net.weights(out, 1)[0]


def test_reset_traces_starts_an_episode_with_empty_traces():
    """With a trace, an update right after reset_traces sees only the current
    input, as on a fresh network; weights and bias are kept."""
    def net_with_trace():
        exr.reseed(5)
        net = exr.Network()
        out = net.add_layer("out", exr.LayerSpec.dense(
            1, False, False, learning_rule=exr.LearningRule.feedback_alignment(0.9).with_bias()))
        net.add_inputs(out, 2)
        net.add_output(out)
        return net, out

    fresh, a = net_with_trace()
    used, b = net_with_trace()
    for _ in range(5):
        used.set_inputs(np.array([1.0, 0.0], np.float32))
        used.step()
    w, bias = used.weights(b, 0).copy(), used.bias(b).copy()
    used.reset_traces(b)
    assert np.array_equal(used.weights(b, 0), w) and np.array_equal(used.bias(b), bias)
    for net, layer in ((fresh, a), (used, b)):
        w0 = net.weights(layer, 0).copy()
        net.set_inputs(np.array([0.0, 1.0], np.float32))
        net.step()
        net.apply_error(np.array([1.0], np.float32), 0.1)
        dw = net.weights(layer, 0) - w0
        assert dw[0] == 0.0 and dw[1] > 0.0, dw
