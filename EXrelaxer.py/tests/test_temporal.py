"""State probes (Network.state_probe, last_inputs, input_trace) and the
determinism the controlled benchmarks rely on (doc/model.md, doc/protocol.md)."""
import numpy as np

import exrelaxer as exr


def small_net(rule=None):
    exr.reseed(11)
    net = exr.Network()
    spec = exr.LayerSpec.dense(8, False, True)
    if rule is not None:
        spec.learning_rule = rule
    h = net.add_layer("h", spec)
    net.add_inputs(h, 3, "x")
    net.connect(h, h)
    net.add_output(h)
    return net, h


def drive(net, ticks=30, seed=0, lr=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(ticks):
        net.set_inputs("x", rng.uniform(-1, 1, 3).astype(np.float32))
        net.step()
        out.append(np.asarray(net.outputs()).copy())
        if lr:
            net.apply_reward(float(rng.choice([-1.0, 1.0])), lr)
    return np.array(out)


def test_state_probe_fields_and_values():
    net, h = small_net(exr.LearningRule.traced(0.5, 0.0))
    drive(net, 5)
    s = net.state_probe(h)
    for key in ("output", "threshold", "resting_threshold", "eligibility", "output_trace",
                "habituation_streak", "previous_input"):
        assert s[key].shape == (8,), key
    np.testing.assert_array_equal(s["output"], net.outputs())
    np.testing.assert_array_equal(s["threshold"], net.neuron_state(h)["threshold"])
    assert np.all(s["resting_threshold"] == np.float32(exr.constants.baseline_threshold))
    above = s["threshold"] > s["resting_threshold"]
    np.testing.assert_allclose(s["eligibility"][above], s["threshold"][above] / s["resting_threshold"][above] - 1,
                               rtol=1e-6)
    assert np.all(s["eligibility"][~above] == 0)
    # last_inputs: the sensors, then the layer's own previous outputs (self-connection: t-1).
    assert net.last_inputs(h, 0).shape == (3 + 8,)
    assert net.input_trace(h, 0).shape == (3 + 8,)


def test_probing_does_not_change_the_run():
    net_a, h = small_net()
    net_b, _ = small_net()
    rng = np.random.default_rng(1)
    for _ in range(40):
        x = rng.uniform(-1, 1, 3).astype(np.float32)
        for net in (net_a, net_b):
            net.set_inputs("x", x)
            net.step()
        net_a.state_probe(h)
        net_a.last_inputs(h, 3)
        np.testing.assert_array_equal(net_a.outputs(), net_b.outputs())


def test_same_seed_same_result_and_thread_count_does_not_matter():
    results = []
    for threads in (1, 4, 1):
        exr.set_threads(threads)
        net, _ = small_net()
        results.append(drive(net, 50, lr=0.01))
    exr.set_threads(1)
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


def test_build_info_names_the_kernel_variant():
    assert "simd" in exr.build_info()


def test_state_layer_outputs_threshold_above_rest_and_habituation():
    exr.reseed(3)
    net = exr.Network()
    spec = exr.LayerSpec.dense(5, True, True)
    spec.habituation_rule = exr.Habituation(steps=4)
    h = net.add_layer("h", spec)
    tap = net.add_layer("h_state", exr.LayerSpec.state())
    net.add_inputs(h, 3, "x")
    net.connect(h, tap)
    net.add_output(h)
    net.add_output(tap)
    x = np.array([1.0, -0.5, 0.25], np.float32)
    for _ in range(8):  # the same input: streaks run past the onset
        net.set_inputs("x", x)
        net.step()
        s = net.state_probe(h)
        out = np.asarray(net.outputs())
        np.testing.assert_array_equal(out[:5], s["output"])
        np.testing.assert_allclose(out[5::2], s["threshold"] - s["resting_threshold"], rtol=0, atol=0)
        np.testing.assert_array_equal(out[6::2], np.minimum(s["habituation_streak"] / 4.0, 1.0).astype(np.float32))
    assert np.all(out[6::2] == 1.0)
