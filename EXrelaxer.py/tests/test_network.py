"""Tests of the Python bindings: building, running, learning, inspection, save/load."""
import numpy as np
import pytest

import exrelaxer as exr


def small_net(seed=1):
    """in (4) -> hidden (8) -> out (2), with 3 sensors and feedback out -> hidden."""
    exr.reseed(seed)
    net = exr.Network()
    inp = net.add_layer("in", exr.LayerSpec.dense(4))
    hid = net.add_layer("hid", exr.LayerSpec.dense(8))
    out = net.add_layer("out", exr.LayerSpec.dense(2, er=False))
    net.add_inputs(inp, 3)
    net.connect(inp, hid)
    net.connect(hid, out)
    net.add_feedback(out, hid, 2)
    net.add_output(out)
    return net


def stimulus(ticks, width, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (ticks, width)).astype(np.float32)


def test_building_and_inspection():
    net = small_net()
    assert net.layer_count == 3
    assert net.input_count == 3
    assert net.find_layer("hid") == 1
    assert net.layer_name(2) == "out"
    assert net.layer_size(1) == 10  # 8 + 2 feedback neurons
    assert net.layer_type(0) == exr.LayerType.Dense
    assert net.layer_shape(1) == exr.Shape(10, 1, 1)
    assert net.output_layers == [2]
    assert net.update_order == [0, 1, 2]
    kinds = [(e.source, e.target, e.kind) for e in net.edges]
    assert kinds == [(0, 1, exr.EdgeKind.Forward), (1, 2, exr.EdgeKind.Forward), (2, 1, exr.EdgeKind.Feedback)]
    assert not net.layer_spec(2).has_er
    assert "hid" in net.describe()


def test_same_seed_same_results():
    x = stimulus(200, 3)
    assert np.array_equal(small_net(5).run(x), small_net(5).run(x))
    assert not np.array_equal(small_net(5).run(x), small_net(6).run(x))


def test_run_matches_a_step_loop():
    x = stimulus(300, 3)
    rewards = np.where(np.arange(300) % 3 == 0, 1.0, -1.0).astype(np.float32)

    batch = small_net()
    y_batch = batch.run(x, rewards=rewards, learning_rate=0.01)

    loop = small_net()
    y_loop = []
    for t in range(len(x)):
        loop.set_inputs(x[t])
        loop.step()
        y_loop.append(loop.outputs())
        loop.apply_reward(float(rewards[t]), 0.01)

    assert y_batch.shape == (300, 2)
    assert y_batch.dtype == np.float32
    assert np.array_equal(y_batch, np.array(y_loop))
    assert np.array_equal(batch.weights(1, 0), loop.weights(1, 0))


def test_inputs_accept_lists_and_other_dtypes():
    net = small_net()
    net.set_inputs([0.5, -0.25, 1.0])
    assert net.inputs.tolist() == [0.5, -0.25, 1.0]
    net.set_inputs(np.array([1.0, 2.0, 3.0]))  # float64 is converted
    assert net.inputs.tolist() == [1.0, 2.0, 3.0]
    net.set_input(0, 7.0)
    assert net.inputs[0] == 7.0


def test_errors_become_python_exceptions():
    net = small_net()
    with pytest.raises(ValueError, match="expected 3 input values"):
        net.set_inputs(np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="ticks x 3"):
        net.run(np.zeros((5, 2), np.float32))
    with pytest.raises(ValueError, match="one value per tick"):
        net.run(np.zeros((5, 3), np.float32), rewards=np.zeros(4, np.float32))
    with pytest.raises(IndexError):
        net.find_layer("missing")
    with pytest.raises(IndexError):
        net.set_input(3, 0.0)
    with pytest.raises(ValueError, match="duplicate layer name"):
        net.add_layer("in", exr.LayerSpec.dense(1))
    with pytest.raises(ValueError, match="not a Conv2D layer"):
        net.kernel(1, 0)


def test_weights_can_be_read_and_set():
    net = small_net()
    w = net.weights(2, 0)
    assert w.shape == (10,)  # out reads all 10 hidden neurons
    net.set_weights(2, 0, np.arange(10, dtype=np.float32))
    assert net.weights(2, 0).tolist() == list(range(10))
    net.set_weights(2, 0, [0.0] * 10)
    assert not net.weights(2, 0).any()
    with pytest.raises(ValueError):
        net.set_weights(2, 0, [1.0])


def test_freezing_stops_learning():
    net = small_net()
    net.freeze(1)
    assert net.is_frozen(1)
    before = net.weights(1, 0)
    net.run(stimulus(100, 3), rewards=np.ones(100, np.float32), learning_rate=0.05)
    assert np.array_equal(net.weights(1, 0), before)
    net.unfreeze(1)
    net.run(stimulus(100, 3, seed=1), rewards=np.ones(100, np.float32), learning_rate=0.05)
    assert not np.array_equal(net.weights(1, 0), before)


def test_neuron_state_and_jitter():
    exr.reseed(3)
    net = exr.Network()
    layer = net.add_layer("res", exr.LayerSpec.dense(50, recovery_jitter=exr.Jitter.normal(0.02).around(0.95)))
    state = net.neuron_state(layer)
    assert set(state) == {"threshold", "recovery", "learning_gain", "alpha"}
    assert state["recovery"].shape == (50,)
    assert state["recovery"].std() > 0
    assert abs(state["recovery"].mean() - 0.95) < 0.02
    assert np.all(state["learning_gain"] == exr.constants.default_learning_gain)
    net.set_learning_jitter(layer, exr.Jitter.uniform_relative())
    gains = net.neuron_state(layer)["learning_gain"]
    assert gains.min() >= 1.0 and gains.max() <= 3.0 and gains.std() > 0


def test_reset_state_returns_er_neurons_to_rest():
    exr.reseed(4)
    net = exr.Network()
    layer = net.add_layer("h", exr.LayerSpec.dense(20, False, True, frozen=True,
                                                   recovery_jitter=exr.Jitter.normal(0.02).around(0.95)))
    net.add_inputs(layer, 5, "x")
    net.add_output(layer)
    weights = [np.asarray(net.weights(layer, i)) for i in range(20)]
    recovery = net.neuron_state(layer)["recovery"]
    net.run(np.ones((3, 5), np.float32) * 3.0)
    assert np.any(net.neuron_state(layer)["threshold"] != exr.constants.baseline_threshold)
    net.reset_state(layer)
    state = net.neuron_state(layer)
    assert np.allclose(state["threshold"], exr.constants.baseline_threshold)
    assert not np.any(net.layer_output(layer))
    assert np.array_equal(state["recovery"], recovery)
    assert all(np.array_equal(np.asarray(net.weights(layer, i)), weights[i]) for i in range(20))


@pytest.mark.parametrize("mode", [exr.DeserializeMode.FullState, exr.DeserializeMode.WeightsOnly])
def test_save_and_load(tmp_path, mode):
    net = small_net()
    net.run(stimulus(50, 3), rewards=np.ones(50, np.float32), learning_rate=0.01)
    later = stimulus(100, 3, seed=9)

    restored = exr.Network.from_bytes(net.to_bytes(), mode)
    path = tmp_path / "net.bin"
    net.save(path)
    from_file = exr.Network.load(path, mode)

    for copy in (restored, from_file):
        assert copy.layer_count == 3
        assert copy.describe() == net.describe()
        assert np.array_equal(copy.weights(1, 3), net.weights(1, 3))

    expected = net.run(later)
    if mode == exr.DeserializeMode.FullState:  # continues bit for bit
        assert np.array_equal(restored.run(later), expected)
        assert np.array_equal(from_file.run(later), expected)


def test_load_rejects_bad_data():
    with pytest.raises(RuntimeError, match="network::load"):
        exr.Network.from_bytes(b"not a network")
