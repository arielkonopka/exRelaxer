"""Neuron models other than E-R (NeuronModel): building, running, probing, saving."""
import numpy as np
import pytest

import exrelaxer as exr

MODELS = [
    exr.NeuronModel.sigmoid(),
    exr.NeuronModel.tanh(2.0, smoothing=0.3),
    exr.NeuronModel.softplus(),
    exr.NeuronModel.lif(leak=0.8, threshold=0.5),
    exr.NeuronModel.alif(adaptation=0.3),
    exr.NeuronModel.chattering(),
    exr.NeuronModel.resonator(frequency=0.15, damping=0.9, threshold=0.5),
]


def run(net, ticks, t0=0):
    out = []
    for t in range(t0, t0 + ticks):
        net.set_inputs(np.array([np.sin(0.3 * t), np.cos(0.7 * t)], dtype=np.float32))
        net.step()
        out.append(net.layer_output(0).copy())
    return np.array(out)


def build(model):
    exr.reseed(3)
    net = exr.Network()
    h = net.add_layer("h", exr.LayerSpec.dense(8, neuron_model=model))
    net.add_inputs(h, 2)
    net.add_output(h)
    return net


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_models_run_save_and_continue(model):
    net = build(model)
    spec = net.layer_spec(0)
    assert spec.neuron_model == model
    assert not spec.has_er
    y = run(net, 30)
    assert np.isfinite(y).all()
    if model.spiking:
        assert set(np.unique(y)) <= {0.0, 1.0}
    restored = exr.Network.from_bytes(net.to_bytes(), exr.DeserializeMode.FullState)
    assert restored.layer_spec(0).neuron_model == model
    np.testing.assert_array_equal(run(restored, 40, 30), run(net, 40, 30))


def test_state_probe_reports_membrane_and_charge():
    net = build(exr.NeuronModel.lif(leak=0.9, threshold=10.0))
    run(net, 3)
    probe = net.state_probe(0)
    assert np.all(probe["charge"] == pytest.approx(probe["membrane"] / 10.0))
    assert "recovery_variable" in probe


def test_invalid_combinations_raise():
    spec = exr.LayerSpec.dense(4)
    spec.neuron_model = exr.NeuronModel.lif()  # still has E-R
    with pytest.raises(ValueError):
        exr.Network().add_layer("x", spec)
    with pytest.raises(ValueError):
        exr.Network().add_layer("y", exr.LayerSpec.dense(4, neuron_model=exr.NeuronModel.lif(leak=1.5)))


def test_tanh_layer_learns_with_feedback_alignment():
    exr.reseed(5)
    net = exr.Network()
    spec = exr.LayerSpec.dense(8, neuron_model=exr.NeuronModel.tanh(),
                               learning_rule=exr.LearningRule.feedback_alignment())
    h = net.add_layer("h", spec)
    o = net.add_layer("o", exr.LayerSpec.dense(1, neuron_model=exr.NeuronModel.tanh(),
                                               learning_rule=exr.LearningRule.feedback_alignment()))
    net.add_inputs(h, 2)
    net.connect(h, o)
    net.add_output(o)
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(4000):
        x = rng.uniform(-1, 1, 2).astype(np.float32)
        net.set_inputs(x)
        net.step()
        e = np.tanh(0.8 * x[0] - 0.5 * x[1]) - net.outputs()[0]
        errors.append(e * e)
        net.apply_error(np.array([e], dtype=np.float32), 0.02)
    assert np.mean(errors[-500:]) < 0.3 * np.mean(errors[:500])
