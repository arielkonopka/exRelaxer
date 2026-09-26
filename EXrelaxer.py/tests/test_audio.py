"""Audio layers (Cochlea, History) from Python, and the exrelaxer.audio helper."""
import wave

import numpy as np
import pytest

import exrelaxer as exr
from exrelaxer import audio


def small_spec(**changes):
    spec = exr.CochleaSpec(sample_rate=8000, hop=64, window=256, bands=16, min_frequency=100,
                           compression=exr.Compression.Linear, gain=1.0)
    for key, value in changes.items():
        setattr(spec, key, value)
    return spec


def ear_network(spec, length=None, **kwargs):
    net = exr.Network()
    ear = net.add_layer("ear", exr.LayerSpec.cochlea(spec, False, False, **kwargs))
    net.add_inputs(ear, spec.hop)
    top = ear
    if length:
        top = net.add_layer("past", exr.LayerSpec.history(length))
        net.connect(ear, top)
    net.add_output(top)
    return net, ear, top


def test_cochlea_hears_a_tone_in_its_band():
    spec = small_spec()
    for frequency in (300.0, 1200.0, 3000.0):
        net, ear, _ = ear_network(spec)
        ys = net.run(audio.frames(audio.tone(frequency, 0.2, spec.sample_rate), spec.hop))
        low, centre, high = net.cochlea_bands(ear)[int(np.argmax(ys[-1]))]
        assert low <= frequency <= high
    power = net.cochlea_power(ear)
    assert power.shape == (spec.window // 2 + 1,)


def test_cochlea_matches_numpy_fft():
    spec = small_spec()
    net, ear, _ = ear_network(spec)
    rng = np.random.default_rng(3)
    sound = rng.uniform(-0.5, 0.5, 4 * spec.hop).astype(np.float32)
    net.run(audio.frames(sound, spec.hop))
    window = np.hanning(spec.window + 1)[:-1]  # periodic Hann, as the layer uses
    spectrum = np.fft.rfft(sound[-spec.window:] * window)
    expected = np.abs(spectrum) ** 2 * (2 / window.sum()) ** 2
    assert np.allclose(net.cochlea_power(ear), expected, rtol=1e-4, atol=1e-5)


def test_history_makes_a_spectrogram_that_survives_save_and_load():
    spec = small_spec(compression=exr.Compression.Log, gain=50.0)
    exr.reseed(4)
    net, ear, past = ear_network(spec, length=10)
    assert net.layer_output(past).shape == (1, 16, 10)
    sweep = audio.frames(audio.chirp(200, 3000, 0.4, spec.sample_rate), spec.hop)
    net.run(sweep[:20])
    data = net.to_bytes()
    original = net.run(sweep[20:])
    again = exr.Network.from_bytes(data).run(sweep[20:])
    assert np.array_equal(original, again)
    assert "1x16x10" in net.describe()


def test_bad_specs_raise():
    net = exr.Network()
    with pytest.raises(ValueError):
        net.add_layer("x", exr.LayerSpec.cochlea(small_spec(window=300)))
    ear = net.add_layer("ear", exr.LayerSpec.cochlea(small_spec()))
    with pytest.raises(ValueError):
        net.add_inputs(ear, 10)
    with pytest.raises(ValueError):
        net.add_layer("h", exr.LayerSpec.history(0))


def test_frames_and_signals():
    f = audio.frames(np.arange(10), 4)
    assert f.shape == (3, 4) and f.dtype == np.float32
    assert f[2].tolist() == [8, 9, 0, 0]
    assert len(audio.tone(440, 0.5, 8000)) == 4000
    up = audio.chirp(100, 1000, 1.0, 8000)
    assert np.max(np.abs(up)) <= 0.5 + 1e-6
    assert len(audio.resample(np.zeros(8000), 8000, 16000)) == 16000


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_read_wav(tmp_path, width):
    rate, signal = 8000, audio.tone(500, 0.1, 8000, amplitude=0.5)
    path = tmp_path / "t.wav"
    scale = float(1 << (8 * width - 1))
    ints = np.round(signal * (scale - 1)).astype(np.int64)
    if width == 1:
        raw = (ints + 128).astype(np.uint8).tobytes()
    elif width == 3:
        raw = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in ints)
    else:
        raw = ints.astype(f"<i{width}").tobytes()
    stereo = np.frombuffer(raw, np.uint8).reshape(len(signal), width)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(np.concatenate([stereo, stereo], axis=1).tobytes())
    data, got_rate = audio.read_wav(path)
    assert got_rate == rate and len(data) == len(signal)
    assert np.allclose(data, signal, atol=2.0 / (1 << (8 * width - 1)) + 1e-6)
    resampled, new_rate = audio.read_wav(path, sample_rate=16000)
    assert new_rate == 16000 and len(resampled) == 2 * len(signal)
