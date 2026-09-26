"""Sound for a Cochlea layer: reading WAV files, test signals, and cutting a
signal into the hop-sized frames a network reads one per tick.

    spec = exr.CochleaSpec(sample_rate=16000, hop=160, bands=40)
    ear = net.add_layer("ear", exr.LayerSpec.cochlea(spec))
    net.add_inputs(ear, spec.hop)
    ...
    sound, rate = audio.read_wav("speech.wav", sample_rate=spec.sample_rate)
    ys = net.run(audio.frames(sound, spec.hop))       # one row per tick
"""
import wave

import numpy as np

__all__ = ["chirp", "frames", "read_wav", "resample", "tone"]


def frames(signal, hop):
    """`signal` as ticks x hop float32 frames; a partial last frame is zero-padded."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if hop < 1:
        raise ValueError("hop must be at least 1")
    count = -(-len(signal) // hop)
    out = np.zeros((count, hop), dtype=np.float32)
    out.reshape(-1)[:len(signal)] = signal
    return out


def resample(signal, rate, new_rate):
    """Linear-interpolation resampling (enough for experiments, not for hi-fi)."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if rate == new_rate or len(signal) == 0:
        return signal
    count = int(round(len(signal) * new_rate / rate))
    t = np.arange(count, dtype=np.float64) * (rate / new_rate)
    return np.interp(t, np.arange(len(signal)), signal).astype(np.float32)


def read_wav(path, sample_rate=None):
    """A PCM WAV file (8, 16, 24 or 32 bit) as mono float32 in [-1, 1] and its
    sample rate; channels are averaged. With `sample_rate`, resampled to it."""
    with wave.open(str(path), "rb") as f:
        channels, width, rate = f.getnchannels(), f.getsampwidth(), f.getframerate()
        raw = f.readframes(f.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        data = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    data = data.reshape(-1, channels).mean(axis=1)
    if sample_rate is not None and sample_rate != rate:
        return resample(data, rate, sample_rate), sample_rate
    return data.astype(np.float32), rate


def tone(frequency, seconds, sample_rate=16000, amplitude=0.5, phase=0.0):
    """A sine tone."""
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t + phase)).astype(np.float32)


def chirp(start, end, seconds, sample_rate=16000, amplitude=0.5, phase=0.0, logarithmic=True):
    """A tone gliding from `start` to `end` Hz: exponentially (logarithmic,
    even on a mel-like scale) or linearly."""
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float64) / sample_rate
    if logarithmic and start != end:
        k = np.log(end / start) / seconds
        angle = 2 * np.pi * start * (np.exp(k * t) - 1) / k
    else:
        angle = 2 * np.pi * (start * t + (end - start) * t * t / (2 * seconds))
    return (amplitude * np.sin(angle + phase)).astype(np.float32)
