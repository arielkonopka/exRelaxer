// Shared by the audio layer types: the Cochlea's spec (how sound becomes
// frequency bands).
#pragma once
#include <cstddef>
#include <cstdint>

namespace exr {

enum class FrequencyScale : std::uint8_t {
    Mel = 0,    // bands evenly spaced in mel: narrow at low frequencies, like the cochlea
    Linear = 1  // bands evenly spaced in Hz
};

enum class Compression : std::uint8_t {
    Log = 0,    // log(1 + gain * energy): loudness-like, 0 for silence
    Linear = 1  // gain * energy (no compression)
};

// A Cochlea reads `hop` new audio samples per tick from each of `channels`
// microphones and keeps the last `window` samples of each. Each tick it takes their spectrum (Hann window, FFT) and
// sums the power into `bands` triangular bands between minFrequency and
// maxFrequency.
struct CochleaSpec
{
    float sampleRate = 16000.0f;  // samples per second
    size_t hop = 160;             // new samples per tick (the sensor count): 10 ms at 16 kHz
    size_t window = 512;          // samples analysed per tick, a power of two >= hop
    size_t bands = 40;            // frequency bands (neurons), lowest first
    float minFrequency = 50.0f;   // Hz, lower edge of the lowest band
    float maxFrequency = 0.0f;    // Hz, upper edge of the highest band; 0 = sampleRate / 2
    FrequencyScale scale = FrequencyScale::Mel;
    Compression compression = Compression::Log;
    float gain = 100.0f;          // band energy multiplier before compression
    size_t channels = 1;          // microphones (e.g. 2 for stereo): hop sensors each, channel after channel

    bool operator==(const CochleaSpec&) const = default;
};

} // namespace exr
