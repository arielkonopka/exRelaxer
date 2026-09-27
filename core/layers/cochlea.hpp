// The entry point for sound: reads `hop` new audio samples per tick and has
// one neuron per frequency band, lowest band first. With habituation and E-R
// on, each band adapts like a group of hair cells (a steady tone fades,
// onsets and changes stand out); with both off it passes the band values
// through.
//
// Each tick (CochleaSpec):
//   1. the last `window` samples: the previous ones shifted by `hop`, then
//      this tick's sensors;
//   2. times a Hann window, then an FFT; the power of each frequency bin is
//      normalised so a sine of amplitude A centred on a bin has power A^2;
//   3. each band sums the power under its triangular filter (peak 1), the
//      bands evenly spaced in mel or in Hz between minFrequency and
//      maxFrequency, each overlapping half of its neighbours;
//   4. compression: log(1 + gain * energy) (Log) or gain * energy (Linear),
//      the neuron's weighted sum.
//
// Several microphones (CochleaSpec::channels) are analysed independently, one
// output channel each, with the same bands.
//
// Output shape: channels x bands x 1 (frequency as the height), so a History
// layer turns it into a channels x bands x ticks spectrogram for spatial layers. The filters
// are fixed: the cochlea does not learn.
#pragma once
#include <optional>
#include <vector>
#include "audio.hpp"
#include "neuron_layer.hpp"

namespace exr {

class cochlea final : public neuron_layer
{
public:
    // Throws std::invalid_argument for an invalid spec (see CochleaSpec).
    explicit cochlea(const CochleaSpec& spec, bool hasHabituation = true, bool hasER = true,
                     const Jitter& recoveryJitter = {}, const Jitter& learningJitter = {},
                     const Jitter& alphaJitter = {});

    LayerType type() const override { return LayerType::Cochlea; }
    Shape shape() const override { return {spec_.channels, spec_.bands, 1}; }
    const CochleaSpec& spec() const { return spec_; }

    // A band's triangular filter over the FFT bins (bin k is k * sampleRate /
    // window Hz): weights for bins firstBin, firstBin + 1, ...
    struct Band
    {
        float lowFrequency;     // Hz, where the triangle starts
        float centreFrequency;  // Hz, its peak
        float highFrequency;    // Hz, where it ends
        size_t firstBin;
        std::vector<float> weights;
    };
    const std::vector<Band>& bands() const { return bands_; }

    // The last `window` samples of each channel, oldest first, channel after
    // channel (the analysis windows' input).
    const std::vector<float>& samples() const { return samples_; }
    // The power spectrum of the last forward(): window / 2 + 1 bins per
    // channel, channel after channel.
    const std::vector<float>& power() const { return power_; }

    // The sound: exactly channels * hop sensors (hop per channel, channel
    // after channel), read once per tick. Once; the cochlea reads nothing else.
    void attachInputs(const InputRange& sensors) override;
    void forward() override;

    // The neurons, then the sample window (FullState restores it,
    // WeightsOnly zeroes it), so a restored network continues exactly.
    void serialize(std::ostream& os) const override;
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

protected:
    void copyWeights(size_t, std::vector<float>& out) const override { out.clear(); }
    size_t expectedWeights(size_t) const override { return 0; }
    void storeWeights(size_t, std::span<const float>) override {}
    // The band count is fixed by the spec: never rebuilt with another count.
    bool wired() const override { return true; }

private:
    void fft();  // in place on re_, im_

    CochleaSpec spec_;
    std::vector<Band> bands_;
    std::vector<float> hann_;
    float power_scale_;             // (2 / sum of the Hann window)^2
    std::vector<size_t> reversed_;  // bit-reversal permutation of the window
    std::vector<float> cos_, sin_;  // twiddle factors, window / 2 of each
    std::vector<float> samples_, re_, im_, power_;
    std::optional<InputRange> sensors_;
};

} // namespace exr
