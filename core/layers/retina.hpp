// The entry point for images: reads sensors holding a channels x height x
// width image and has one neuron per sample and channel. With habituation
// and E-R on, each sample adapts like a photoreceptor (a static image fades,
// changes stand out); with both off it passes values through (Input2D).
//
// Sampling (RetinaSpec):
//   Grid    one sample per pixel, output shape = the image shape.
//   Spiral  sample 0 at the image centre, then outwards along an Archimedean
//           spiral r = spacing * theta / (2 pi): turns are `spacing` pixels
//           apart and consecutive samples are `spacing` pixels apart along
//           the curve, so samples cover the disc about evenly, in order of
//           distance from the centre. Each sample interpolates its 4 nearest
//           pixels (bilinear). Output shape: channels x 1 x samples.
//
// Sample weights are fixed: the retina does not learn.
#pragma once
#include <cstdint>
#include <optional>
#include <vector>
#include "neuron_layer.hpp"
#include "spatial.hpp"

namespace exr {

class retina final : public neuron_layer
{
public:
    // Throws std::invalid_argument for an empty image or, with Spiral, a
    // spacing <= 0.
    explicit retina(const RetinaSpec& spec, bool hasHabituation = true, bool hasER = true,
                    const Jitter& recoveryJitter = {}, const Jitter& learningJitter = {},
                    const Jitter& alphaJitter = {});

    LayerType type() const override { return LayerType::Retina; }
    Shape shape() const override;
    const RetinaSpec& spec() const { return spec_; }

    // Samples per channel, and where sample k is in the image (pixel
    // coordinates: x to the right, y down, pixel centres at whole numbers).
    struct Point
    {
        double x;
        double y;
    };
    size_t samples() const { return points_.size(); }
    Point samplePoint(size_t k) const { return points_.at(k); }

    // The image: exactly spec().input.size() sensors, channel after channel,
    // row after row. Once; the retina reads nothing else.
    void attachInputs(const InputRange& sensors) override;
    void forward() override;

protected:
    void copyWeights(size_t, std::vector<float>& out) const override { out.clear(); }
    size_t expectedWeights(size_t) const override { return 0; }
    void storeWeights(size_t, std::span<const float>) override {}
    bool wired() const override { return sensors_.has_value(); }

private:
    struct Tap
    {
        std::uint32_t pixel;  // index in one channel of the image
        float weight;
    };

    retina(const RetinaSpec& spec, std::vector<Point> points, bool hasHabituation, bool hasER,
           const Jitter& recoveryJitter, const Jitter& learningJitter, const Jitter& alphaJitter);
    static std::vector<Point> samplePoints(const RetinaSpec& spec);

    RetinaSpec spec_;
    std::vector<Point> points_;
    std::vector<Tap> taps_;
    std::vector<size_t> tap_begin_;  // taps of sample k: [tap_begin_[k], tap_begin_[k + 1])
    std::optional<InputRange> sensors_;
};

} // namespace exr
