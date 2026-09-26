// Classic fixed receptive fields for Conv2D layers: Gaussian blur,
// centre-surround (difference of Gaussians, like retinal ganglion cells) and
// Gabor filters (oriented stripes and edges, like V1 simple cells).
//
// Learning is weak beyond the last layer, so a good way to see is a frozen
// Conv2D with a filter bank, then pooling, then learned layers:
//
//   auto v1 = net.addLayer("v1", LayerSpec::Conv2D(8, Window2D::square(7, 1, 3), false, false));
//   net.connect(eye, v1);
//   filters::load(net.layerAs<conv2d>(v1), filters::gaborBank(7, 4, 4.0f, 2.0f));
//   net.freeze(v1);
//
// Scale: DoG and Gabor filters have zero mean, so a uniform image gives 0,
// and are scaled so the best-matching full-contrast pattern (pixels 0 where
// the filter is negative, 1 where it is positive) gives exactly `gain`.
#pragma once
#include <cstddef>
#include <initializer_list>
#include <numbers>
#include <span>
#include <vector>

namespace exr {

class conv2d;

namespace filters {

// One single-channel filter: height x width weights, row after row.
struct Filter
{
    size_t height = 0;
    size_t width = 0;
    std::vector<float> weights;

    float at(size_t y, size_t x) const { return weights[y * width + x]; }
};

enum class Polarity { OnCentre, OffCentre };

// size x size Gaussian blur, weights summing to `gain`.
Filter gaussian(size_t size, float sigma, float gain = 1.0f);

// Centre-surround: a narrow Gaussian minus a wide one. OnCentre responds to
// a bright spot on a dark surround, OffCentre to the opposite.
Filter differenceOfGaussians(size_t size, float centreSigma, float surroundSigma,
                             Polarity polarity = Polarity::OnCentre, float gain = 1.0f);

// Gabor: a sinusoid of `wavelength` pixels under a Gaussian envelope.
// `orientation` (radians) is the direction of the preferred stripes and
// edges, measured from the x axis (x right, y down): 0 horizontal, pi/2
// vertical. `phase` 0 prefers a bright bar on the centre line, pi/2 an edge.
// `aspect` < 1 stretches the envelope along the stripes.
Filter gabor(size_t size, float orientation, float wavelength, float sigma, float phase = 0.0f,
             float aspect = 1.0f, float gain = 1.0f);

// `orientations` evenly spaced orientations in [0, pi), each with every
// phase, orientation-major: {o0 p0, o0 p1, o1 p0, ...}.
std::vector<Filter> gaborBank(size_t size, size_t orientations, float wavelength, float sigma,
                              std::initializer_list<float> phases = {0.0f, std::numbers::pi_v<float> / 2},
                              float aspect = 1.0f, float gain = 1.0f);

// {OnCentre, OffCentre} difference of Gaussians.
std::vector<Filter> centreSurroundBank(size_t size, float centreSigma, float surroundSigma, float gain = 1.0f);

// Output channel k of `layer` gets bank[k], on every input channel (divided
// by their count, so a grey image gives the same response whatever the
// channel count). The bank must have one filter per output channel, each of
// the layer's window size; the layer must be wired. Throws
// std::invalid_argument otherwise.
void load(conv2d& layer, std::span<const Filter> bank);

// Output channel `channel` gets `filter` on input channel `input` only.
void load(conv2d& layer, size_t channel, const Filter& filter, size_t input);

} // namespace filters
} // namespace exr
