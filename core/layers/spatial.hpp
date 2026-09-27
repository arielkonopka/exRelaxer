// Shared by the spatial layer types (Conv2D, LocallyConnected2D, Pool2D,
// Retina): window geometry, their specs, and the input channels a spatial
// layer reads.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
#include "layer.hpp"

namespace exr {

// Resize2D: every input channel resampled to height x width.
enum class Interpolation : std::uint8_t {
    Nearest = 0,   // the input pixel under the output pixel's centre
    Bilinear = 1,  // weighted by distance from the four nearest input pixels (pixel centres aligned)
    Area = 2       // the mean of the input pixels the output pixel covers (for shrinking; bilinear when enlarging)
};

struct ResizeSpec
{
    size_t height = 0;  // output height and width
    size_t width = 0;
    Interpolation interpolation = Interpolation::Bilinear;
    bool operator==(const ResizeSpec&) const = default;
};

// Disparity: how well a left and a right view match at each horizontal shift.
enum class DisparityMeasure : std::uint8_t {
    Correlation = 0,  // mean of left * right: high where both are active together
    Difference = 1,   // mean of |left - right|: 0 where they match
    Normalized = 2    // normalized cross-correlation over the window, in [-1, 1] (0 where either is flat)
};

// Output channel d compares left pixel x with right pixel x - (minDisparity + d),
// averaged over a window x window box and every channel. Near objects appear
// further left in the right view, so their disparity is larger.
struct DisparitySpec
{
    int minDisparity = 0;
    int maxDisparity = 4;   // inclusive: maxDisparity - minDisparity + 1 output channels
    size_t window = 3;      // odd box size, in pixels
    DisparityMeasure measure = DisparityMeasure::Correlation;

    size_t count() const { return static_cast<size_t>(maxDisparity - minDisparity + 1); }
    bool operator==(const DisparitySpec&) const = default;
};

// A sliding window over height x width, with zero padding on each side.
// Output height = (height + 2 * padY - kernelHeight) / strideY + 1, likewise
// for the width; 0 when the kernel does not fit.
struct Window2D
{
    size_t kernelHeight = 1;
    size_t kernelWidth = 1;
    size_t strideY = 1;
    size_t strideX = 1;
    size_t padY = 0;
    size_t padX = 0;

    static constexpr Window2D square(size_t kernel, size_t stride = 1, size_t padding = 0)
    {
        return {kernel, kernel, stride, stride, padding, padding};
    }
    constexpr size_t area() const { return kernelHeight * kernelWidth; }
    size_t outputHeight(size_t height) const;
    size_t outputWidth(size_t width) const;
    bool operator==(const Window2D&) const = default;
};

enum class PoolMode : std::uint8_t { Max = 0, Average = 1 };

enum class Sampling : std::uint8_t {
    Grid = 0,   // one sample per pixel, in image order: output channels x height x width
    Spiral = 1  // sample 0 at the image centre, then outwards on a tight spiral: output channels x 1 x samples
};

struct RetinaSpec
{
    Shape input{};                       // the image: channels x height x width
    Sampling sampling = Sampling::Grid;
    float spacing = 1.0f;                // Spiral: pixels between samples along the spiral and between turns
    float radius = 0.0f;                 // Spiral: outermost radius in pixels; 0 = the largest circle inside the image
    bool operator==(const RetinaSpec&) const = default;
};

// The input channels of a spatial layer: every channel of each source it
// joined, in join order, then channels added by the sources' growth. All
// sources have the same height x width (a flat layer is n x 1 x 1: n
// channels of one pixel).
class spatial_inputs
{
public:
    bool empty() const { return sources_.empty(); }
    size_t channels() const { return channels_.size(); }
    size_t height() const { return height_; }
    size_t width() const { return width_; }
    bool reads(const layer& source) const;

    // Appends every channel of `source` and returns how many. Throws
    // std::invalid_argument if its height x width differs from earlier sources'.
    size_t add(const layer& source);
    // `source` grew by `count` outputs from `offset`: appends them as new
    // channels and returns how many. Throws std::logic_error unless they are
    // whole channels.
    size_t grew(const layer& source, size_t offset, size_t count);

    // Copies every channel, in order, into `tensor` (channels x height x width).
    void gather(std::vector<float>& tensor) const;

private:
    struct Channel
    {
        std::reference_wrapper<const layer> source;
        size_t offset;  // first output of this channel in the source
    };

    std::vector<Channel> channels_;
    std::vector<std::reference_wrapper<const layer>> sources_;
    size_t height_ = 0;
    size_t width_ = 0;
};

// The window of output position (oy, ox) over `tensor` (channels x height x
// width), channel by channel, row by row, zero outside the input.
// `out` has channels * window.area() entries.
void gatherWindow(std::span<const float> tensor, const Shape& shape, const Window2D& window, size_t oy,
                  size_t ox, std::span<float> out);

} // namespace exr
