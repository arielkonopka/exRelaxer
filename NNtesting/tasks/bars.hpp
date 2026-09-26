#pragma once
// Oriented bars: a bright bar at a random position on a noisy dark
// background, vertical or horizontal. A small vision task with a known
// answer, for orientation-selective features (e.g. a Gabor bank).
#include <cstddef>
#include <random>
#include <vector>

namespace bars {

struct BarImages
{
    size_t side = 24;      // images are side x side
    size_t length = 12;    // bar length in pixels
    size_t width = 2;      // bar width in pixels
    float background = 0.1f;
    float foreground = 0.9f;
    float noise = 0.1f;    // uniform +-noise on every pixel

    // Draws one image into `image` (side * side values, row after row) and
    // returns its label: +1 vertical, -1 horizontal.
    float draw(std::mt19937& g, std::vector<float>& image) const
    {
        std::uniform_real_distribution<float> jitter(-noise, noise);
        const size_t margin = length / 2 - 2;
        std::uniform_int_distribution<size_t> pos(margin, side - margin - 1), coin(0, 1);
        image.resize(side * side);
        for (float& v : image) v = background + jitter(g);
        const bool vertical = coin(g) == 1;
        const size_t cy = pos(g), cx = pos(g);
        for (size_t along = 0; along < length; ++along)
            for (size_t across = 0; across < width; ++across) {
                // Wraps below 0 like any unsigned value: rejected by the bound check.
                const size_t y = vertical ? cy - length / 2 + along : cy + across;
                const size_t x = vertical ? cx + across : cx - length / 2 + along;
                if (y < side && x < side)
                    image[y * side + x] = foreground + jitter(g);
            }
        return vertical ? 1.0f : -1.0f;
    }
};

} // namespace bars
