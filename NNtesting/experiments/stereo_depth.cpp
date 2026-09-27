// Stereo vision: where is the near square in a random-dot stereogram?
//
// Each eye sees only random dots. A square of dots floats in front of the
// background: in the right eye it is shifted `near` pixels to the left
// (disparity `near`), the background has disparity 0 (Julesz's random-dot
// stereogram). The square is in the left or the right half of the view. With
// one eye the answer is invisible; only matching the two views finds it.
//
//   stereo:   left eye, right eye -> Disparity (0..max, normalized) -> Pool2D average -> dense out (learns)
//   mono:     left eye -> Pool2D average -> dense out (learns)
//   two_eyes: left eye, right eye -> Pool2D average -> dense out (learns): both views, no matching
//
// Both eyes are named input sources ("left_eye", "right_eye"). The readout
// learns error-driven (a reward only for wrong answers), as in chirp_direction.
#include <random>
#include <string>
#include <vector>
#include "experiment.hpp"
#include "network.hpp"

namespace {

using namespace exr;

struct Stereograms
{
    size_t height = 16, width = 32, square = 8;
    int near = 3;         // the square's disparity in pixels
    double flip = 0.0;    // probability that a right-eye dot is redrawn (decorrelates the views)

    // Fills both views (+-1 dots); returns +1 if the square is in the left half, -1 in the right.
    float draw(std::mt19937& g, std::vector<float>& left, std::vector<float>& right) const
    {
        std::uniform_int_distribution<int> coin(0, 1);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        auto dot = [&] { return coin(g) ? 1.0f : -1.0f; };
        left.resize(height * width);
        right.resize(height * width);
        for (float& v : left)
            v = dot();
        right = left;
        const bool in_left = coin(g) == 1;
        const size_t half = width / 2, margin = static_cast<size_t>(near) + 1;
        std::uniform_int_distribution<size_t> x_at(in_left ? margin : half, (in_left ? half : width - margin) - square);
        std::uniform_int_distribution<size_t> y_at(0, height - square);
        const size_t x0 = x_at(g), y0 = y_at(g);
        for (size_t y = y0; y < y0 + square; ++y) {
            // The square's dots in the right eye sit `near` pixels further left;
            // the strip it uncovers on its right gets new dots.
            for (size_t x = x0 + square - static_cast<size_t>(near); x < x0 + square; ++x)
                right[y * width + x] = dot();
            for (size_t x = x0; x < x0 + square; ++x)
                right[y * width + x - static_cast<size_t>(near)] = left[y * width + x];
        }
        if (flip > 0.0)
            for (float& v : right)
                if (unit(g) < flip)
                    v = dot();
        return in_left ? 1.0f : -1.0f;
    }
};

struct Viewer
{
    std::unique_ptr<network> net = std::make_unique<network>();

    Viewer(const nnt::Params& p, const Stereograms& task, const std::string& model)
    {
        const Shape view{1, task.height, task.width};
        const auto left = net->addLayer("left", LayerSpec::Retina({view}, false, false));
        const auto right = net->addLayer("right", LayerSpec::Retina({view}, false, false));
        net->addInputs(left, view, "left_eye");
        net->addInputs(right, view, "right_eye");
        const size_t pool = static_cast<size_t>(p.getInt("pool"));
        const auto pooled = net->addLayer("pool", LayerSpec::Pool2D(Window2D::square(pool, pool), PoolMode::Average));
        if (model == "stereo") {
            DisparitySpec spec;
            spec.minDisparity = 0;
            spec.maxDisparity = static_cast<int>(p.getInt("max_disparity"));
            spec.window = static_cast<size_t>(p.getInt("window"));
            const std::string measure = p.getString("measure");
            spec.measure = measure == "correlation" ? DisparityMeasure::Correlation
                         : measure == "difference"  ? DisparityMeasure::Difference
                                                    : DisparityMeasure::Normalized;
            const auto depth = net->addLayer("depth", LayerSpec::Disparity(spec));
            net->connect(left, depth);
            net->connect(right, depth);
            net->connect(depth, pooled);
        } else if (model == "mono") {
            net->connect(left, pooled);
        } else if (model == "two_eyes") {
            net->connect(left, pooled);
            net->connect(right, pooled);
        } else {
            throw std::invalid_argument("unknown model '" + model + "'");
        }
        const auto out = net->addLayer("out", LayerSpec::Dense(1, false, false));
        net->connect(pooled, out);
        net->addOutput(out);
    }

    float see(const std::vector<float>& left, const std::vector<float>& right)
    {
        net->setInputs("left_eye", left);
        net->setInputs("right_eye", right);
        net->step();
        return net->outputs()[0];
    }
};

double run(Viewer& v, const Stereograms& task, std::uint32_t seed, int count, float learningRate)
{
    std::mt19937 g(seed);
    std::vector<float> left, right;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float label = task.draw(g, left, right);
        const float y = v.see(left, right);
        correct += y * label > 0.0f;
        if (learningRate > 0.0f && y * label <= 0.0f)
            v.net->applyReward(label, learningRate);
    }
    return static_cast<double>(correct) / count;
}

nnt::Register experiment({
    .name = "stereo_depth",
    .description = "random-dot stereograms: is the near square left or right? stereo (Disparity layer) vs one eye "
                   "vs both eyes without matching",
    .tags = {"vision", "stereo", "learning", "quick"},
    .params = {
        {"height", "16", "view height"},
        {"width", "32", "view width"},
        {"square", "8", "side of the near square"},
        {"near", "3", "the square's disparity in pixels"},
        {"flip", "0", "probability that a right-eye dot is redrawn"},
        {"max_disparity", "5", "Disparity: largest disparity tested (from 0)"},
        {"window", "3", "Disparity: matching window"},
        {"measure", "normalized", "Disparity: correlation, difference or normalized"},
        {"pool", "4", "average pooling window and stride"},
        {"train", "1000", "training stereograms"},
        {"test", "200", "test stereograms"},
        {"lr", "0.01", "learning rate"},
    },
    .trials = 5,
    .expect = {{.metric = "accuracy_stereo", .min = 0.9},
               {.metric = "accuracy_mono", .max = 0.7},
               {.metric = "accuracy_two_eyes", .max = 0.7}},
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        Stereograms task;
        task.height = static_cast<size_t>(p.getInt("height"));
        task.width = static_cast<size_t>(p.getInt("width"));
        task.square = static_cast<size_t>(p.getInt("square"));
        task.near = static_cast<int>(p.getInt("near"));
        task.flip = p.getDouble("flip");
        if (task.near < 1 || task.width < 2 * (task.square + static_cast<size_t>(task.near) + 1) ||
            task.height < task.square)
            throw std::invalid_argument("the square and its disparity must fit in each half of the view");
        const int train_count = static_cast<int>(p.getInt("train")), test_count = static_cast<int>(p.getInt("test"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        for (const std::string model : {"stereo", "mono", "two_eyes"}) {
            exr::reseed(t.seed());
            Viewer v(p, task, model);
            run(v, task, 2000 + t.seed(), train_count, lr);
            t.record("accuracy_" + model, run(v, task, 1000 + t.seed(), test_count, 0.0f));
        }
    },
});

} // namespace
