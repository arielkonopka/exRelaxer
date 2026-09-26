// Fixed filter banks for Conv2D, and a frozen-features vision pipeline that
// learns to tell vertical from horizontal bars.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <numbers>
#include <random>
#include <vector>
#include "../core/filters.hpp"
#include "../core/network.hpp"
#include "../core/layers/conv2d.hpp"
#include "../core/layers/dense.hpp"
#include "../core/layers/retina.hpp"

using namespace exr;
using filters::Filter;

namespace {

constexpr float pi = std::numbers::pi_v<float>;

double sum(const Filter& f)
{
    double s = 0.0;
    for (float w : f.weights) s += w;
    return s;
}

// Response to the best-matching full-contrast pattern: 1 where w > 0.
double bestResponse(const Filter& f)
{
    double s = 0.0;
    for (float w : f.weights) s += w > 0.0f ? w : 0.0f;
    return s;
}

// Response to stripes of `wavelength` along `orientation`, best phase of 8.
double gratingResponse(const Filter& f, float orientation, float wavelength)
{
    const double c = (static_cast<double>(f.width) - 1.0) / 2.0;
    double best = 0.0;
    for (int k = 0; k < 8; ++k) {
        double s = 0.0;
        for (size_t y = 0; y < f.height; ++y)
            for (size_t x = 0; x < f.width; ++x) {
                const double across = -(x - c) * std::sin(orientation) + (y - c) * std::cos(orientation);
                const double pixel = 0.5 + 0.5 * std::cos(2.0 * std::numbers::pi * across / wavelength + k * std::numbers::pi / 4);
                s += f.at(y, x) * pixel;
            }
        best = std::max(best, std::abs(s));
    }
    return best;
}

retina passThrough(const Shape& shape) { return retina(RetinaSpec{shape}, false, false); }

} // namespace

TEST(FiltersTest, GaussianIsNormalizedAndCentred)
{
    const Filter g = filters::gaussian(5, 1.0f, 2.0f);
    EXPECT_NEAR(sum(g), 2.0, 1e-5);
    EXPECT_EQ(g.at(2, 2), *std::max_element(g.weights.begin(), g.weights.end()));
    EXPECT_FLOAT_EQ(g.at(0, 1), g.at(1, 0));
}

TEST(FiltersTest, DifferenceOfGaussiansIsCentreSurround)
{
    const Filter on = filters::differenceOfGaussians(9, 1.0f, 3.0f);
    const Filter off = filters::differenceOfGaussians(9, 1.0f, 3.0f, filters::Polarity::OffCentre);
    EXPECT_NEAR(sum(on), 0.0, 1e-6);         // a uniform image gives 0
    EXPECT_NEAR(bestResponse(on), 1.0, 1e-5);  // the best pattern gives the gain
    EXPECT_GT(on.at(4, 4), 0.0f);            // bright centre
    EXPECT_LT(on.at(4, 1), 0.0f);            // dark surround
    for (size_t i = 0; i < on.weights.size(); ++i)
        EXPECT_FLOAT_EQ(off.weights[i], -on.weights[i]);
    EXPECT_THROW(filters::differenceOfGaussians(9, 3.0f, 1.0f), std::invalid_argument);
}

TEST(FiltersTest, GaborIsOrientationSelective)
{
    const Filter horizontal = filters::gabor(9, 0.0f, 4.0f, 2.0f);
    const Filter vertical = filters::gabor(9, pi / 2, 4.0f, 2.0f);
    EXPECT_NEAR(sum(horizontal), 0.0, 1e-6);
    EXPECT_NEAR(bestResponse(horizontal), 1.0, 1e-5);
    EXPECT_GT(horizontal.at(4, 4), 0.0f);  // phase 0: a bright bar on the centre line
    // Horizontal stripes vs vertical stripes, and the other way round.
    EXPECT_GT(gratingResponse(horizontal, 0.0f, 4.0f), 10.0 * gratingResponse(horizontal, pi / 2, 4.0f));
    EXPECT_GT(gratingResponse(vertical, pi / 2, 4.0f), 10.0 * gratingResponse(vertical, 0.0f, 4.0f));
    // The same filter turned by 90 degrees.
    for (size_t y = 0; y < 9; ++y)
        for (size_t x = 0; x < 9; ++x)
            EXPECT_NEAR(vertical.at(y, x), horizontal.at(x, y), 1e-6);
}

TEST(FiltersTest, BanksCoverOrientationsAndPhases)
{
    const auto bank = filters::gaborBank(7, 4, 4.0f, 2.0f);
    ASSERT_EQ(bank.size(), 8u);  // 4 orientations x {0, pi/2}
    const Filter expected = filters::gabor(7, pi / 4, 4.0f, 2.0f, pi / 2);
    EXPECT_EQ(bank[3].weights, expected.weights);  // orientation 1, phase 1
    EXPECT_EQ(filters::centreSurroundBank(7, 1.0f, 2.5f).size(), 2u);
}

TEST(FiltersTest, LoadIntoConv2D)
{
    // The same filter on every channel of a grey RGB image gives the
    // single-channel response.
    const Shape rgb{3, 9, 9};
    retina eye = passThrough(rgb), grey = passThrough({1, 9, 9});
    std::vector<float> image(rgb.size()), one(81);
    std::mt19937 g(1);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    for (size_t i = 0; i < 81; ++i) one[i] = image[i] = image[81 + i] = image[162 + i] = d(g);
    eye.attachInputs(image);
    grey.attachInputs(one);
    conv2d a(2, Window2D::square(5), false, false), b(2, Window2D::square(5), false, false);
    a.join(eye);
    b.join(grey);
    const auto bank = filters::centreSurroundBank(5, 0.8f, 2.0f);
    filters::load(a, bank);
    filters::load(b, bank);
    eye.forward(); grey.forward(); a.forward(); b.forward();
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a.output()[i], b.output()[i], 1e-5);

    // One filter on one input channel.
    filters::load(a, 1, bank[0], 2);
    const std::vector<float> k = a.kernel(1);
    for (size_t i = 0; i < 25; ++i) {
        EXPECT_EQ(k[i], 0.0f);
        EXPECT_EQ(k[50 + i], bank[0].weights[i]);
    }

    EXPECT_THROW(filters::load(a, filters::gaborBank(5, 3, 4.0f, 2.0f)), std::invalid_argument);  // 6 != 2
    EXPECT_THROW(filters::load(a, std::vector<Filter>{bank[0], filters::gaussian(3, 1.0f)}), std::invalid_argument);
    conv2d unwired(2, Window2D::square(5));
    EXPECT_THROW(filters::load(unwired, bank), std::invalid_argument);
}

// =============================================================================
// Frozen Gabor features + learned readout: vertical vs horizontal bars
// =============================================================================
namespace {

constexpr size_t SIDE = 24;

// A 2-pixel-wide, 12-pixel-long bar at a random position, on a noisy
// background. Returns +1 for vertical, -1 for horizontal.
float drawBar(std::mt19937& g, std::vector<float>& image)
{
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);
    std::uniform_int_distribution<size_t> pos(4, SIDE - 5), coin(0, 1);
    for (float& v : image) v = 0.1f + noise(g);
    const bool vertical = coin(g) == 1;
    const size_t cy = pos(g), cx = pos(g);
    for (size_t along = 0; along < 12; ++along)
        for (size_t across = 0; across < 2; ++across) {
            const size_t y = vertical ? cy - 6 + along : cy + across;
            const size_t x = vertical ? cx + across : cx - 6 + along;
            if (y < SIDE && x < SIDE) image[y * SIDE + x] = 0.9f + noise(g);
        }
    return vertical ? 1.0f : -1.0f;
}

struct OrientationNet
{
    std::unique_ptr<network> net = std::make_unique<network>();
    network::LayerId out{};

    OrientationNet()
    {
        const Shape image{1, SIDE, SIDE};
        const auto eye = net->addLayer("eye", LayerSpec::Retina({image}, false, false));
        const auto v1 = net->addLayer("v1", LayerSpec::Conv2D(8, Window2D::square(7, 1, 3), false, false));
        const auto pool = net->addLayer("pool", LayerSpec::Pool2D(Window2D::square(6, 6)));
        // Pooled orientation energies are all >= 0, but the learning rule
        // only uses input signs: a frozen random mixing layer turns them
        // into features whose signs differ between the classes.
        const auto mix = net->addLayer("mix", LayerSpec::Dense(64, false, false));
        out = net->addLayer("out", LayerSpec::Dense(1, false, false));
        net->addInputs(eye, image);
        net->connect(eye, v1);
        net->connect(v1, pool);
        net->connect(pool, mix);
        net->connect(mix, out);
        net->addOutput(out);
        filters::load(net->layerAs<conv2d>(v1), filters::gaborBank(7, 4, 5.0f, 2.0f));
        for (network::LayerId frozen : {v1, mix})
            net->freeze(frozen);
    }

    float classify(const std::vector<float>& image)
    {
        net->setInputs(image);
        net->step();  // no E-R or habituation: one step crosses the whole path
        return net->outputs()[0];
    }
};

struct OrientationResult
{
    double before = 0, after = 0, control = 0;
};

OrientationResult runOrientationTrial(std::uint32_t seed)
{
    constexpr int TRAIN = 1500, TEST = 200;
    auto accuracy = [](OrientationNet& n, std::uint32_t testSeed) {
        std::mt19937 g(testSeed);
        std::vector<float> image(SIDE * SIDE);
        int correct = 0;
        for (int i = 0; i < TEST; ++i) {
            const float label = drawBar(g, image);
            correct += n.classify(image) * label > 0.0f;
        }
        return static_cast<double>(correct) / TEST;
    };
    auto train = [](OrientationNet& n, std::uint32_t trainSeed, float learningRate) {
        std::mt19937 g(trainSeed);
        std::vector<float> image(SIDE * SIDE);
        for (int i = 0; i < TRAIN; ++i) {
            const float label = drawBar(g, image);
            if (n.classify(image) * label <= 0.0f)        // error-driven: learn only when wrong
                n.net->applyReward(label, learningRate);
        }
    };

    OrientationResult r;
    reseed(seed);
    OrientationNet learner;
    r.before = accuracy(learner, 1000 + seed);
    train(learner, 2000 + seed, 0.01f);
    r.after = accuracy(learner, 1000 + seed);

    reseed(seed);
    OrientationNet control;
    train(control, 2000 + seed, 0.0f);
    r.control = accuracy(control, 1000 + seed);
    return r;
}

} // namespace

TEST(FiltersTest, FrozenGaborFeaturesLearnBarOrientation)
{
    constexpr int TRIALS = 10;
    double before = 0, after = 0, control = 0, worst = 1;
    for (int t = 0; t < TRIALS; ++t) {
        const OrientationResult r = runOrientationTrial(static_cast<std::uint32_t>(t));
        before += r.before / TRIALS;
        after += r.after / TRIALS;
        control += r.control / TRIALS;
        worst = std::min(worst, r.after);
    }
    std::ostringstream text;  // its own format: std::cout's may have been changed by other tests
    text << std::setprecision(4) << "\n==========================================\n"
         << " [Bar orientation: frozen Gabor bank + learned readout - " << TRIALS << " trials]\n"
         << "==========================================\n"
         << " Accuracy before " << before << ", after " << after << " (worst " << worst
         << "), control " << control << "\n"
         << "==========================================\n";
    std::cout << text.str();
    // Measured: 0.52 before and in the control, 0.99 after (worst trial
    // 0.965). Mixing width and training length were the limits: 32 mixing
    // neurons and 400 images gave 0.87, 64 and 400 gave 0.94.
    EXPECT_GT(after, 0.95);
    EXPECT_GT(worst, 0.9);
    EXPECT_LT(control, 0.6);
}
