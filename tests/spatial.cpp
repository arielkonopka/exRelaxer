// Spatial layers: Retina (grid and spiral sampling), Conv2D, LocallyConnected2D
// and Pool2D, each against a plain scalar reference.
#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <sstream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../core/network.hpp"
#include "../core/layers/conv2d.hpp"
#include "../core/layers/dense.hpp"
#include "../core/layers/locally_connected2d.hpp"
#include "../core/layers/pool2d.hpp"
#include "../core/layers/retina.hpp"

using namespace exr;

namespace {

bool sameBits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

std::vector<float> randomImage(const Shape& shape, std::uint32_t seed, float scale = 1.0f)
{
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> image(shape.size());
    for (float& v : image) v = d(g);
    return image;
}

// A retina passing the image through unchanged (no habituation, no E-R).
retina passThrough(const Shape& shape) { return retina(RetinaSpec{shape}, false, false); }

// The reference window: channel by channel, row by row, zero outside.
std::vector<float> referenceWindow(const std::vector<float>& image, const Shape& in, const Window2D& w, size_t oy,
                                   size_t ox)
{
    std::vector<float> out;
    for (size_t c = 0; c < in.channels; ++c)
        for (size_t ky = 0; ky < w.kernelHeight; ++ky)
            for (size_t kx = 0; kx < w.kernelWidth; ++kx) {
                const long y = static_cast<long>(oy * w.strideY + ky) - static_cast<long>(w.padY);
                const long x = static_cast<long>(ox * w.strideX + kx) - static_cast<long>(w.padX);
                const bool inside = y >= 0 && x >= 0 && y < static_cast<long>(in.height) && x < static_cast<long>(in.width);
                out.push_back(inside ? image[(c * in.height + static_cast<size_t>(y)) * in.width + static_cast<size_t>(x)]
                                     : 0.0f);
            }
    return out;
}

} // namespace

// =============================================================================
// Window2D
// =============================================================================
TEST(SpatialTest, WindowOutputSize)
{
    EXPECT_EQ(Window2D::square(3).outputHeight(10), 8u);
    EXPECT_EQ(Window2D::square(3, 1, 1).outputHeight(10), 10u);  // "same" padding
    EXPECT_EQ(Window2D::square(2, 2).outputWidth(320), 160u);
    EXPECT_EQ(Window2D::square(5, 2, 2).outputWidth(9), 5u);
    EXPECT_EQ(Window2D::square(5).outputHeight(4), 0u);          // does not fit
}

// =============================================================================
// Retina
// =============================================================================
TEST(RetinaTest, GridPassesTheImageThrough)
{
    const Shape shape{3, 4, 5};
    retina eye = passThrough(shape);
    EXPECT_EQ(eye.shape(), shape);
    ASSERT_EQ(eye.size(), shape.size());

    std::vector<float> image = randomImage(shape, 1);
    eye.attachInputs(image);
    eye.forward();
    for (size_t i = 0; i < image.size(); ++i)
        EXPECT_EQ(eye.output()[i], image[i]);
}

TEST(RetinaTest, SpiralStartsAtTheCentreAndWindsOutwards)
{
    const RetinaSpec spec{{1, 200, 320}, Sampling::Spiral, 1.0f};
    retina eye(spec, false, false);
    const size_t n = eye.samples();
    ASSERT_GT(n, 1000u);
    EXPECT_EQ(eye.shape(), (Shape{1, 1, n}));

    const double cx = 159.5, cy = 99.5;
    EXPECT_DOUBLE_EQ(eye.samplePoint(0).x, cx);
    EXPECT_DOUBLE_EQ(eye.samplePoint(0).y, cy);

    // About one sample per spacing^2 of the disc of radius 99.5 (the largest
    // circle inside the image).
    EXPECT_NEAR(static_cast<double>(n), std::numbers::pi * 99.5 * 99.5, 0.03 * std::numbers::pi * 99.5 * 99.5);

    double previous_radius = 0.0;
    for (size_t k = 1; k < n; ++k) {
        const retina::Point p = eye.samplePoint(k), q = eye.samplePoint(k - 1);
        const double radius = std::hypot(p.x - cx, p.y - cy);
        EXPECT_GT(radius, previous_radius) << "sample " << k;   // strictly outwards
        EXPECT_LE(radius, 99.5 + 1e-9);                          // inside the image
        previous_radius = radius;
        if (k >= 2) {                                            // `spacing` apart along the spiral
            EXPECT_NEAR(std::hypot(p.x - q.x, p.y - q.y), 1.0, 0.1) << "sample " << k;
        }
    }
}

TEST(RetinaTest, SpiralInterpolatesBilinearly)
{
    // On a linear image, bilinear interpolation is exact: sample k must read
    // the image's value at its own position.
    const Shape shape{2, 30, 40};
    retina eye(RetinaSpec{shape, Sampling::Spiral, 0.7f}, false, false);
    std::vector<float> image(shape.size());
    for (size_t c = 0; c < 2; ++c)
        for (size_t y = 0; y < 30; ++y)
            for (size_t x = 0; x < 40; ++x)
                image[(c * 30 + y) * 40 + x] = 0.05f * x - 0.03f * y + (c == 0 ? 0.25f : -1.0f);
    eye.attachInputs(image);
    eye.forward();

    const size_t n = eye.samples();
    for (size_t c = 0; c < 2; ++c)
        for (size_t k = 0; k < n; ++k) {
            const retina::Point p = eye.samplePoint(k);
            const double expected = 0.05 * p.x - 0.03 * p.y + (c == 0 ? 0.25 : -1.0);
            EXPECT_NEAR(eye.output()[c * n + k], expected, 1e-5) << "channel " << c << " sample " << k;
        }
}

TEST(RetinaTest, RadiusLimitsTheSpiral)
{
    retina small(RetinaSpec{{1, 100, 100}, Sampling::Spiral, 1.0f, 10.0f}, false, false);
    for (size_t k = 0; k < small.samples(); ++k)
        EXPECT_LE(std::hypot(small.samplePoint(k).x - 49.5, small.samplePoint(k).y - 49.5), 10.0 + 1e-9);
    EXPECT_NEAR(static_cast<double>(small.samples()), std::numbers::pi * 100.0, 0.1 * std::numbers::pi * 100.0);
}

TEST(RetinaTest, RejectsWrongInputs)
{
    EXPECT_THROW(retina(RetinaSpec{{0, 0, 0}}), std::invalid_argument);
    EXPECT_THROW(retina(RetinaSpec{{1, 5, 5}, Sampling::Spiral, 0.0f}), std::invalid_argument);

    retina eye = passThrough({1, 4, 4});
    std::vector<float> wrong(15), right(16);
    EXPECT_THROW(eye.attachInputs(wrong), std::invalid_argument);
    eye.attachInputs(right);
    EXPECT_THROW(eye.attachInputs(right), std::logic_error);
    dense other(3);
    EXPECT_THROW(eye.join(other), std::logic_error);
}

// =============================================================================
// Conv2D
// =============================================================================

// Every output neuron against a scalar reference: its window, the channel's
// kernel, a copy of its neuron. Multiple tiles, the parallel path, stride,
// padding and two sources.
TEST(Conv2DTest, ForwardMatchesScalarReferenceBitForBit)
{
    reseed(3);
    const Shape a_shape{2, 37, 41}, b_shape{1, 37, 41};
    retina a = passThrough(a_shape), b = passThrough(b_shape);
    std::vector<float> a_image = randomImage(a_shape, 4, 2.0f), b_image = randomImage(b_shape, 5, 2.0f);
    a.attachInputs(a_image);
    b.attachInputs(b_image);

    const Window2D w{5, 3, 2, 1, 2, 1};
    conv2d conv(9, w, true, true);
    conv.join(a);
    conv.join(b);
    ASSERT_EQ(conv.inputChannels(), 3u);
    ASSERT_EQ(conv.windowSize(), 3u * 15u);
    const Shape out = conv.shape();
    EXPECT_EQ(out, (Shape{9, w.outputHeight(37), w.outputWidth(41)}));

    std::vector<neuron> reference(conv.neurons().begin(), conv.neurons().end());
    std::vector<float> image = a_image;
    image.insert(image.end(), b_image.begin(), b_image.end());
    const Shape in{3, 37, 41};

    for (int t = 0; t < 3; ++t) {
        a.forward();
        b.forward();
        conv.forward();
        const size_t positions = out.height * out.width;
        for (size_t c = 0; c < out.channels; ++c) {
            const std::vector<float> kernel = conv.kernel(c);
            for (size_t p = 0; p < positions; ++p) {
                const std::vector<float> window = referenceWindow(image, in, w, p / out.width, p % out.width);
                const float expected = reference[c * positions + p].step(window, kernel);
                ASSERT_TRUE(sameBits(conv.output()[c * positions + p], expected)) << "tick " << t << " channel " << c
                                                                                   << " position " << p;
            }
        }
    }
}

// One position: the shared kernel learns exactly like one dense neuron.
TEST(Conv2DTest, SinglePositionLearnsLikeADenseNeuron)
{
    retina eye = passThrough({1, 3, 3});
    std::vector<float> image = {0.5f, -1.0f, 0.0f, 2.0f, 0.25f, -0.5f, 1.0f, 1.5f, -2.0f};
    eye.attachInputs(image);
    conv2d conv(1, Window2D::square(3), false, false);
    conv.join(eye);
    ASSERT_EQ(conv.size(), 1u);
    std::vector<float> kernel(9, 0.3f);
    conv.setKernel(0, kernel);

    eye.forward();
    conv.forward();
    neuron reference(false, false);
    reference.step(image, kernel);
    conv.applyReward(1.0f, 0.05f);
    reference.learn(kernel, image, 1.0f, 0.05f);
    const std::vector<float> learned = conv.kernel(0);
    for (size_t j = 0; j < 9; ++j)
        EXPECT_TRUE(sameBits(learned[j], kernel[j])) << "weight " << j;
}

// The kernel moves by the mean of its eligible neurons' updates, not the sum.
TEST(Conv2DTest, SharedKernelLearnsTheMeanUpdate)
{
    retina eye = passThrough({1, 1, 4});
    std::vector<float> image = {1.0f, 1.0f, 1.0f, 1.0f};
    eye.attachInputs(image);
    conv2d conv(1, Window2D{1, 2}, false, false);  // 3 positions, same window values
    conv.join(eye);
    conv.setKernel(0, {0.5f, 0.5f});
    eye.forward();
    conv.forward();
    conv.applyReward(1.0f, 0.1f);
    const float step = 0.1f * default_learning_gain;  // each neuron's delta, sign +1
    for (float w : conv.kernel(0))
        EXPECT_FLOAT_EQ(w, 0.5f + step);
}

// Learning over many positions uses fixed chunks: the same result for any
// thread count.
TEST(Conv2DTest, LearningDoesNotDependOnThreadCount)
{
    auto run = [](int threads) {
#ifdef _OPENMP
        const int saved = omp_get_max_threads();
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        reseed(8);
        const Shape shape{2, 90, 90};
        retina eye = passThrough(shape);
        std::vector<float> image = randomImage(shape, 9);
        eye.attachInputs(image);
        conv2d conv(8, Window2D::square(5, 1, 2), false, true);
        conv.join(eye);
        for (int t = 0; t < 4; ++t) {
            eye.forward();
            conv.forward();
            conv.applyReward(t % 2 == 0 ? 1.0f : -0.5f, 0.01f);
        }
#ifdef _OPENMP
        omp_set_num_threads(saved);
#endif
        std::vector<float> all;
        for (size_t c = 0; c < 8; ++c) {
            const std::vector<float> k = conv.kernel(c);
            all.insert(all.end(), k.begin(), k.end());
        }
        all.insert(all.end(), conv.output().begin(), conv.output().end());
        return all;
    };
    const std::vector<float> serial = run(1), parallel = run(8);
    ASSERT_EQ(serial.size(), parallel.size());
    for (size_t i = 0; i < serial.size(); ++i)
        ASSERT_TRUE(sameBits(serial[i], parallel[i])) << "value " << i;
}

// A flat source is n channels of one pixel; when it grows, the new channels
// are appended to every kernel.
TEST(Conv2DTest, FlatSourceGrowthAddsChannels)
{
    dense source(3), feeder(2);
    std::vector<float> sensor(1, 0.5f);
    source.attachInputs(sensor);
    conv2d conv(2, Window2D::square(1));
    conv.join(source);
    EXPECT_EQ(conv.shape(), (Shape{2, 1, 1}));
    EXPECT_EQ(conv.kernel(0).size(), 3u);
    const std::vector<float> before = conv.kernel(1);

    feeder.addFeedback(source, 2);  // source: 3 -> 5 outputs
    EXPECT_EQ(conv.inputChannels(), 5u);
    const std::vector<float> after = conv.kernel(1);
    ASSERT_EQ(after.size(), 5u);
    for (size_t j = 0; j < 3; ++j)
        EXPECT_EQ(after[j], before[j]);
    source.forward();
    EXPECT_NO_THROW(conv.forward());
}

TEST(Conv2DTest, RejectsMismatchedOrTooSmallInputs)
{
    retina a = passThrough({1, 10, 10}), b = passThrough({1, 10, 12});
    conv2d conv(4, Window2D::square(3));
    conv.join(a);
    EXPECT_THROW(conv.join(b), std::invalid_argument);  // other height x width
    conv2d big(4, Window2D::square(11));
    EXPECT_THROW(big.join(a), std::invalid_argument);   // the window does not fit
    EXPECT_THROW(conv2d(0, Window2D::square(3)), std::invalid_argument);
    EXPECT_THROW(conv.addNeurons(2, a), std::logic_error);
}

// =============================================================================
// LocallyConnected2D
// =============================================================================
TEST(LocallyConnected2DTest, MatchesPerNeuronReferenceBitForBit)
{
    reseed(12);
    const Shape shape{2, 12, 14};
    retina eye = passThrough(shape);
    std::vector<float> image = randomImage(shape, 13, 2.0f);
    eye.attachInputs(image);
    const Window2D w = Window2D::square(3, 2, 1);
    locally_connected2d lc(11, w, true, true);  // 11 channels: a partial block of 8
    lc.join(eye);
    const Shape out = lc.shape();
    const size_t positions = out.height * out.width;

    std::vector<neuron> reference(lc.neurons().begin(), lc.neurons().end());
    std::vector<std::vector<float>> weights;
    for (size_t i = 0; i < lc.size(); ++i)
        weights.push_back(lc.weights(i));

    for (int t = 0; t < 20; ++t) {
        for (float& v : image) v = -v * 0.9f + 0.1f * static_cast<float>(t % 3);
        eye.forward();
        lc.forward();
        for (size_t i = 0; i < lc.size(); ++i) {
            const size_t p = i % positions;
            const std::vector<float> window = referenceWindow(image, shape, w, p / out.width, p % out.width);
            ASSERT_TRUE(sameBits(lc.output()[i], reference[i].step(window, weights[i]))) << "tick " << t << " neuron " << i;
        }
        const float reward = t % 2 == 0 ? 1.0f : -1.0f;
        lc.applyReward(reward, 0.02f);
        for (size_t i = 0; i < lc.size(); ++i) {
            const size_t p = i % positions;
            reference[i].learn(weights[i], referenceWindow(image, shape, w, p / out.width, p % out.width), reward, 0.02f);
        }
    }
    for (size_t i = 0; i < lc.size(); ++i) {
        const std::vector<float> actual = lc.weights(i);
        for (size_t j = 0; j < actual.size(); ++j)
            ASSERT_TRUE(sameBits(actual[j], weights[i][j])) << "neuron " << i << " weight " << j;
    }
}

// =============================================================================
// Pool2D
// =============================================================================
TEST(Pool2DTest, MaxAndAverageIgnorePadding)
{
    retina eye = passThrough({1, 3, 3});
    std::vector<float> image = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9};
    eye.attachInputs(image);
    pool2d maxPool(Window2D::square(2, 2, 1), PoolMode::Max);
    pool2d avgPool(Window2D::square(2, 2, 1), PoolMode::Average);
    maxPool.join(eye);
    avgPool.join(eye);
    EXPECT_EQ(maxPool.shape(), (Shape{1, 2, 2}));
    eye.forward();
    maxPool.forward();
    avgPool.forward();
    // Windows (with 1 pixel of padding): {1}, {2,3}, {4,7}, {5,6,8,9}
    EXPECT_EQ(std::vector<float>(maxPool.output().begin(), maxPool.output().end()), (std::vector<float>{1, 3, 7, 9}));
    EXPECT_EQ(std::vector<float>(avgPool.output().begin(), avgPool.output().end()), (std::vector<float>{1, 2.5f, 5.5f, 7}));
}

TEST(Pool2DTest, PoolsEveryChannelOfEverySource)
{
    retina a = passThrough({2, 4, 4}), b = passThrough({1, 4, 4});
    std::vector<float> ia(32), ib(16);
    // Within the neurons' +-10 output range: a retina clamps like any neuron.
    for (size_t i = 0; i < 32; ++i) ia[i] = 0.25f * static_cast<float>(i);
    for (size_t i = 0; i < 16; ++i) ib[i] = -0.25f * static_cast<float>(i);
    a.attachInputs(ia);
    b.attachInputs(ib);
    pool2d pool(Window2D::square(2, 2));
    pool.join(a);
    dense reader(1);
    reader.join(pool);                 // reads 2 x 2 x 2 = 8 outputs
    pool.join(b);                      // a third channel: the reader is extended
    EXPECT_EQ(pool.shape(), (Shape{3, 2, 2}));
    EXPECT_EQ(reader.inputCount(0), 12u);
    a.forward();
    b.forward();
    pool.forward();
    EXPECT_EQ(pool.output()[0], 1.25f);   // max of {0, 0.25, 1, 1.25}
    EXPECT_EQ(pool.output()[4], 5.25f);   // channel 1: max of {4, 4.25, 5, 5.25}
    EXPECT_EQ(pool.output()[8], 0.0f);    // channel 2: max of {0, -0.25, -1, -1.25}
}

// =============================================================================
// A vision pipeline in a network
// =============================================================================
namespace {

std::unique_ptr<network> buildVision()
{
    auto net = std::make_unique<network>();
    const Shape image{1, 20, 24};
    const auto eye = net->addLayer("eye", LayerSpec::Retina({image}, true, true));
    const auto v1 = net->addLayer("v1", LayerSpec::Conv2D(4, Window2D::square(3, 1, 1), false, true));
    const auto pool = net->addLayer("pool", LayerSpec::Pool2D(Window2D::square(2, 2)));
    const auto v2 = net->addLayer("v2", LayerSpec::LocallyConnected2D(3, Window2D::square(3), false, true));
    const auto out = net->addLayer("out", LayerSpec::Dense(2, false, false));
    net->addInputs(eye, image);
    net->connect(eye, v1);
    net->connect(v1, pool);
    net->connect(pool, v2);
    net->connect(v2, out);
    net->addOutput(out);
    return net;
}

std::vector<float> frame(int t)
{
    std::vector<float> image(20 * 24);
    for (size_t y = 0; y < 20; ++y)
        for (size_t x = 0; x < 24; ++x)
            image[y * 24 + x] = std::sin(0.3f * static_cast<float>(x + static_cast<size_t>(t)) + 0.2f * static_cast<float>(y));
    return image;
}

std::vector<float> run(network& net, int from, int ticks)
{
    std::vector<float> trace;
    for (int t = from; t < from + ticks; ++t) {
        net.setInputs(frame(t));
        net.step();
        const std::vector<float> y = net.outputs();
        trace.insert(trace.end(), y.begin(), y.end());
        net.applyReward(y[0] > 0 ? -1.0f : 1.0f, 0.01f);
    }
    return trace;
}

} // namespace

TEST(VisionNetworkTest, BuildsRunsAndContinuesExactlyAfterLoad)
{
    reseed(21);
    auto net = buildVision();
    EXPECT_EQ(net->getLayer(1).shape(), (Shape{4, 20, 24}));
    EXPECT_EQ(net->getLayer(2).shape(), (Shape{4, 10, 12}));
    EXPECT_EQ(net->getLayer(3).shape(), (Shape{3, 8, 10}));
    std::ostringstream text;
    net->describe(text);
    EXPECT_NE(text.str().find("4x10x12"), std::string::npos) << text.str();

    run(*net, 0, 30);
    std::stringstream data;
    net->save(data);
    const std::vector<float> original = run(*net, 30, 20);

    reseed(999);  // loading must not depend on the random streams
    auto restored = network::load(data);
    const std::vector<float> continued = run(*restored, 30, 20);
    ASSERT_EQ(original.size(), continued.size());
    for (size_t i = 0; i < original.size(); ++i)
        ASSERT_TRUE(sameBits(original[i], continued[i])) << "value " << i;
}

TEST(VisionNetworkTest, SpiralRetinaFeedsA1DConvolution)
{
    reseed(5);
    network net;
    const auto eye = net.addLayer("eye", LayerSpec::Retina({{1, 32, 32}, Sampling::Spiral, 1.0f}, false, false));
    const auto conv = net.addLayer("along", LayerSpec::Conv2D(2, Window2D{1, 5}, false, false));
    net.addInputs(eye, Shape{1, 32, 32});
    net.connect(eye, conv);
    const size_t samples = dynamic_cast<const retina&>(net.getLayer(eye)).samples();
    EXPECT_EQ(net.getLayer(conv).shape(), (Shape{2, 1, samples - 4}));
    net.setInputs(std::vector<float>(32 * 32, 0.5f));
    EXPECT_NO_THROW(net.step());
}
