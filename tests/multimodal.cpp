// Multimodal and stereo building blocks: named input sources, a cochlea with
// several microphones, Resize2D and Disparity, and saving networks that use
// them.
#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <sstream>
#include <vector>
#include "../core/network.hpp"
#include "../core/layers/cochlea.hpp"
#include "../core/layers/dense.hpp"
#include "../core/layers/disparity.hpp"
#include "../core/layers/resize2d.hpp"
#include "../core/layers/retina.hpp"

using namespace exr;

namespace {

bool sameBits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

std::vector<float> tone(double frequency, double sampleRate, size_t from, size_t count)
{
    std::vector<float> s(count);
    for (size_t n = 0; n < count; ++n)
        s[n] = static_cast<float>(0.5 * std::sin(2.0 * std::numbers::pi * frequency *
                                                 static_cast<double>(from + n) / sampleRate));
    return s;
}

CochleaSpec stereoSpec()
{
    CochleaSpec spec;
    spec.sampleRate = 8000.0f;
    spec.hop = 64;
    spec.window = 256;
    spec.bands = 16;
    spec.minFrequency = 100.0f;
    spec.compression = Compression::Linear;
    spec.gain = 1.0f;
    spec.channels = 2;
    return spec;
}

// A layer whose output is set directly, as a view for Disparity / Resize2D.
struct fixed_map final : layer
{
    explicit fixed_map(const Shape& shape) : shape_(shape) { output_.assign(shape.size(), 0.0f); }
    LayerType type() const override { return LayerType::Dense; }
    Shape shape() const override { return shape_; }
    void forward() override {}
    void serialize(std::ostream&) const override {}
    void deserialize(std::istream&, DeserializeMode, std::uint32_t) override {}
    void set(const std::vector<float>& values) { output_ = values; }
    void grow(size_t channels)
    {
        const size_t old = output_.size();
        shape_.channels += channels;
        output_.resize(shape_.size(), 0.0f);
        outputGrew(old);
    }
    Shape shape_;
};

} // namespace

// =============================================================================
// Named input sources
// =============================================================================
TEST(InputSourceTest, NamedSourcesAreSetByNameAndFeedSeveralLayers)
{
    network net;
    const auto eye = net.addLayer("eye", LayerSpec::Retina({{1, 2, 3}}, false, false));
    const auto ear = net.addLayer("ear", LayerSpec::Dense(2, false, false));
    const auto both = net.addLayer("both", LayerSpec::Dense(1, false, false));
    EXPECT_EQ(net.addInputs(eye, Shape{1, 2, 3}, "camera"), 0u);
    EXPECT_EQ(net.addInputs(ear, 4, "microphone"), 6u);
    net.connectInputs("camera", both);  // the same six sensors, read by a second layer

    ASSERT_EQ(net.inputSources().size(), 2u);
    EXPECT_EQ(net.inputSource("camera").shape, (Shape{1, 2, 3}));
    EXPECT_EQ(net.inputSource("camera").targets, (std::vector<network::LayerId>{eye, both}));
    EXPECT_EQ(net.inputSource("microphone").first, 6u);
    EXPECT_EQ(net.inputCount(), 10u);

    const std::vector<float> image{1, 2, 3, 4, 5, 6}, sound{-1, -2, -3, -4};
    net.setInputs("camera", image);
    net.setInputs("microphone", sound);
    EXPECT_EQ(std::vector<float>(net.inputs("camera").begin(), net.inputs("camera").end()), image);
    EXPECT_EQ(net.inputs()[6], -1.0f);
    net.step();
    EXPECT_EQ(net.getLayer(eye).output()[5], 6.0f);

    EXPECT_THROW(net.setInputs("camera", sound), std::invalid_argument);
    EXPECT_THROW(net.setInputs("nose", sound), std::out_of_range);
    EXPECT_THROW(net.addInputs(ear, 1, "camera"), std::invalid_argument);
    EXPECT_THROW(net.connectInputs("camera", both), std::invalid_argument);
    EXPECT_THROW(net.connectInputs("", both), std::out_of_range);

    std::ostringstream text;
    net.describe(text);
    EXPECT_NE(text.str().find("inputs 'camera': 6 (1x2x3) -> eye, both"), std::string::npos) << text.str();
}

TEST(InputSourceTest, SourcesSurviveSaveAndLoad)
{
    reseed(3);
    network net;
    const auto left = net.addLayer("left", LayerSpec::Retina({{1, 4, 4}}, false, false));
    const auto right = net.addLayer("right", LayerSpec::Retina({{1, 4, 4}}, false, false));
    const auto mono = net.addLayer("mono", LayerSpec::Dense(3, false, false));
    net.addInputs(left, Shape{1, 4, 4}, "left_eye");
    net.addInputs(right, Shape{1, 4, 4}, "right_eye");
    net.connectInputs("left_eye", mono);
    net.addOutput(mono);
    std::vector<float> image(16);
    for (size_t i = 0; i < 16; ++i)
        image[i] = 0.1f * static_cast<float>(i);
    net.setInputs("left_eye", image);
    net.step();

    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    ASSERT_EQ(loaded->inputSources().size(), 2u);
    EXPECT_EQ(loaded->inputSource("right_eye").shape, (Shape{1, 4, 4}));
    EXPECT_EQ(loaded->inputSource("left_eye").targets, (std::vector<network::LayerId>{left, mono}));
    loaded->step();
    net.step();
    for (size_t i = 0; i < net.outputs().size(); ++i)
        EXPECT_TRUE(sameBits(net.outputs()[i], loaded->outputs()[i]));
}

// =============================================================================
// Multichannel cochlea
// =============================================================================
TEST(CochleaChannelsTest, EachMicrophoneIsAnalysedLikeASingleOne)
{
    const CochleaSpec spec = stereoSpec();
    CochleaSpec mono = spec;
    mono.channels = 1;
    cochlea both(spec, false, false), left(mono, false, false), right(mono, false, false);
    ASSERT_EQ(both.shape(), (Shape{2, 16, 1}));
    std::vector<float> sensors(2 * spec.hop), l(spec.hop), r(spec.hop);
    both.attachInputs(sensors);
    left.attachInputs(l);
    right.attachInputs(r);
    for (size_t t = 0; t < 8; ++t) {
        const auto a = tone(440.0, spec.sampleRate, t * spec.hop, spec.hop);
        const auto b = tone(2000.0, spec.sampleRate, t * spec.hop, spec.hop);
        std::copy(a.begin(), a.end(), sensors.begin());
        std::copy(b.begin(), b.end(), sensors.begin() + static_cast<std::ptrdiff_t>(spec.hop));
        l = a;
        r = b;
        both.forward();
        left.forward();
        right.forward();
    }
    for (size_t k = 0; k < spec.bands; ++k) {
        EXPECT_TRUE(sameBits(both.output()[k], left.output()[k])) << k;
        EXPECT_TRUE(sameBits(both.output()[spec.bands + k], right.output()[k])) << k;
    }
    EXPECT_EQ(both.power().size(), 2 * (spec.window / 2 + 1));

    std::vector<float> wrong(spec.hop);
    cochlea other(spec, false, false);
    EXPECT_THROW(other.attachInputs(wrong), std::invalid_argument);
    CochleaSpec none = spec;
    none.channels = 0;
    EXPECT_THROW(cochlea{none}, std::invalid_argument);
}

TEST(CochleaChannelsTest, StereoCochleaSavesItsChannels)
{
    network net;
    const auto ear = net.addLayer("ears", LayerSpec::Cochlea(stereoSpec(), false, false));
    net.addInputs(ear, 2 * stereoSpec().hop, "microphones");
    net.addOutput(ear);
    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_EQ(loaded->layerAs<cochlea>(ear).spec(), net.layerAs<cochlea>(ear).spec());
    EXPECT_EQ(loaded->layerAs<cochlea>(ear).spec().channels, 2u);
    EXPECT_EQ(loaded->getLayer(ear).shape(), (Shape{2, 16, 1}));
}

// =============================================================================
// Resize2D
// =============================================================================
TEST(Resize2DTest, InterpolationModes)
{
    fixed_map source(Shape{1, 2, 4});
    source.set({0, 1, 2, 3, 4, 5, 6, 7});

    resize2d nearest(ResizeSpec{4, 8, Interpolation::Nearest});
    nearest.join(source);
    ASSERT_EQ(nearest.shape(), (Shape{1, 4, 8}));
    nearest.forward();
    EXPECT_EQ(nearest.output()[0], 0.0f);
    EXPECT_EQ(nearest.output()[3], 1.0f);   // x 3 -> input 1
    EXPECT_EQ(nearest.output()[3 * 8 + 7], 7.0f);

    resize2d area(ResizeSpec{1, 2, Interpolation::Area});
    area.join(source);
    area.forward();
    EXPECT_FLOAT_EQ(area.output()[0], (0 + 1 + 4 + 5) / 4.0f);
    EXPECT_FLOAT_EQ(area.output()[1], (2 + 3 + 6 + 7) / 4.0f);

    resize2d same(ResizeSpec{2, 4, Interpolation::Bilinear});
    same.join(source);
    same.forward();
    for (size_t i = 0; i < 8; ++i)
        EXPECT_FLOAT_EQ(same.output()[i], static_cast<float>(i));

    resize2d wide(ResizeSpec{2, 8, Interpolation::Bilinear});
    wide.join(source);
    wide.forward();
    EXPECT_FLOAT_EQ(wide.output()[0], 0.0f);   // clamped at the edge
    EXPECT_FLOAT_EQ(wide.output()[1], 0.25f);  // centre 1.5 / 2 - 0.5 = 0.25
    EXPECT_FLOAT_EQ(wide.output()[7], 3.0f);

    EXPECT_THROW(resize2d(ResizeSpec{0, 3}), std::invalid_argument);
}

TEST(Resize2DTest, BringsDifferentSizesToOneGrid)
{
    // A 6 x 8 camera and a 16-band spectrogram (1 x 16 x 4) read together by one conv layer.
    network net;
    const auto eye = net.addLayer("eye", LayerSpec::Retina({{1, 6, 8}}, false, false));
    const auto ear = net.addLayer("ear", LayerSpec::Retina({{2, 16, 4}}, false, false));
    const auto eye4 = net.addLayer("eye4", LayerSpec::Resize2D(4, 4, Interpolation::Area));
    const auto ear4 = net.addLayer("ear4", LayerSpec::Resize2D(4, 4));
    const auto fuse = net.addLayer("fuse", LayerSpec::Conv2D(2, Window2D::square(3, 1, 1), false, false));
    net.addInputs(eye, Shape{1, 6, 8}, "camera");
    net.addInputs(ear, Shape{2, 16, 4}, "spectrogram");
    net.connect(eye, eye4);
    net.connect(ear, ear4);
    net.connect(eye4, fuse);
    net.connect(ear4, fuse);
    EXPECT_EQ(net.getLayer(fuse).shape(), (Shape{2, 4, 4}));
    std::vector<float> ones(48, 1.0f);
    net.setInputs("camera", ones);
    net.step();
    for (float v : net.getLayer(eye4).output())
        EXPECT_FLOAT_EQ(v, 1.0f);

    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_EQ(loaded->layerSpec(ear4).resize, (ResizeSpec{4, 4, Interpolation::Bilinear}));
    EXPECT_EQ(loaded->layerSpec(eye4).resize.interpolation, Interpolation::Area);
    loaded->step();
    net.step();
    for (size_t i = 0; i < net.getLayer(fuse).size(); ++i)
        EXPECT_TRUE(sameBits(net.getLayer(fuse).output()[i], loaded->getLayer(fuse).output()[i]));
}

// =============================================================================
// Disparity
// =============================================================================
namespace {

// A random-dot pair at disparity `shift`: right[x] = left[x + shift], so left
// pixel x matches right pixel x - shift.
std::pair<std::vector<float>, std::vector<float>> stereoPair(size_t H, size_t W, int shift, std::uint32_t seed)
{
    std::vector<float> left(H * W), right(H * W);
    std::uint32_t state = seed;
    auto dot = [&state] {
        state = state * 1664525u + 1013904223u;
        return (state >> 16) & 1 ? 1.0f : -1.0f;
    };
    const size_t wide = W + 16;
    std::vector<float> scene(H * wide);
    for (float& v : scene)
        v = dot();
    for (size_t y = 0; y < H; ++y)
        for (size_t x = 0; x < W; ++x) {
            left[y * W + x] = scene[y * wide + x + 8];
            right[y * W + x] = scene[y * wide + static_cast<size_t>(static_cast<int>(x) + 8 + shift)];
        }
    return {left, right};
}

size_t bestDisparity(const layer& d, size_t count, size_t H, size_t W, size_t y, size_t x, bool lowest)
{
    size_t best = 0;
    for (size_t k = 1; k < count; ++k) {
        const float v = d.output()[k * H * W + y * W + x], b = d.output()[best * H * W + y * W + x];
        if (lowest ? v < b : v > b)
            best = k;
    }
    return best;
}

} // namespace

TEST(DisparityTest, FindsTheShiftOfARandomDotPair)
{
    const size_t H = 8, W = 16;
    for (DisparityMeasure measure :
         {DisparityMeasure::Correlation, DisparityMeasure::Difference, DisparityMeasure::Normalized}) {
        for (int shift : {0, 2, 3}) {
            auto [l, r] = stereoPair(H, W, shift, 7);
            fixed_map left(Shape{1, H, W}), right(Shape{1, H, W});
            left.set(l);
            right.set(r);
            disparity d(DisparitySpec{0, 4, 3, measure});
            d.join(left);
            d.join(right);
            ASSERT_EQ(d.shape(), (Shape{5, H, W}));
            d.forward();
            const bool lowest = measure == DisparityMeasure::Difference;
            EXPECT_EQ(bestDisparity(d, 5, H, W, 4, 10, lowest), static_cast<size_t>(shift))
                << "measure " << static_cast<int>(measure);
            if (measure == DisparityMeasure::Normalized) {
                EXPECT_NEAR(d.output()[static_cast<size_t>(shift) * H * W + 4 * W + 10], 1.0f, 1e-5f);
            }
        }
    }
}

TEST(DisparityTest, NegativeDisparitiesAndChannels)
{
    const size_t H = 4, W = 12;
    auto [l0, r0] = stereoPair(H, W, -2, 11);  // far: left x matches right x + 2
    auto [l1, r1] = stereoPair(H, W, -2, 12);
    std::vector<float> l = l0, r = r0;
    l.insert(l.end(), l1.begin(), l1.end());
    r.insert(r.end(), r1.begin(), r1.end());
    fixed_map left(Shape{2, H, W}), right(Shape{2, H, W});
    left.set(l);
    right.set(r);
    disparity d(DisparitySpec{-3, 1, 1, DisparityMeasure::Correlation});
    d.join(left);
    d.join(right);
    d.forward();
    // Window 1: exactly 1 (mean of +-1 * +-1 over two channels) at the true shift -2 (channel 1).
    for (size_t x = 0; x < W - 2; ++x)
        EXPECT_FLOAT_EQ(d.output()[1 * H * W + 2 * W + x], 1.0f) << x;
    // Outside the right view: 0.
    EXPECT_EQ(d.output()[0 * H * W + 0 * W + W - 1], 0.0f);
}

TEST(DisparityTest, RejectsBadSpecsAndViews)
{
    EXPECT_THROW(disparity(DisparitySpec{3, 2}), std::invalid_argument);
    EXPECT_THROW(disparity(DisparitySpec{0, 2, 2}), std::invalid_argument);
    fixed_map a(Shape{1, 4, 4}), b(Shape{1, 4, 5}), c(Shape{1, 4, 4}), e(Shape{1, 4, 4});
    disparity d(DisparitySpec{});
    d.forward();  // nothing joined: no-op
    d.join(a);
    EXPECT_THROW(d.join(b), std::invalid_argument);
    d.join(c);
    EXPECT_THROW(d.join(e), std::logic_error);
    c.grow(1);
    EXPECT_THROW(d.forward(), std::logic_error);
}

TEST(DisparityTest, StereoNetworkContinuesExactlyAfterLoad)
{
    reseed(21);
    network net;
    const auto left = net.addLayer("left", LayerSpec::Retina({{1, 6, 10}}, false, false));
    const auto right = net.addLayer("right", LayerSpec::Retina({{1, 6, 10}}, false, false));
    const auto depth = net.addLayer("depth", LayerSpec::Disparity({0, 3, 3, DisparityMeasure::Normalized}));
    const auto pool = net.addLayer("pool", LayerSpec::Pool2D(Window2D::square(2, 2), PoolMode::Average));
    const auto out = net.addLayer("out", LayerSpec::Dense(2, false, true));
    net.addInputs(left, Shape{1, 6, 10}, "left_eye");
    net.addInputs(right, Shape{1, 6, 10}, "right_eye");
    net.connect(left, depth);
    net.connect(right, depth);
    net.connect(depth, pool);
    net.connect(pool, out);
    net.addOutput(out);
    EXPECT_EQ(net.getLayer(depth).shape(), (Shape{4, 6, 10}));

    auto run = [](network& n, std::uint32_t from, size_t ticks) {
        std::vector<float> trace;
        for (std::uint32_t t = from; t < from + ticks; ++t) {
            auto [l, r] = stereoPair(6, 10, static_cast<int>(t % 4), t);
            n.setInputs("left_eye", l);
            n.setInputs("right_eye", r);
            n.step();
            const auto y = n.outputs();
            trace.insert(trace.end(), y.begin(), y.end());
            n.applyReward(y[0] > y[1] ? -1.0f : 1.0f, 0.02f);
        }
        return trace;
    };
    run(net, 0, 20);
    std::stringstream data;
    net.save(data);
    const auto original = run(net, 20, 20);
    reseed(1);
    auto loaded = network::load(data);
    EXPECT_EQ(loaded->layerSpec(depth).disparity, (DisparitySpec{0, 3, 3, DisparityMeasure::Normalized}));
    const auto continued = run(*loaded, 20, 20);
    ASSERT_EQ(original.size(), continued.size());
    for (size_t i = 0; i < original.size(); ++i)
        ASSERT_TRUE(sameBits(original[i], continued[i])) << i;
}
