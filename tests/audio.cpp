// Audio layers: Cochlea (spectrum into frequency bands) against a direct
// double-precision DFT, and History (the last ticks side by side).
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <sstream>
#include <vector>
#include "../core/network.hpp"
#include "../core/layers/cochlea.hpp"
#include "../core/layers/dense.hpp"
#include "../core/layers/history.hpp"
#include "../core/layers/retina.hpp"

using namespace exr;

namespace {

bool sameBits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

std::vector<float> tone(double frequency, double sampleRate, size_t from, size_t count, double amplitude = 0.5)
{
    std::vector<float> s(count);
    for (size_t n = 0; n < count; ++n)
        s[n] = static_cast<float>(amplitude * std::sin(2.0 * std::numbers::pi * frequency *
                                                       static_cast<double>(from + n) / sampleRate));
    return s;
}

// A cochlea passing its band values through (no habituation, no E-R).
CochleaSpec smallSpec()
{
    CochleaSpec spec;
    spec.sampleRate = 8000.0f;
    spec.hop = 64;
    spec.window = 256;
    spec.bands = 16;
    spec.minFrequency = 100.0f;
    spec.compression = Compression::Linear;
    spec.gain = 1.0f;
    return spec;
}

// Feeds `signal` hop by hop; returns the band values after the last hop.
std::vector<float> listen(cochlea& ear, const std::vector<float>& signal)
{
    const size_t hop = ear.spec().hop;
    std::vector<float> sensors(hop);
    ear.attachInputs(sensors);
    for (size_t at = 0; at + hop <= signal.size(); at += hop) {
        std::copy_n(signal.begin() + static_cast<std::ptrdiff_t>(at), hop, sensors.begin());
        ear.forward();
    }
    return {ear.output().begin(), ear.output().end()};
}

size_t loudest(const std::vector<float>& bands)
{
    return static_cast<size_t>(std::ranges::max_element(bands) - bands.begin());
}

} // namespace

// =============================================================================
// Cochlea
// =============================================================================
TEST(CochleaTest, BandEnergiesMatchADirectDft)
{
    const CochleaSpec spec = smallSpec();
    cochlea ear(spec, false, false);
    ASSERT_EQ(ear.shape(), (Shape{1, 16, 1}));

    // Two tones and a little noise-like wobble, longer than one window.
    std::vector<float> signal = tone(440.0, spec.sampleRate, 0, 640);
    const std::vector<float> second = tone(1900.0, spec.sampleRate, 0, 640, 0.2);
    for (size_t n = 0; n < signal.size(); ++n)
        signal[n] += second[n] + 0.05f * static_cast<float>(std::sin(0.37 * static_cast<double>(n * n % 101)));
    const std::vector<float> bands = listen(ear, signal);

    // The reference: the last `window` samples, Hann window, DFT in double.
    const size_t N = spec.window;
    const std::vector<float> last(signal.end() - static_cast<std::ptrdiff_t>(N), signal.end());
    ASSERT_EQ(ear.samples(), last);
    std::vector<double> w(N);
    double sum = 0.0;
    for (size_t n = 0; n < N; ++n)
        sum += w[n] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) / static_cast<double>(N));
    std::vector<double> power(N / 2 + 1);
    for (size_t k = 0; k < power.size(); ++k) {
        std::complex<double> x = 0.0;
        for (size_t n = 0; n < N; ++n)
            x += static_cast<double>(last[n]) * w[n] *
                 std::polar(1.0, -2.0 * std::numbers::pi * static_cast<double>(k * n) / static_cast<double>(N));
        power[k] = std::norm(x) * (2.0 / sum) * (2.0 / sum);
        EXPECT_NEAR(ear.power()[k], power[k], 1e-4 + 1e-4 * power[k]) << "bin " << k;
    }
    for (size_t b = 0; b < bands.size(); ++b) {
        const cochlea::Band& band = ear.bands()[b];
        double energy = 0.0;
        for (size_t i = 0; i < band.weights.size(); ++i)
            energy += band.weights[i] * power[band.firstBin + i];
        EXPECT_NEAR(bands[b], energy, 1e-4 + 1e-4 * energy) << "band " << b;
    }
}

TEST(CochleaTest, BandsAreOrderedTrianglesOnTheMelScale)
{
    cochlea ear(smallSpec(), false, false);
    const auto& bands = ear.bands();
    ASSERT_EQ(bands.size(), 16u);
    EXPECT_FLOAT_EQ(bands.front().lowFrequency, 100.0f);
    EXPECT_NEAR(bands.back().highFrequency, 4000.0f, 1e-2f);  // maxFrequency 0 = Nyquist
    for (size_t b = 0; b + 1 < bands.size(); ++b) {
        EXPECT_LT(bands[b].centreFrequency, bands[b + 1].centreFrequency);
        EXPECT_FLOAT_EQ(bands[b].centreFrequency, bands[b + 1].lowFrequency);  // half overlap
        // Mel: wider towards the top.
        EXPECT_LT(bands[b].highFrequency - bands[b].lowFrequency,
                  bands[b + 1].highFrequency - bands[b + 1].lowFrequency + 1e-3f);
        EXPECT_FALSE(bands[b].weights.empty());
        for (float v : bands[b].weights)
            EXPECT_TRUE(v >= 0.0f && v <= 1.0f);
    }

    CochleaSpec linear = smallSpec();
    linear.scale = FrequencyScale::Linear;
    cochlea flat(linear, false, false);
    const float width = flat.bands()[0].highFrequency - flat.bands()[0].lowFrequency;
    for (const auto& band : flat.bands())
        EXPECT_NEAR(band.highFrequency - band.lowFrequency, width, 1e-2f);
}

TEST(CochleaTest, ATonePeaksInItsBand)
{
    const CochleaSpec spec = smallSpec();
    for (double f : {200.0, 700.0, 1500.0, 3000.0}) {
        cochlea ear(spec, false, false);
        const std::vector<float> bands = listen(ear, tone(f, spec.sampleRate, 0, 1024));
        const cochlea::Band& peak = ear.bands()[loudest(bands)];
        EXPECT_GE(f, peak.lowFrequency) << f << " Hz";
        EXPECT_LE(f, peak.highFrequency) << f << " Hz";
    }
    // Power normalisation: a bin-centred sine of amplitude A has power A^2.
    cochlea ear(spec, false, false);
    const double bin = spec.sampleRate / static_cast<double>(spec.window);
    listen(ear, tone(20 * bin, spec.sampleRate, 0, 1024, 0.5));
    EXPECT_NEAR(ear.power()[20], 0.25, 1e-3);
}

TEST(CochleaTest, SilenceIsZeroAndLogCompressionIsMonotonic)
{
    CochleaSpec spec = smallSpec();
    spec.compression = Compression::Log;
    spec.gain = 100.0f;
    cochlea quiet(spec, false, false);
    for (float v : listen(quiet, std::vector<float>(512, 0.0f)))
        EXPECT_EQ(v, 0.0f);

    cochlea soft(spec, false, false), loud(spec, false, false);
    const size_t b = loudest(listen(soft, tone(1000.0, spec.sampleRate, 0, 512, 0.05)));
    const std::vector<float> l = listen(loud, tone(1000.0, spec.sampleRate, 0, 512, 0.5));
    EXPECT_GT(l[b], soft.output()[b]);
    const cochlea::Band& band = loud.bands()[b];
    double energy = 0.0;
    for (size_t i = 0; i < band.weights.size(); ++i)
        energy += band.weights[i] * loud.power()[band.firstBin + i];
    EXPECT_NEAR(l[b], std::log1p(100.0 * energy), 1e-4);
}

TEST(CochleaTest, HabituationFadesASteadyTone)
{
    CochleaSpec spec = smallSpec();
    spec.compression = Compression::Log;
    cochlea ear(spec, true, true);
    std::vector<float> sensors(spec.hop);
    ear.attachInputs(sensors);
    std::vector<float> trace;
    for (size_t t = 0; t < 200; ++t) {
        sensors = tone(1000.0, spec.sampleRate, t * spec.hop, spec.hop);
        ear.forward();
        trace.push_back(*std::ranges::max_element(ear.output()));
    }
    const float peak = *std::ranges::max_element(trace);
    EXPECT_GT(peak, 0.0f);
    EXPECT_LT(trace.back(), peak);  // the steady tone fades
}

TEST(CochleaTest, RejectsBadSpecsAndInputs)
{
    auto with = [](auto change) {
        CochleaSpec spec = smallSpec();
        change(spec);
        return spec;
    };
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.window = 300; })), std::invalid_argument);
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.hop = 512; })), std::invalid_argument);
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.hop = 0; })), std::invalid_argument);
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.bands = 0; })), std::invalid_argument);
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.maxFrequency = 5000.0f; })), std::invalid_argument);
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.minFrequency = 4000.0f; })), std::invalid_argument);
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.gain = 0.0f; })), std::invalid_argument);
    EXPECT_THROW(cochlea(with([](CochleaSpec& s) { s.sampleRate = 0.0f; })), std::invalid_argument);

    cochlea ear(smallSpec(), false, false);
    std::vector<float> wrong(63), right(64);
    EXPECT_THROW(ear.attachInputs(wrong), std::invalid_argument);
    ear.attachInputs(right);
    EXPECT_THROW(ear.attachInputs(right), std::logic_error);
    retina other(RetinaSpec{{1, 2, 2}}, false, false);
    EXPECT_THROW(ear.join(other), std::logic_error);
}

// =============================================================================
// History
// =============================================================================
TEST(HistoryTest, ShiftsTheNewestTickInLast)
{
    retina a(RetinaSpec{{2, 3, 1}}, false, false);
    std::vector<float> in(6);
    a.attachInputs(in);
    history past(4);
    past.join(a);
    ASSERT_EQ(past.shape(), (Shape{2, 3, 4}));
    for (int t = 1; t <= 5; ++t) {
        for (size_t i = 0; i < 6; ++i)
            in[i] = static_cast<float>(t) + 0.1f * static_cast<float>(i);
        a.forward();
        past.forward();
    }
    // Row r (channel c, height y) holds ticks 2, 3, 4, 5, oldest first.
    for (size_t r = 0; r < 6; ++r)
        for (size_t col = 0; col < 4; ++col)
            EXPECT_FLOAT_EQ(past.output()[r * 4 + col], static_cast<float>(col + 2) + 0.1f * static_cast<float>(r))
                << "row " << r << " column " << col;
    EXPECT_THROW(history(0), std::invalid_argument);
}

TEST(HistoryTest, KeepsWideSourcesAndJoinsMore)
{
    retina a(RetinaSpec{{1, 2, 2}}, false, false), b(RetinaSpec{{1, 2, 2}}, false, false);
    std::vector<float> ia{1, 2, 3, 4}, ib{5, 6, 7, 8};
    a.attachInputs(ia);
    b.attachInputs(ib);
    history past(3);
    past.join(a);
    ASSERT_EQ(past.shape(), (Shape{1, 2, 6}));
    a.forward();
    past.forward();
    EXPECT_EQ(std::vector<float>(past.output().begin(), past.output().end()),
              (std::vector<float>{0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 3, 4}));

    // A second source is one more channel; a reader is extended.
    dense reader(1);
    reader.join(past);
    past.join(b);
    EXPECT_EQ(past.shape(), (Shape{2, 2, 6}));
    EXPECT_EQ(reader.inputCount(0), 24u);
    a.forward();
    b.forward();
    past.forward();
    // The old channel keeps its past; the new one starts from zeros.
    EXPECT_EQ(std::vector<float>(past.output().begin(), past.output().end()),
              (std::vector<float>{0, 0, 1, 2, 1, 2, 0, 0, 3, 4, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 7, 8}));
}

TEST(HistoryTest, FlatSourceGrowthAddsChannels)
{
    dense source(2, false, false), feeder(1);
    std::vector<float> sensor(1, 0.5f);
    source.attachInputs(sensor);
    history past(4);
    past.join(source);
    EXPECT_EQ(past.shape(), (Shape{2, 1, 4}));
    source.forward();
    past.forward();
    const std::vector<float> before(past.output().begin(), past.output().end());
    feeder.addFeedback(source, 1);  // source: 2 -> 3 outputs
    EXPECT_EQ(past.shape(), (Shape{3, 1, 4}));
    for (size_t i = 0; i < before.size(); ++i)
        EXPECT_EQ(past.output()[i], before[i]);
    for (size_t i = before.size(); i < 12; ++i)
        EXPECT_EQ(past.output()[i], 0.0f);
    source.forward();
    EXPECT_NO_THROW(past.forward());
}

// =============================================================================
// A hearing pipeline in a network
// =============================================================================
namespace {

CochleaSpec networkSpec()
{
    CochleaSpec spec;
    spec.sampleRate = 8000.0f;
    spec.hop = 80;
    spec.window = 256;
    spec.bands = 12;
    spec.maxFrequency = 3500.0f;
    spec.scale = FrequencyScale::Linear;
    spec.gain = 30.0f;
    return spec;
}

std::unique_ptr<network> buildHearing()
{
    auto net = std::make_unique<network>();
    const CochleaSpec spec = networkSpec();
    const auto ear = net->addLayer("ear", LayerSpec::Cochlea(spec));
    const auto past = net->addLayer("past", LayerSpec::History(8));
    const auto v1 = net->addLayer("v1", LayerSpec::Conv2D(3, Window2D::square(3, 1, 1), false, true));
    const auto out = net->addLayer("out", LayerSpec::Dense(2, false, false));
    net->addInputs(ear, spec.hop);
    net->connect(ear, past);
    net->connect(past, v1);
    net->connect(v1, out);
    net->addOutput(out);
    return net;
}

std::vector<float> run(network& net, size_t from, size_t ticks)
{
    std::vector<float> trace;
    for (size_t t = from; t < from + ticks; ++t) {
        // A tone gliding up.
        std::vector<float> sound(80);
        for (size_t n = 0; n < 80; ++n) {
            const double time = static_cast<double>(t * 80 + n) / 8000.0;
            sound[n] = static_cast<float>(0.4 * std::sin(2.0 * std::numbers::pi * (300.0 + 400.0 * time) * time));
        }
        net.setInputs(sound);
        net.step();
        const std::vector<float> y = net.outputs();
        trace.insert(trace.end(), y.begin(), y.end());
        net.applyReward(y[0] > 0 ? -1.0f : 1.0f, 0.01f);
    }
    return trace;
}

} // namespace

TEST(HearingNetworkTest, BuildsRunsAndContinuesExactlyAfterLoad)
{
    reseed(5);
    auto net = buildHearing();
    EXPECT_EQ(net->getLayer(0).shape(), (Shape{1, 12, 1}));
    EXPECT_EQ(net->getLayer(1).shape(), (Shape{1, 12, 8}));
    EXPECT_EQ(net->getLayer(2).shape(), (Shape{3, 12, 8}));
    std::ostringstream text;
    net->describe(text);
    EXPECT_NE(text.str().find("1x12x8"), std::string::npos) << text.str();

    run(*net, 0, 40);
    std::stringstream data;
    net->save(data);
    const std::vector<float> original = run(*net, 40, 30);

    reseed(999);
    auto restored = network::load(data);
    EXPECT_EQ(restored->layerAs<cochlea>(0).spec(), networkSpec());
    EXPECT_EQ(restored->layerAs<history>(1).length(), 8u);
    const std::vector<float> continued = run(*restored, 40, 30);
    ASSERT_EQ(original.size(), continued.size());
    for (size_t i = 0; i < original.size(); ++i)
        ASSERT_TRUE(sameBits(original[i], continued[i])) << "value " << i;
}
