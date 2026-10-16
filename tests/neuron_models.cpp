// Neuron models other than E-R (NeuronModel in neuron.hpp): rate units
// (sigmoid, tanh, softplus, with optional smoothing), leaky integrate-and-fire
// (with refractoriness and adaptation: ALIF), Izhikevich and resonate-and-fire
// neurons; their layer option, learning rules, derivatives and saving.
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <sstream>
#include <vector>
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"

using namespace exr;

namespace {

neuron withModel(const NeuronModel& model, bool habituation = false)
{
    neuron n(habituation, false);
    n.setModel(model);
    return n;
}

// The ticks (from 0) on which the neuron spikes for the inputs.
std::vector<int> spikes(neuron& n, const std::vector<float>& inputs)
{
    std::vector<int> out;
    for (size_t t = 0; t < inputs.size(); ++t)
        if (n.activate(inputs[t]) != 0.0f)
            out.push_back(static_cast<int>(t));
    return out;
}

int spikeCount(neuron& n, float input, int ticks)
{
    return static_cast<int>(spikes(n, std::vector<float>(ticks, input)).size());
}

} // namespace

// --- Rate units ---------------------------------------------------------------

TEST(NeuronModels, RateUnitsApplyTheirFunctionToTheScaledSum)
{
    neuron sig = withModel(NeuronModel::sigmoid());
    EXPECT_FLOAT_EQ(sig.activate(0.0f), 0.5f);
    EXPECT_FLOAT_EQ(sig.activate(2.0f), 1.0f / (1.0f + std::exp(-2.0f)));
    neuron steep = withModel(NeuronModel::sigmoid(4.0f));
    EXPECT_FLOAT_EQ(steep.activate(0.5f), 1.0f / (1.0f + std::exp(-2.0f)));

    neuron th = withModel(NeuronModel::tanh(2.0f));
    EXPECT_FLOAT_EQ(th.activate(-0.3f), std::tanh(-0.6f));

    neuron sp = withModel(NeuronModel::softplus(2.0f));
    EXPECT_NEAR(sp.activate(1.0f), std::log1p(std::exp(2.0f)) / 2.0f, 1e-6f);
    EXPECT_NEAR(sp.activate(10.0f), 10.0f, 1e-5f);  // ~ReLU far from 0
    EXPECT_GT(sp.activate(-10.0f), 0.0f);
}

TEST(NeuronModels, SmoothingMakesALeakyIntegrator)
{
    // y = m * y + (1 - m) * f(s): a step input approaches f(s) geometrically.
    neuron n = withModel(NeuronModel::tanh(1.0f, 0.5f));
    const float target = std::tanh(1.0f);
    float expected = 0.0f;
    for (int t = 0; t < 6; ++t) {
        expected = 0.5f * expected + 0.5f * target;
        EXPECT_FLOAT_EQ(n.activate(1.0f), expected);
    }
    EXPECT_FLOAT_EQ(n.membrane(), expected);
}

TEST(NeuronModels, RateUnitDerivativesMatchFiniteDifferences)
{
    for (const NeuronModel& model : {NeuronModel::sigmoid(1.5f), NeuronModel::tanh(0.7f), NeuronModel::softplus(2.0f),
                                     NeuronModel::tanh(1.0f, 0.3f)}) {
        for (float s : {-1.2f, 0.0f, 0.4f}) {
            neuron n = withModel(model);
            n.activate(0.5f);  // a non-zero previous output for smoothing
            neuron lo = n, hi = n;
            Derivatives d;
            n.activate(s, 0.5f, d);
            const float h = 1e-3f;
            const float numeric = (hi.activate(s + h) - lo.activate(s - h)) / (2 * h);
            EXPECT_NEAR(d.dyds, numeric, 1e-3f) << neuronModelName(model.type) << " at " << s;
            EXPECT_FLOAT_EQ(d.dydthr, model.smoothing);
            EXPECT_FLOAT_EQ(d.dthrdthr, model.smoothing);
            EXPECT_FLOAT_EQ(d.dthrds, d.dyds);
        }
    }
}

// --- Leaky integrate-and-fire --------------------------------------------------

TEST(NeuronModels, LifIntegratesLeaksAndResets)
{
    // v = 0.9 v + 0.3: 0.3, 0.57, 0.813, 1.0317 > 1 -> spike on tick 3, reset to 0.
    neuron n = withModel(NeuronModel::lif(0.9f, 1.0f));
    EXPECT_EQ(spikes(n, std::vector<float>(8, 0.3f)), (std::vector<int>{3, 7}));
    // Below the leak's fixed point (0.09 / 0.1 = 0.9 < 1) it never fires.
    neuron weak = withModel(NeuronModel::lif(0.9f, 1.0f));
    EXPECT_EQ(spikeCount(weak, 0.09f, 200), 0);
    EXPECT_NEAR(weak.membrane(), 0.9f, 1e-3f);
    EXPECT_NEAR(weak.charge(), 0.9f, 1e-3f);
    // Inhibition pushes the membrane below rest.
    neuron inhibited = withModel(NeuronModel::lif());
    inhibited.activate(-0.5f);
    EXPECT_FLOAT_EQ(inhibited.membrane(), -0.5f);
}

TEST(NeuronModels, LifRefractoryTicksIgnoreTheInput)
{
    // Threshold 0.5 with input 1: a spike every tick without refractoriness,
    // every third with two refractory ticks.
    neuron plain = withModel(NeuronModel::lif(0.9f, 0.5f));
    EXPECT_EQ(spikeCount(plain, 1.0f, 9), 9);
    neuron refractory = withModel(NeuronModel::lif(0.9f, 0.5f, 2, -0.1f));
    EXPECT_EQ(spikes(refractory, std::vector<float>(9, 1.0f)), (std::vector<int>{0, 3, 6}));
    refractory.activate(1.0f);  // tick 9: refractory, held at reset
    EXPECT_FLOAT_EQ(refractory.membrane(), -0.1f);
}

TEST(NeuronModels, AdaptiveLifSlowsDownUnderConstantInput)
{
    const std::vector<float> drive(400, 0.5f);
    neuron lif = withModel(NeuronModel::lif(0.9f, 1.0f));
    neuron alif = withModel(NeuronModel::alif(0.9f, 1.0f, 0.5f, 0.98f));
    const auto plain = spikes(lif, drive);
    const auto adapted = spikes(alif, drive);
    ASSERT_GE(adapted.size(), 3u);
    EXPECT_LT(adapted.size(), plain.size());
    // The first spike comes when LIF's does; then intervals grow (spike-frequency adaptation).
    EXPECT_EQ(adapted[0], plain[0]);
    EXPECT_GT(adapted[1] - adapted[0], plain[1] - plain[0]);
    EXPECT_GT(adapted.back() - adapted[adapted.size() - 2], adapted[1] - adapted[0]);
    EXPECT_GT(alif.recoveryVariable(), 0.0f);
}

// --- Izhikevich -----------------------------------------------------------------

TEST(NeuronModels, IzhikevichRestsWithoutInputAndSpikesWithIt)
{
    neuron n = withModel(NeuronModel::regularSpiking());
    EXPECT_EQ(spikeCount(n, 0.0f, 200), 0);
    EXPECT_NEAR(n.membrane(), -70.0f, 1.0f);  // regular spiking rests near -70 mV
    const int regular = spikeCount(n, 1.0f, 1000);  // I = 10
    EXPECT_GT(regular, 5);
    neuron fast = withModel(NeuronModel::fastSpiking());
    EXPECT_GT(spikeCount(fast, 1.0f, 1000), regular);
    // Strong inhibition keeps it finite and silent.
    neuron inhibited = withModel(NeuronModel::regularSpiking());
    EXPECT_EQ(spikeCount(inhibited, -10.0f, 100), 0);
    EXPECT_TRUE(std::isfinite(inhibited.membrane()));
}

TEST(NeuronModels, IzhikevichPresetsDifferInFiringPattern)
{
    // Chattering fires bursts: some inter-spike intervals are much shorter
    // than others; regular spiking does not.
    const auto ratio = [](const NeuronModel& model) {
        neuron n = withModel(model);
        const auto s = spikes(n, std::vector<float>(1000, 1.0f));
        int lo = 1 << 30, hi = 0;
        for (size_t i = 1; i < s.size(); ++i) {
            lo = std::min(lo, s[i] - s[i - 1]);
            hi = std::max(hi, s[i] - s[i - 1]);
        }
        return static_cast<float>(hi) / static_cast<float>(lo);
    };
    EXPECT_GT(ratio(NeuronModel::chattering()), 3.0f * ratio(NeuronModel::regularSpiking()));
}

// --- Resonate-and-fire -------------------------------------------------------------

TEST(NeuronModels, ResonatorPrefersInputAtItsFrequency)
{
    const float frequency = 0.1f;
    const auto drive = [](float f) {
        std::vector<float> x(400);
        for (size_t t = 0; t < x.size(); ++t)
            x[t] = 0.3f * std::sin(2.0f * std::numbers::pi_v<float> * f * static_cast<float>(t));
        return x;
    };
    neuron tuned = withModel(NeuronModel::resonator(frequency, 0.95f, 1.0f));
    neuron slow = withModel(NeuronModel::resonator(frequency, 0.95f, 1.0f));
    neuron fast = withModel(NeuronModel::resonator(frequency, 0.95f, 1.0f));
    neuron constant = withModel(NeuronModel::resonator(frequency, 0.95f, 1.0f));
    const size_t at = spikes(tuned, drive(frequency)).size();
    EXPECT_GT(at, 10u);
    EXPECT_LT(spikes(slow, drive(0.02f)).size(), at / 4);
    EXPECT_LT(spikes(fast, drive(0.3f)).size(), at / 4);
    EXPECT_EQ(spikeCount(constant, 0.3f, 400), 0);
}

// --- Habituation still applies to the sum -------------------------------------------

TEST(NeuronModels, HabituationSuppressesARepeatedSumBeforeTheModel)
{
    neuron n(true, false);
    n.setHabituation({.steps = 3});
    n.setModel(NeuronModel::sigmoid());
    EXPECT_GT(n.activate(2.0f), 0.8f);
    n.activate(2.0f);
    n.activate(2.0f);
    EXPECT_FLOAT_EQ(n.activate(2.0f), 0.5f);  // third repeat cut to 0: sigmoid(0)
}

// --- Layers -------------------------------------------------------------------------

TEST(NeuronModels, LayerOptionValidation)
{
    network net;
    EXPECT_THROW(net.addLayer("er", [] {
        LayerSpec s = LayerSpec::Dense(4);
        s.neuronModel = NeuronModel::lif();
        return s;
    }()), std::invalid_argument);
    EXPECT_THROW(net.addLayer("gated", [] {
        LayerSpec s = LayerSpec::Dense(4, NeuronModel::tanh());
        s.gate = 0.5f;
        return s;
    }()), std::invalid_argument);
    EXPECT_THROW(net.addLayer("bad", LayerSpec::Dense(4, NeuronModel::lif(1.0f))), std::invalid_argument);
    const auto id = net.addLayer("ok", LayerSpec::Dense(4, NeuronModel::alif()));
    auto& layer = net.layerAs<dense>(id);
    EXPECT_EQ(layer.neuronModel(), NeuronModel::alif());
    EXPECT_THROW(layer.setRectified(true), std::invalid_argument);
    for (const neuron& n : layer.neurons())
        EXPECT_EQ(n.model(), NeuronModel::alif());
    // Grown neurons get the layer's model.
    net.addInputs(id, 2);
    net.growLayer(id, 2);
    EXPECT_EQ(layer.neurons().back().model(), NeuronModel::alif());
}

TEST(NeuronModels, SummaryNamesTheModel)
{
    network net;
    net.addLayer("lif", LayerSpec::Dense(3, NeuronModel::lif()));
    net.addLayer("izh", LayerSpec::Dense(3, NeuronModel::chattering()));
    std::ostringstream os;
    net.describe(os);
    EXPECT_NE(os.str().find("lif"), std::string::npos);
    EXPECT_NE(os.str().find("izh"), std::string::npos);
}

TEST(NeuronModels, SaveAndLoadKeepTheModelAndItsState)
{
    for (const NeuronModel& model : {NeuronModel::tanh(1.5f, 0.4f), NeuronModel::alif(0.8f, 0.7f, 0.3f, 0.9f, 1),
                                     NeuronModel::intrinsicallyBursting(), NeuronModel::resonator(0.15f, 0.9f, 0.8f, 2)}) {
        network net;
        const auto in = net.addLayer("in", LayerSpec::Dense(6, model));
        const auto out = net.addLayer("out", LayerSpec::Dense(2, NeuronModel::sigmoid()));
        net.addInputs(in, 3);
        net.connect(in, out);
        net.addOutput(out);
        const auto run = [](network& n, int t0, int ticks) {
            std::vector<float> trace;
            for (int t = t0; t < t0 + ticks; ++t) {
                n.setInputs({std::sin(0.3f * t), std::cos(0.7f * t), 0.5f});
                n.step();
                for (float y : n.layerAs<dense>(0).output())
                    trace.push_back(y);
                for (float y : n.outputs())
                    trace.push_back(y);
            }
            return trace;
        };
        run(net, 0, 17);  // mid-integration
        std::stringstream ss;
        net.save(ss);
        const auto restored = network::load(ss, DeserializeMode::FullState);
        EXPECT_EQ(restored->layerAs<dense>(in).neuronModel(), model);
        EXPECT_EQ(run(*restored, 17, 40), run(net, 17, 40)) << neuronModelName(model.type);
    }
}

TEST(NeuronModels, CombineWithPositiveValuesAnd8Bit)
{
    // A positive-only layer cuts tanh at 0; 8-bit values put outputs on the
    // value grid; both survive saving next to the model.
    for (const NeuronModel& model : {NeuronModel::tanh(), NeuronModel::lif(0.8f, 0.5f)}) {
        network net;
        LayerSpec spec = LayerSpec::Dense(6, model).positive();
        spec.precision = Precision::Int8();
        const auto id = net.addLayer("h", spec);
        net.addInputs(id, 2);
        net.addOutput(id);
        for (int t = 0; t < 30; ++t) {
            net.setInputs({std::sin(0.4f * t), -std::cos(0.9f * t)});
            net.step();
            for (float y : net.outputs()) {
                EXPECT_GE(y, 0.0f) << neuronModelName(model.type);
                EXPECT_FLOAT_EQ(y * 16.0f, std::round(y * 16.0f)) << neuronModelName(model.type);
            }
        }
        std::stringstream ss;
        net.save(ss);
        const auto restored = network::load(ss, DeserializeMode::FullState);
        const LayerSpec& back = restored->layerSpec(id);
        EXPECT_EQ(back.neuronModel, model);
        EXPECT_TRUE(back.positiveValues);
        EXPECT_EQ(back.precision, Precision::Int8());
    }
}

TEST(NeuronModels, WeightsOnlyLoadPutsTheModelAtRest)
{
    network net;
    const auto id = net.addLayer("n", LayerSpec::Dense(1, NeuronModel::regularSpiking()));
    net.addInputs(id, 1);
    net.layerAs<dense>(id).setWeights(0, {1.0f});
    net.setInputs({1.0f});
    for (int t = 0; t < 7; ++t)
        net.step();
    std::stringstream ss;
    net.save(ss);
    const auto restored = network::load(ss, DeserializeMode::WeightsOnly);
    EXPECT_FLOAT_EQ(restored->layerAs<dense>(id).neurons()[0].membrane(), -65.0f);
}

// --- Learning ----------------------------------------------------------------------

namespace {

// Mean squared error over the last `window` of `ticks` ticks of a one-output
// regression y = tanh(0.8 x1 - 0.5 x2) with a hidden layer of `hidden`, learning
// with `rule` (applyError every tick).
float regressionError(const LayerSpec& hidden, const LearningRule& rule, float rate, int ticks)
{
    network net;
    LayerSpec h = hidden;
    h.learningRule = rule;
    h.normalize = false;
    LayerSpec o = LayerSpec::Dense(1, NeuronModel::tanh());
    o.learningRule = rule;
    o.normalize = false;
    const auto hid = net.addLayer("h", h);
    const auto out = net.addLayer("o", o);
    net.addInputs(hid, 2);
    net.connect(hid, out);
    net.addOutput(out);
    std::uint32_t state = 12345;
    const auto uniform = [&state] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    };
    float error = 0.0f;
    const int window = ticks / 5;
    for (int t = 0; t < ticks; ++t) {
        const float x1 = uniform(), x2 = uniform();
        net.setInputs({x1, x2});
        net.step();
        const float e = std::tanh(0.8f * x1 - 0.5f * x2) - net.outputs()[0];
        if (t >= ticks - window)
            error += e * e / static_cast<float>(window);
        const float errors[] = {e};
        net.applyError(errors, rate);
    }
    return error;
}

} // namespace

TEST(NeuronModels, RateUnitsLearnWithFeedbackAlignmentAndSurrogateGradients)
{
    for (const LearningRule& rule : {LearningRule::feedbackAlignment(), LearningRule::surrogate(1)}) {
        const float before = regressionError(LayerSpec::Dense(8, NeuronModel::tanh()), rule, 0.0f, 3000);
        const float after = regressionError(LayerSpec::Dense(8, NeuronModel::tanh()), rule, 0.02f, 3000);
        EXPECT_LT(after, 0.3f * before) << learningRuleName(rule.type) << ": " << before << " -> " << after;
    }
}

TEST(NeuronModels, SpikingLayersTakeEveryRuleAndStayFinite)
{
    for (const NeuronModel& model : {NeuronModel::lif(0.8f, 0.5f), NeuronModel::alif(0.8f, 0.5f),
                                     NeuronModel::regularSpiking(), NeuronModel::resonator(0.1f, 0.9f, 0.5f)}) {
        for (const LearningRule& rule :
             {LearningRule::sign(), LearningRule::traced(0.5f), LearningRule::feedbackAlignment(0.5f),
              LearningRule::eligibility(), LearningRule::eprop(0.5f), LearningRule::surrogate(4)}) {
            const float e = regressionError(LayerSpec::Dense(8, model), rule, 0.01f, 400);
            EXPECT_TRUE(std::isfinite(e)) << neuronModelName(model.type) << " " << learningRuleName(rule.type);
        }
    }
}

TEST(NeuronModels, EPropTracesFollowTheLifMembrane)
{
    // e-prop's eligibility for LIF is psi * (x + leak * a), a = leak * a + x
    // (Bellec et al. 2020): the layer's derivatives carry exactly that.
    network net;
    LayerSpec spec = LayerSpec::Dense(1, NeuronModel::lif(0.8f, 1.0f));
    spec.learningRule = LearningRule::eprop();
    spec.normalize = false;
    const auto id = net.addLayer("n", spec);
    net.addInputs(id, 1);
    net.layerAs<dense>(id).setWeights(0, {1.0f});
    net.setInputs({0.7f});
    net.step();
    const Derivatives d = net.layerAs<dense>(id).derivatives(0);
    EXPECT_FLOAT_EQ(d.dthrdthr, 0.8f);
    EXPECT_FLOAT_EQ(d.dthrds, 1.0f);
    EXPECT_FLOAT_EQ(d.dydthr, 0.8f * d.dyds);
    EXPECT_FLOAT_EQ(d.dyds, (1.0f - 0.3f / 0.5f) / 0.5f);  // the bump at v = 0.7, half-width 0.5
}
