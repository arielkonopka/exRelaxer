// Positive-only values and weights (ValueRange, LayerSpec::positive*): the
// neuron's range, what a layer keeps at or above 0 under every learning rule,
// folding, hand-set weights, saving, and that the default is unchanged.
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "../core/layers/conv2d.hpp"
#include "../core/layers/dense.hpp"
#include "../core/layers/locally_connected2d.hpp"
#include "../core/network.hpp"
#include "../core/random.hpp"

using namespace exr;

namespace {

// Every weight (and bias) of a layer with weights.
std::vector<float> allWeights(const neuron_layer& layer)
{
    std::vector<float> out;
    if (const auto* c = dynamic_cast<const conv2d*>(&layer)) {
        for (size_t k = 0; k < c->outputChannels(); ++k)
            for (float w : c->kernel(k))
                out.push_back(w);
    } else if (const auto* d = dynamic_cast<const dense*>(&layer)) {
        for (size_t i = 0; i < d->size(); ++i)
            for (float w : d->weights(i))
                out.push_back(w);
    } else if (const auto* l = dynamic_cast<const locally_connected2d*>(&layer)) {
        for (size_t i = 0; i < l->size(); ++i)
            for (float w : l->weights(i))
                out.push_back(w);
    }
    for (size_t i = 0; i < layer.size(); ++i)
        out.push_back(layer.bias(i));
    return out;
}

float smallest(const std::vector<float>& v)
{
    float m = std::numeric_limits<float>::infinity();
    for (float x : v)
        m = std::min(m, x);
    return m;
}

} // namespace

TEST(PositiveTest, NeuronRange)
{
    neuron linear(false, false);
    EXPECT_EQ(linear.activate(-3.0f), -3.0f);  // the default keeps negative values
    EXPECT_EQ(linear.activate(30.0f), max_output);

    neuron positive(false, false);
    positive.setValueRange({true, 5.0f});
    EXPECT_EQ(positive.activate(-3.0f), 0.0f);
    EXPECT_EQ(positive.activate(2.0f), 2.0f);
    EXPECT_EQ(positive.activate(30.0f), 5.0f);

    neuron ceiling(false, false);
    ceiling.setValueRange({false, 2.0f});
    EXPECT_EQ(ceiling.activate(-3.0f), -2.0f);

    // Binary without rectification: 1 or 0, never -1.
    neuron perceptron(false, false);
    perceptron.setValueRange({true, max_output});
    perceptron.setBinary(true);
    EXPECT_EQ(perceptron.activate(-0.5f), 0.0f);
    EXPECT_EQ(perceptron.activate(0.5f), 1.0f);

    // E-R: negative sums are silence; spontaneous firing is never negative.
    neuron er(false, true);
    er.setValueRange({true, max_output});
    er.setSpontaneous({min_threshold, 1.0f, 1.0f});
    bool spontaneous = false;
    for (int t = 0; t < 200; ++t) {
        EXPECT_GE(er.activate(t % 3 == 0 ? -4.0f : 0.0f), 0.0f);
        spontaneous = spontaneous || er.lastFiring() == neuron::Firing::Spontaneous;
    }
    EXPECT_TRUE(spontaneous);

    // The derivative: zero below 0, like a ReLU.
    Derivatives d;
    neuron grad(false, false);
    grad.setValueRange({true, max_output});
    grad.activate(-1.0f, 0.5f, d);
    EXPECT_EQ(d.dyds, 0.0f);
    grad.activate(1.0f, 0.5f, d);
    EXPECT_EQ(d.dyds, 1.0f);
}

TEST(PositiveTest, LayerSettersValidate)
{
    network net;
    const auto h = net.addLayer("h", LayerSpec::Dense(3));
    auto& layer = net.layerAs<dense>(h);
    EXPECT_THROW(layer.setValueRange({true, 0.0f}), std::invalid_argument);
    EXPECT_THROW(layer.setValueRange({true, std::numeric_limits<float>::infinity()}), std::invalid_argument);
    EXPECT_THROW(net.addLayer("pool", LayerSpec::Pool2D(Window2D::square(2, 2)).positive()), std::invalid_argument);
    LayerSpec retina = LayerSpec::Retina(RetinaSpec{{1, 4, 4}});
    retina.positiveWeights = true;  // fixed filters: nothing to keep positive
    EXPECT_THROW(net.addLayer("eye", retina), std::invalid_argument);
    EXPECT_NO_THROW(net.addLayer("eye2", LayerSpec::Retina(RetinaSpec{{1, 4, 4}}).positive()));  // values only
    EXPECT_FALSE(net.layerSpec(net.layerCount() - 1).positiveWeights);
}

TEST(PositiveTest, DefaultIsUnchangedBitForBit)
{
    // Explicitly setting the default range must not move a single bit.
    auto run = [](bool explicitDefault) {
        reseed(11);
        network net;
        LayerSpec spec = LayerSpec::Dense(10);
        if (explicitDefault)
            spec.ceiling = max_output;
        const auto h = net.addLayer("h", spec);
        net.addInputs(h, 4);
        if (explicitDefault)
            net.layerAs<dense>(h).setValueRange({});
        std::mt19937 gen(2);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> trace;
        for (int t = 0; t < 100; ++t) {
            for (size_t i = 0; i < 4; ++i)
                net.setInput(i, u(gen));
            net.step();
            net.applyReward(u(gen), 0.05f);
            for (float y : net.getLayer(h).output())
                trace.push_back(y);
        }
        for (float w : allWeights(net.layerAs<dense>(h)))
            trace.push_back(w);
        return trace;
    };
    EXPECT_EQ(run(false), run(true));
}

TEST(PositiveTest, InitialWeightsAreMagnitudesOfTheUsualDraw)
{
    auto draw = [](bool positive) {
        reseed(5);
        network net;
        LayerSpec spec = LayerSpec::Dense(6);
        if (positive)
            spec = spec.positive();
        const auto h = net.addLayer("h", spec);
        net.addInputs(h, 5);
        return allWeights(net.layerAs<dense>(h));
    };
    const std::vector<float> plain = draw(false), positive = draw(true);
    ASSERT_EQ(plain.size(), positive.size());
    bool anyNegative = false;
    for (size_t i = 0; i < plain.size(); ++i) {
        EXPECT_EQ(positive[i], std::abs(plain[i]));
        anyNegative = anyNegative || plain[i] < 0.0f;
    }
    EXPECT_TRUE(anyNegative);
}

TEST(PositiveTest, TurningOnFoldsAndHandSetWeightsClamp)
{
    reseed(3);
    network net;
    LayerSpec spec = LayerSpec::Dense(2, false, false);
    spec.learningRule = LearningRule::traced().withBias();
    const auto h = net.addLayer("h", spec);
    net.addInputs(h, 3);
    auto& layer = net.layerAs<dense>(h);
    layer.setWeights(0, {-1.0f, 2.0f, -0.5f});
    layer.setBias(0, -0.3f);
    layer.setPositiveWeights(true);
    EXPECT_EQ(layer.weights(0), (std::vector<float>{1.0f, 2.0f, 0.5f}));
    EXPECT_EQ(layer.bias(0), 0.0f);
    layer.setWeights(1, {-1.0f, 0.25f, -7.0f});
    EXPECT_EQ(layer.weights(1), (std::vector<float>{0.0f, 0.25f, 0.0f}));
    layer.setBias(1, -2.0f);
    EXPECT_EQ(layer.bias(1), 0.0f);
    EXPECT_EQ(layer.minWeight(), 0.0f);
    layer.setPositiveWeights(false);
    EXPECT_EQ(layer.minWeight(), -max_weight);
    layer.setWeights(1, {-1.0f, 0.25f, -7.0f});
    EXPECT_EQ(layer.weights(1), (std::vector<float>{-1.0f, 0.25f, -7.0f}));
}

// Every rule, driven the way it learns (reward, own error, feedback,
// unsupervised): a positive layer's values and weights never go below 0,
// and every rule still moves the weights.
TEST(PositiveTest, EveryRuleStaysPositive)
{
    const std::vector<std::pair<const char*, LearningRule>> rules = {
        {"sign", LearningRule::sign()},
        {"trace", LearningRule::traced(0.5f).withBias()},
        {"feedback_alignment", LearningRule::feedbackAlignment(0.5f).withBias()},
        {"perturbation", LearningRule::perturbation()},
        {"oja", LearningRule::oja()},
        {"bcm", LearningRule::bcm()},
        {"eligibility", LearningRule::eligibility().withBias()},
        {"eprop", LearningRule::eprop(0.5f).withBias()},
        {"surrogate", LearningRule::surrogate(4).withBias()},
    };
    for (const auto& [name, rule] : rules) {
        for (bool er : {false, true}) {
            SCOPED_TRACE(std::string(name) + (er ? " E-R" : " plain"));
            reseed(9);
            network net;
            LayerSpec hs = LayerSpec::Dense(12, false, er).positive(4.0f);
            hs.learningRule = rule;
            LayerSpec os = LayerSpec::Dense(3, false, er).positive(4.0f);
            os.learningRule = rule.type == LearningRuleType::FeedbackAlignment ? LearningRule::traced(0.5f) : rule;
            const auto hidden = net.addLayer("hidden", hs);
            const auto out = net.addLayer("out", os);
            net.addInputs(hidden, 6);
            net.connect(hidden, out);
            net.connect(hidden, hidden);  // recurrent weights live in the layer too
            net.addOutput(out);
            const std::vector<float> before = allWeights(net.layerAs<dense>(hidden));
            std::mt19937 gen(4);
            std::uniform_real_distribution<float> u(-2.0f, 2.0f);
            for (int t = 0; t < 300; ++t) {
                for (size_t i = 0; i < 6; ++i)
                    net.setInput(i, u(gen));  // negative inputs too
                net.step();
                for (auto id : {hidden, out})
                    for (float y : net.getLayer(id).output()) {
                        ASSERT_GE(y, 0.0f);
                        ASSERT_LE(y, 4.0f);
                    }
                const std::vector<float> y = net.outputs();
                std::vector<float> errors(y.size());
                for (size_t k = 0; k < y.size(); ++k)
                    errors[k] = (k == static_cast<size_t>(t) % 3 ? 2.0f : 0.0f) - y[k];
                if (rule.type == LearningRuleType::Sign || rule.type == LearningRuleType::Trace ||
                    rule.type == LearningRuleType::Eligibility)
                    net.applyReward(u(gen), 0.05f);
                else
                    net.applyError(errors, 0.05f);
            }
            for (auto id : {hidden, out})
                EXPECT_GE(smallest(allWeights(net.layerAs<dense>(id))), 0.0f);
            EXPECT_NE(allWeights(net.layerAs<dense>(hidden)), before);
        }
    }
}

TEST(PositiveTest, SpatialLayersStayPositive)
{
    for (bool shared : {true, false}) {
        SCOPED_TRACE(shared ? "conv2d" : "locally_connected2d");
        reseed(6);
        network net;
        const Window2D window = Window2D::square(3, 1, 1);
        LayerSpec spec = (shared ? LayerSpec::Conv2D(4, window) : LayerSpec::LocallyConnected2D(4, window)).positive();
        spec.learningRule = LearningRule::traced(0.5f);
        const auto eye = net.addLayer("eye", LayerSpec::Retina(RetinaSpec{{1, 6, 6}}, false, false));
        const auto v1 = net.addLayer("v1", spec);
        net.addInputs(eye, Shape{1, 6, 6});
        net.connect(eye, v1);
        auto& layer = net.layerAs<neuron_layer>(v1);
        EXPECT_GE(smallest(allWeights(layer)), 0.0f);  // drawn as magnitudes
        std::mt19937 gen(1);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int t = 0; t < 100; ++t) {
            for (size_t i = 0; i < 36; ++i)
                net.setInput(i, u(gen));
            net.step();
            net.applyReward(u(gen), 0.2f);
            for (float y : net.getLayer(v1).output())
                ASSERT_GE(y, 0.0f);
        }
        EXPECT_GE(smallest(allWeights(layer)), 0.0f);
        if (shared) {
            net.layerAs<conv2d>(v1).setKernel(0, std::vector<float>(9, -1.0f));
            EXPECT_EQ(net.layerAs<conv2d>(v1).kernel(0), std::vector<float>(9, 0.0f));
        }
    }
}

TEST(PositiveTest, SavedAndLoaded)
{
    reseed(4);
    network net;
    const auto h = net.addLayer("h", LayerSpec::Dense(5).positive(3.0f));
    LayerSpec weightsOnly = LayerSpec::Dense(2);
    weightsOnly.positiveWeights = true;
    const auto o = net.addLayer("o", weightsOnly);
    net.addInputs(h, 4);
    net.connect(h, o);
    std::stringstream data;
    net.save(data);
    const auto loaded = network::load(data);
    for (auto id : {h, o}) {
        const LayerSpec& a = net.layerSpec(id);
        const LayerSpec& b = loaded->layerSpec(id);
        EXPECT_EQ(a.positiveValues, b.positiveValues);
        EXPECT_EQ(a.ceiling, b.ceiling);
        EXPECT_EQ(a.positiveWeights, b.positiveWeights);
        EXPECT_EQ(allWeights(net.layerAs<dense>(id)), allWeights(loaded->layerAs<dense>(id)));
    }
    EXPECT_EQ(loaded->layerAs<dense>(h).valueRange(), (ValueRange{true, 3.0f}));
    EXPECT_EQ(loaded->layerAs<dense>(h).neurons()[0].valueRange(), (ValueRange{true, 3.0f}));
    EXPECT_TRUE(loaded->layerAs<dense>(o).positiveWeights());
    EXPECT_FALSE(loaded->layerSpec(o).positiveValues);

    std::ostringstream text;
    net.describe(text);
    EXPECT_NE(text.str().find("range h: values [0, 3], weights >= 0"), std::string::npos) << text.str();
    EXPECT_NE(text.str().find("range o: values [-10, 10], weights >= 0"), std::string::npos) << text.str();
}
