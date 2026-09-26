#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <random>
#include "../core/layers/dense.hpp"
#include "er_scales.hpp"
#include "../core/neuron.hpp"

TEST(DenseLayerTest, RecursiveTopologyAndCascadePropagation)
{
    // -------------------------------------------------------------------------
    // Step 1: Initial topology configuration (A: 2, B: 3, C: 4)
    // -------------------------------------------------------------------------
    dense A(2);
    dense B(3);
    dense C(4);

    EXPECT_TRUE(B.join(A));
    EXPECT_TRUE(C.join(B));

    // Verify initial layer dimensions and connection bounds
    EXPECT_EQ(A.size(), 2);
    EXPECT_EQ(A.getOutput().size(), 2);

    EXPECT_EQ(B.size(), 3);
    EXPECT_EQ(B.getOutput().size(), 3);
    for (const auto& n : B.getNeurons())
    {
        EXPECT_EQ(n.getWeights().size(), 2); // B reads from A (2 outputs)
    }

    EXPECT_EQ(C.size(), 4);
    EXPECT_EQ(C.getOutput().size(), 4);
    for (const auto& n : C.getNeurons())
    {
        EXPECT_EQ(n.getWeights().size(), 3); // C reads from B (3 outputs)
    }

    // -------------------------------------------------------------------------
    // Step 2: Feedback connection A -> C (Attach 2 neurons to C reading from A)
    // -------------------------------------------------------------------------
    EXPECT_TRUE(A.addFeedback(C, 2));

    // C expands to 6 neurons (4 existing + 2 new feedback neurons)
    EXPECT_EQ(C.size(), 6);
    EXPECT_EQ(C.getOutput().size(), 6);

    // -------------------------------------------------------------------------
    // Step 3: Feedback connection C -> A (Attach 3 neurons to A reading from C)
    // -------------------------------------------------------------------------
    EXPECT_TRUE(C.addFeedback(A, 3));

    // A expands from 2 to 5 neurons
    EXPECT_EQ(A.size(), 5);
    EXPECT_EQ(A.getOutput().size(), 5);

    // CASCADE: B listens to A, so B's input vector automatically expands
    // from 2 to 5 inputs. Weights of original 3 neurons in B expand accordingly.
    for (const auto& n : B.getNeurons())
    {
        EXPECT_EQ(n.getWeights().size(), 5);
    }

    // -------------------------------------------------------------------------
    // Step 4: Feedback connection B -> A (Attach 2 neurons to A reading from B)
    // -------------------------------------------------------------------------
    EXPECT_TRUE(B.addFeedback(A, 2));

    // A expands from 5 to 7 neurons
    EXPECT_EQ(A.size(), 7);
    EXPECT_EQ(A.getOutput().size(), 7);

    // CASCADE: B's input group connected to A expands again to 7 inputs
    for (const auto& n : B.getNeurons())
    {
        EXPECT_EQ(n.getWeights().size(), 7);
    }

    // -------------------------------------------------------------------------
    // Step 5: Feedback connection A -> B (Attach 1 neuron to B reading from A)
    // -------------------------------------------------------------------------
    EXPECT_TRUE(A.addFeedback(B, 1));

    // B expands from 3 to 4 neurons
    EXPECT_EQ(B.size(), 4);
    EXPECT_EQ(B.getOutput().size(), 4);

    // CASCADE IN C: C listens to B, so C's input group reading B expands
    // from 3 to 4 inputs. Weights of C's first 4 neurons expand accordingly.
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(C.getNeurons()[i].getWeights().size(), 4);
    }

    // -------------------------------------------------------------------------
    // Step 6: Execute forward pass and verify output pointer validity
    // -------------------------------------------------------------------------
    EXPECT_NO_THROW(A.forward());
    EXPECT_NO_THROW(B.forward());
    EXPECT_NO_THROW(C.forward());

    for (const auto& out_ptr : A.getOutput()) {
        ASSERT_NE(out_ptr, nullptr);
    }
    for (const auto& out_ptr : B.getOutput()) {
        ASSERT_NE(out_ptr, nullptr);
    }
    for (const auto& out_ptr : C.getOutput()) {
        ASSERT_NE(out_ptr, nullptr);
    }
}

TEST(DenseLayerTest, FullStateSerializationPreservesNetworkBehavior)
{
    // 1. Setup connected topology (A -> B) and execute initial forward steps
    dense A_orig(2);
    dense B_orig(3);
    B_orig.join(A_orig);

    *A_orig.getOutput()[0] = 1.5f;
    *A_orig.getOutput()[1] = 0.5f;

    for (int i = 0; i < 10; ++i) {
        B_orig.forward();
    }

    // 2. Serialize both layers to memory stream
    std::stringstream ss;
    A_orig.serialize(ss);
    B_orig.serialize(ss);

    // 3. Reconstruct fresh network instances and restore state via FullState deserialization
    dense A_restored(2);
    dense B_restored(3);
    B_restored.join(A_restored);

    A_restored.deserialize(ss, DeserializeMode::FullState);
    B_restored.deserialize(ss, DeserializeMode::FullState);

    // 4. Feed identical new input and execute forward pass in both networks
    *A_orig.getOutput()[0] = 2.0f;
    *A_orig.getOutput()[1] = 1.0f;

    *A_restored.getOutput()[0] = 2.0f;
    *A_restored.getOutput()[1] = 1.0f;

    B_orig.forward();
    B_restored.forward();

    // 5. Verify bit-exact equality between original and restored layer outputs
    ASSERT_EQ(B_orig.getOutput().size(), B_restored.getOutput().size());
    for (size_t i = 0; i < B_orig.getOutput().size(); ++i) {
        EXPECT_FLOAT_EQ(*B_orig.getOutput()[i], *B_restored.getOutput()[i]);
    }
}

TEST(DenseLayerTest, FileSerializationAndDeserialization)
{
    namespace fs = std::filesystem;
    const fs::path test_file = fs::temp_directory_path() / "dense_layer_test.bin";

    // Clean up temporary file if left over from a previous interrupted run
    if (fs::exists(test_file)) {
        fs::remove(test_file);
    }

    // 1. Setup connected topology (A -> B) and execute initial forward steps
    dense A_orig(2);
    dense B_orig(3);
    B_orig.join(A_orig);

    *A_orig.getOutput()[0] = 1.23f;
    *A_orig.getOutput()[1] = 0.45f;

    for (int i = 0; i < 5; ++i) {
        B_orig.forward();
    }

    // 2. Serialize network state directly to binary file on disk
    {
        std::ofstream out_file(test_file, std::ios::binary);
        ASSERT_TRUE(out_file.is_open());
        A_orig.serialize(out_file);
        B_orig.serialize(out_file);
    }

    // 3. Restore network state from binary file into fresh instances
    dense A_restored(2);
    dense B_restored(3);
    B_restored.join(A_restored);

    {
        std::ifstream in_file(test_file, std::ios::binary);
        ASSERT_TRUE(in_file.is_open());
        A_restored.deserialize(in_file, DeserializeMode::FullState);
        B_restored.deserialize(in_file, DeserializeMode::FullState);
    }

    // 4. Feed identical new input and execute forward pass in both networks
    *A_orig.getOutput()[0] = 2.0f;
    *A_orig.getOutput()[1] = 1.0f;

    *A_restored.getOutput()[0] = 2.0f;
    *A_restored.getOutput()[1] = 1.0f;

    B_orig.forward();
    B_restored.forward();

    // 5. Verify bit-exact equality between original and restored layer outputs
    ASSERT_EQ(B_orig.getOutput().size(), B_restored.getOutput().size());
    for (size_t i = 0; i < B_orig.getOutput().size(); ++i) {
        EXPECT_FLOAT_EQ(*B_orig.getOutput()[i], *B_restored.getOutput()[i]);
    }

    // Clean up temporary test file
    fs::remove(test_file);
}




TEST(DenseLayerTest, MultipleAddFeedbackUpdatesListenersWeightsCorrectly)
{
    // 1. Topology:
    // source   -> sends feedback to target
    // target   -> the layer that grows dynamically
    // listener -> reads from target (becomes its listener)
    dense source(2);
    dense target(3);
    dense listener(2);

    // listener subscribes to target's output
    ASSERT_TRUE(listener.join(target));

    // Initial state
    EXPECT_EQ(target.size(), 3u);
    EXPECT_EQ(target.getOutput().size(), 3u);
    EXPECT_EQ(listener.size(), 2u);

    // 2. First addFeedback call: add 2 new neurons to target
    ASSERT_TRUE(source.addFeedback(target, 2));

    // target grows from 3 to 5 neurons
    EXPECT_EQ(target.size(), 5u);
    EXPECT_EQ(target.getOutput().size(), 5u);

    // Run a forward pass to make sure listener extended its group's neuron
    // weights correctly and nothing reads out of bounds
    EXPECT_NO_THROW(source.forward());
    EXPECT_NO_THROW(target.forward());
    EXPECT_NO_THROW(listener.forward());

    // 3. Second addFeedback call: add 3 more neurons to target
    ASSERT_TRUE(source.addFeedback(target, 3));

    // target grows from 5 to 8 neurons
    EXPECT_EQ(target.size(), 8u);
    EXPECT_EQ(target.getOutput().size(), 8u);

    // Run forward again on all layers
    EXPECT_NO_THROW(source.forward());
    EXPECT_NO_THROW(target.forward());
    EXPECT_NO_THROW(listener.forward());

    // 4. Check listener's outputs are consistent
    for (const auto& out_ptr : listener.getOutput())
    {
        ASSERT_NE(out_ptr, nullptr);
        EXPECT_FALSE(std::isnan(*out_ptr));
    }
}


TEST(DenseLayerTest, StressTestFourLargeLayersWithCrossFeedback)
{
    constexpr size_t BASE_NEURONS = 1000;
    constexpr size_t FEEDBACK_NEURONS = 100;

    // 1. Create 4 large layers (4000 neurons to start with)
    dense l1(BASE_NEURONS);
    dense l2(BASE_NEURONS);
    dense l3(BASE_NEURONS);
    dense l4(BASE_NEURONS);

    // 2. Forward connections (plain feedforward chain)
    // l2 reads from l1 (1,000,000 weights)
    // l3 reads from l2 (1,000,000 weights)
    // l4 reads from l3 (1,000,000 weights)
    ASSERT_TRUE(l2.join(l1));
    ASSERT_TRUE(l3.join(l2));
    ASSERT_TRUE(l4.join(l3));

    // 3. Add feedback (cross) connections
    // l3 adds 100 new neurons to l1
    ASSERT_TRUE(l3.addFeedback(l1, FEEDBACK_NEURONS));

    // l4 adds 100 new neurons to l2
    // l3 reads from l2, so l3's neurons automatically get +100 weights
    ASSERT_TRUE(l4.addFeedback(l2, FEEDBACK_NEURONS));

    // l2 adds 100 new neurons to l4
    ASSERT_TRUE(l2.addFeedback(l4, FEEDBACK_NEURONS));

    // Check sizes grew after propagation
    EXPECT_EQ(l1.size(), BASE_NEURONS + FEEDBACK_NEURONS); // 1100
    EXPECT_EQ(l2.size(), BASE_NEURONS + FEEDBACK_NEURONS); // 1100
    EXPECT_EQ(l3.size(), BASE_NEURONS);                    // 1000
    EXPECT_EQ(l4.size(), BASE_NEURONS + FEEDBACK_NEURONS); // 1100

    // 4. Run the network for 10 time steps
    constexpr int STEPS = 10;
    for (int step = 0; step < STEPS; ++step)
    {
        EXPECT_NO_THROW(l1.forward());
        EXPECT_NO_THROW(l2.forward());
        EXPECT_NO_THROW(l3.forward());
        EXPECT_NO_THROW(l4.forward());
    }

    // 5. Check every neuron's output is valid
    auto verifyOutputs = [](const dense& layer, const std::string& name) {
        const auto& outputs = layer.getOutput();
        EXPECT_EQ(outputs.size(), layer.size());
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            ASSERT_NE(outputs[i], nullptr) << "Null pointer on output " << i << " in " << name;
            EXPECT_FALSE(std::isnan(*outputs[i])) << "NaN detected on output " << i << " in " << name;
        }
    };

    verifyOutputs(l1, "Layer 1");
    verifyOutputs(l2, "Layer 2");
    verifyOutputs(l3, "Layer 3");
    verifyOutputs(l4, "Layer 4");
}


namespace {

// Learning tests run many independent trials instead of one. Each trial
// reseeds the shared RNGs, so its initial weights don't depend on which
// tests ran earlier in the process. The test then asserts on the success
// rate: a single run of a network with E-R, habituation and spontaneous
// firing is too noisy to judge.
constexpr int PAVLOV_TRIALS = 50;

// Conditioned sign inversion: a negative stimulus should give a positive
// response, a positive stimulus a negative one. The stimulus is weak relative
// to the E-R threshold (WEAK_STIMULUS = 5 x baseline_threshold, 0.5 at the
// original baseline 0.1), so the test keeps its meaning when the baseline is
// tuned.
const float STIMULUS_NEG = -WEAK_STIMULUS;
const float STIMULUS_POS = WEAK_STIMULUS;

struct PavlovianResult
{
    float initial_neg, initial_pos;   // responses to STIMULUS_NEG / STIMULUS_POS before training
    float trained_neg, trained_pos;   // responses to STIMULUS_NEG / STIMULUS_POS after training

    bool correct() const { return trained_neg > 0.0f && trained_pos < 0.0f; }
    bool correctBefore() const { return initial_neg > 0.0f && initial_pos < 0.0f; }
    float gap() const { return trained_neg - trained_pos; }  // > 0 means inverted
};

// learningRate = 0 gives a control run: same seed, same stimuli, no learning.
PavlovianResult runPavlovianTrial(std::uint32_t seed, bool hasER, float learningRate, int epochs)
{
    constexpr size_t LAYER_SIZE = 5;
    constexpr int SAMPLE_TICKS = 4;  // response is averaged over this many ticks
    constexpr bool HAS_HABITUATION = true;
    const bool HAS_ER = hasER;

    neuron::reseed(seed);

    dense A(LAYER_SIZE, HAS_HABITUATION, HAS_ER);
    dense B(LAYER_SIZE, HAS_HABITUATION, HAS_ER);
    dense C(LAYER_SIZE, HAS_HABITUATION, HAS_ER);
    dense D(LAYER_SIZE, HAS_HABITUATION, HAS_ER);

    B.join(A);
    C.join(B);
    D.join(C);
    A.addFeedback(C, 20);

    auto sensor = std::make_shared<float>(0.0f);
    A.attachInputs({sensor});

    auto propagate = [&](int ticks = 4) {
        for (int t = 0; t < ticks; ++t) {
            A.forward();
            B.forward();
            C.forward();
            D.forward();
        }
    };

    // Relaxation helper: let activity settle back to rest
    auto relax = [&](int ticks = 6) {
        *sensor = 0.0f;
        propagate(ticks);
    };

    auto getOutputResponse = [&]() {
        float sum = 0.0f;
        const auto& out = D.getOutput();
        for (const auto& v : out) sum += *v;
        return sum / static_cast<float>(out.size());
    };

    // Response to `stimulus` from rest, averaged over every tick of the
    // presentation window, not only the last tick: E-R makes single-tick
    // outputs flicker.
    auto sampleResponse = [&](float stimulus) {
        relax(6);
        *sensor = stimulus;
        float sum = 0.0f;
        for (int t = 0; t < SAMPLE_TICKS; ++t) {
            propagate(1);
            sum += getOutputResponse();
        }
        return sum / static_cast<float>(SAMPLE_TICKS);
    };

    // Presents `stimulus`, then rewards with the sign the response should
    // have. updateWeights() moves every eligible neuron's output toward the
    // reward's sign, so the reward is the desired direction, not a
    // right/wrong score.
    auto train = [&](float stimulus, float desiredSign) {
        relax(20);
        *sensor = stimulus;
        propagate(4);
        A.applyReward(desiredSign, learningRate);
        B.applyReward(desiredSign, learningRate);
        C.applyReward(desiredSign, learningRate);
        D.applyReward(desiredSign, learningRate);
    };

    PavlovianResult r{};

    // PHASE 1: pre-test
    r.initial_neg = sampleResponse(STIMULUS_NEG);
    r.initial_pos = sampleResponse(STIMULUS_POS);

    // PHASE 2: training, both stimuli every epoch
    for (int epoch = 0; epoch < epochs; ++epoch) {
        train(STIMULUS_NEG, +1.0f);
        train(STIMULUS_POS, -1.0f);
    }

    // PHASE 3: post-test
    r.trained_neg = sampleResponse(STIMULUS_NEG);
    r.trained_pos = sampleResponse(STIMULUS_POS);
    return r;
}

struct PavlovianStats
{
    int correct = 0, correct_before = 0, correct_control = 0;
    int gap_beats_control = 0;  // trained gap > control gap on the same seed
    float mean_neg = 0.0f, mean_pos = 0.0f, mean_neg_control = 0.0f, mean_pos_control = 0.0f;
};

PavlovianStats measurePavlovian(bool hasER)
{
    constexpr float LEARNING_RATE = 0.005f; // small learning step
    constexpr int EPOCHS = 60;

    PavlovianStats st;
    for (int trial = 0; trial < PAVLOV_TRIALS; ++trial)
    {
        const auto seed = static_cast<std::uint32_t>(trial);
        const PavlovianResult r = runPavlovianTrial(seed, hasER, LEARNING_RATE, EPOCHS);
        const PavlovianResult c = runPavlovianTrial(seed, hasER, 0.0f, EPOCHS);

        st.correct += r.correct();
        st.correct_before += r.correctBefore();
        st.correct_control += c.correct();
        st.gap_beats_control += r.gap() > c.gap();
        st.mean_neg += r.trained_neg / PAVLOV_TRIALS;
        st.mean_pos += r.trained_pos / PAVLOV_TRIALS;
        st.mean_neg_control += c.trained_neg / PAVLOV_TRIALS;
        st.mean_pos_control += c.trained_pos / PAVLOV_TRIALS;
    }
    return st;
}

void printPavlovianStats(const char* title, const PavlovianStats& st)
{
    std::cout << "\n==========================================\n"
              << " [Pavlovian sign inversion, " << title << " - " << PAVLOV_TRIALS << " trials]\n"
              << "==========================================\n"
              << " Stimulus +-" << WEAK_STIMULUS << " (5 x baseline_threshold)\n"
              << " Mean response to - / + stimulus after training: "
              << st.mean_neg << " / " << st.mean_pos << "\n"
              << " Mean response to - / + stimulus without learning: "
              << st.mean_neg_control << " / " << st.mean_pos_control << "\n"
              << " Correct (- stimulus -> +, + stimulus -> -): before " << st.correct_before
              << ", after " << st.correct << ", control " << st.correct_control
              << "  (of " << PAVLOV_TRIALS << ")\n"
              << " Gap larger than control:      " << st.gap_beats_control << "/" << PAVLOV_TRIALS << "\n"
              << "==========================================\n";
}

} // namespace

TEST(DenseLayerTest, PavlovianSignInversionWithoutER)
{
    const PavlovianStats st = measurePavlovian(false);
    printPavlovianStats("E-R off", st);

    // Measured: 23/50 correct before training and in the control, 50/50 after.
    // The control is a coin flip per seed (~25/50), so it only has to be
    // clearly short of solving the task.
    EXPECT_LE(st.correct_control, PAVLOV_TRIALS * 7 / 10);
    EXPECT_GE(st.correct, PAVLOV_TRIALS * 9 / 10);
    EXPECT_GE(st.gap_beats_control, PAVLOV_TRIALS * 9 / 10);
}

TEST(DenseLayerTest, PavlovianSignInversionWithER)
{
    const PavlovianStats st = measurePavlovian(true);
    printPavlovianStats("E-R on", st);

    // Measured: 0/50 correct before training, 13/50 in the control, 50/50 after.
    // Trained responses sit at the max_output clamp (+-10).
    EXPECT_LT(st.correct_control, PAVLOV_TRIALS / 2);
    EXPECT_GE(st.correct, PAVLOV_TRIALS * 9 / 10);
    EXPECT_GE(st.gap_beats_control, PAVLOV_TRIALS * 9 / 10);
}

// =============================================================================
// Sequence order discrimination: "1 then 2" -> positive, "2 then 1" -> negative
// =============================================================================
namespace {

// Smallest topology that can learn this with the sign-only Hebbian rule:
//
//   sensor -> taps[0] ------------------+
//               |                       +-> hidden (32) -> out (1)
//               +-> delay -> taps[1] ---+
//
// - taps + delay form a one-step memory: delay reads taps[0] from the
//   previous tick, and taps[1] copies delay, so taps = { x(t), x(t-1) }.
//   The delay needs its own layer. A self-feedback neuron inside taps would
//   be stepped after taps[0] in the same forward() call, so it would read
//   x(t) instead of x(t-1).
// - hidden mixes both taps with random weights: h = a*x(t) + b*x(t-1).
//   Only neurons with a/b of opposite sign and similar magnitude flip sign
//   between the two orders, roughly 1 in 4. With 8 neurons, about 1 seed in
//   8 has none and cannot learn; 32 makes that practically impossible.
//   It is required because updateWeights() only uses the sign of each input.
//   Raw taps are always positive here, so the readout could never learn
//   order from them directly. The mixed features change sign between
//   "1,2" and "2,1".
// - out is the readout. It is the only layer that learns.
//
// Habituation is always off. E-R is off by default, so the delay line passes
// values unchanged. SequenceConfig turns E-R on for the memory path
// (taps + delay) and/or the processing path (hidden + out) to compare.
struct SequenceConfig
{
    const char* name;
    bool memoryER;      // E-R on taps and delay
    bool processingER;  // E-R on hidden and out
};

constexpr SequenceConfig SEQ_NO_ER{"E-R off", false, false};
constexpr SequenceConfig SEQ_PROCESSING_ER{"E-R on hidden+out", false, true};
constexpr SequenceConfig SEQ_ALL_ER{"E-R on all layers", true, true};

struct SequenceNet
{
    std::shared_ptr<float> sensor = std::make_shared<float>(0.0f);
    dense taps;
    dense delay;
    dense hidden;
    dense out;

    explicit SequenceNet(const SequenceConfig& config = SEQ_NO_ER)
        : taps(1, false, config.memoryER),
          delay(1, false, config.memoryER),
          hidden(32, false, config.processingER),
          out(1, false, config.processingER)
    {
        taps.attachInputs({sensor});   // taps[0] reads the sensor
        delay.join(taps);              // delay reads taps' output
        delay.addFeedback(taps, 1);    // taps[1] reads delay; delay auto-grows to read taps[1] too
        hidden.join(taps);
        out.join(hidden);

        // Pin the memory path to identity weights: x(t) and x(t-1) copied exactly.
        taps.getNeurons()[0].setWeights({1.0f});
        taps.getNeurons()[1].setWeights({1.0f});
        delay.getNeurons()[0].setWeights({1.0f, 0.0f}); // reads taps[0], ignores taps[1]
    }

    // One time step. delay steps before taps, so it still sees the previous tick's taps[0].
    void tick(float x)
    {
        *sensor = x;
        delay.forward();
        taps.forward();
        hidden.forward();
        out.forward();
    }

    // Clears the memory, presents a two-element sequence, returns the readout.
    float present(float first, float second)
    {
        for (int i = 0; i < 3; ++i)
            tick(0.0f);
        tick(first);
        tick(second);
        return *out.getOutput()[0];
    }
};

} // namespace

TEST(DenseLayerTest, SequenceDelayLineExposesPreviousInput)
{
    SequenceNet net;

    ASSERT_EQ(net.taps.size(), 2u);
    ASSERT_EQ(net.delay.getNeurons()[0].getWeights().size(), 2u); // grew with taps
    ASSERT_EQ(net.hidden.getNeurons()[0].getWeights().size(), 2u);

    net.present(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(*net.taps.getOutput()[0], 2.0f); // x(t)
    EXPECT_FLOAT_EQ(*net.taps.getOutput()[1], 1.0f); // x(t-1)

    net.present(2.0f, 1.0f);
    EXPECT_FLOAT_EQ(*net.taps.getOutput()[0], 1.0f);
    EXPECT_FLOAT_EQ(*net.taps.getOutput()[1], 2.0f);
}

namespace {

constexpr int SEQUENCE_TRIALS = 100;
constexpr float SEQUENCE_LEARNING_RATE = 0.01f;

struct SequenceResult
{
    float initial_12, initial_21;
    float trained_12, trained_21;
};

// learningRate = 0 gives a control run: same seed, same presentations, no
// learning. With E-R the network's state changes over the presentations, so
// "before training" alone is not a fair baseline.
SequenceResult runSequenceTrial(std::uint32_t seed, const SequenceConfig& config, float learningRate)
{
    constexpr int EPOCHS = 200;

    neuron::reseed(seed);
    SequenceNet net(config);

    SequenceResult r{};
    r.initial_12 = net.present(1.0f, 2.0f);
    r.initial_21 = net.present(2.0f, 1.0f);

    // The reward sign is the desired response. updateWeights() moves the
    // output toward the reward's sign: reward "1,2", punish "2,1".
    // applyReward() runs right after present(), so out still reads the
    // hidden values from the final tick.
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        net.present(1.0f, 2.0f);
        net.out.applyReward(+1.0f, learningRate);

        net.present(2.0f, 1.0f);
        net.out.applyReward(-1.0f, learningRate);
    }

    r.trained_12 = net.present(1.0f, 2.0f);
    r.trained_21 = net.present(2.0f, 1.0f);
    return r;
}

struct SequenceStats
{
    int order_visible = 0;    // untrained net gives different outputs for the two orders
    int correct_before = 0;   // (1,2) > 0 and (2,1) < 0 by chance, before training
    int correct_after = 0;
    int correct_control = 0;  // same, after the presentations with learning off
    int gap_widened = 0;
};

bool isCorrect(float r12, float r21) { return r12 > 0.0f && r21 < 0.0f; }

SequenceStats measureSequence(const SequenceConfig& config)
{
    SequenceStats st;
    for (int trial = 0; trial < SEQUENCE_TRIALS; ++trial)
    {
        const auto seed = static_cast<std::uint32_t>(trial);
        const SequenceResult r = runSequenceTrial(seed, config, SEQUENCE_LEARNING_RATE);
        const SequenceResult c = runSequenceTrial(seed, config, 0.0f);
        st.order_visible += r.initial_12 != r.initial_21;
        st.correct_before += isCorrect(r.initial_12, r.initial_21);
        st.correct_after += isCorrect(r.trained_12, r.trained_21);
        st.correct_control += isCorrect(c.trained_12, c.trained_21);
        st.gap_widened += (r.trained_12 - r.trained_21) > (r.initial_12 - r.initial_21);
    }
    return st;
}

void printSequenceStats(const SequenceConfig& config, const SequenceStats& st)
{
    std::cout << " " << config.name << " (of " << SEQUENCE_TRIALS << "): correct before "
              << st.correct_before << ", after " << st.correct_after
              << ", control " << st.correct_control
              << "; gap widened " << st.gap_widened << "\n";
}

} // namespace

TEST(DenseLayerTest, SequenceOrderLearningDiscriminates12From21)
{
    const SequenceStats st = measureSequence(SEQ_NO_ER);

    std::cout << "\n==========================================\n"
              << " [Sequence order discrimination]\n"
              << "==========================================\n";
    printSequenceStats(SEQ_NO_ER, st);
    std::cout << "==========================================\n";

    // Without the delay line the network could not see the order at all.
    EXPECT_EQ(st.order_visible, SEQUENCE_TRIALS);
    // Measured: ~5% correct by chance, ~99% after training.
    EXPECT_LT(st.correct_before, SEQUENCE_TRIALS / 2);
    EXPECT_GE(st.correct_after, SEQUENCE_TRIALS * 9 / 10);
    EXPECT_GE(st.gap_widened, SEQUENCE_TRIALS * 9 / 10);
}

TEST(DenseLayerTest, SequenceOrderLearningWithAndWithoutER)
{
    const SequenceConfig configs[] = {SEQ_NO_ER, SEQ_PROCESSING_ER, SEQ_ALL_ER};

    std::cout << "\n==========================================\n"
              << " [Sequence order discrimination: E-R comparison]\n"
              << "==========================================\n";
    SequenceStats stats[3];
    for (int i = 0; i < 3; ++i)
    {
        stats[i] = measureSequence(configs[i]);
        printSequenceStats(configs[i], stats[i]);
    }
    std::cout << "==========================================\n";

    // Measured (3 rest ticks between presentations), recovery_factor 0.8
    // (0.9 in brackets):
    //   E-R off             99/100 correct, control 5
    //   E-R on hidden+out   43/100 (29), control 1
    //   E-R on all layers   87/100 (100), control 0
    // E-R results depend on how far thresholds relax in the 3 rest ticks.
    // hidden+out stays weak: firing lifts out's threshold far above
    // baseline, so it is often still refractory on the next presentation
    // and outputs exactly 0.
    EXPECT_GE(stats[0].correct_after, SEQUENCE_TRIALS * 9 / 10);
    EXPECT_LT(stats[2].correct_control, SEQUENCE_TRIALS / 2);
    EXPECT_GE(stats[2].correct_after, SEQUENCE_TRIALS * 8 / 10);
}
