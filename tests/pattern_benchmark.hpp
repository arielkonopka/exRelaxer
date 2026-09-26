#pragma once
// Gapped pattern detection benchmark, shared by the tests and by quick
// experiment programs:
//   ..., A, {0..n}, B, {0..n}, C, ...  ->  positive for the n ticks after C,
//   negative everywhere else.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "../core/network.hpp"
#include "../core/layers/dense.hpp"

using namespace exr;

namespace pattern_benchmark {

constexpr float PATTERN_A = 2.2f;
constexpr float PATTERN_B = 5.5f;
constexpr float PATTERN_C = 3.8f;
constexpr float DISTRACTOR_MAX = 10.0f;  // distractors are uniform in [0, DISTRACTOR_MAX]
// ... but never within this of A, B or C. Wide enough for frozen band
// detectors (see addValueDetectors); much wider would merge the zones around
// A, C and B (1.6-1.7 apart) and make any value in that range a symbol.
constexpr float SYMBOL_MARGIN = 0.5f;
constexpr int PATTERN_MAX_GAP = 1;       // n: 0..n ignored values between A-B and B-C
constexpr int PATTERN_HOLD = 1;          // n: positive ticks after C

constexpr int PATTERN_TRIALS = 50;
constexpr size_t PATTERN_TRAIN_TICKS = 10000;
constexpr size_t PATTERN_TEST_TICKS = 3000;
constexpr float PATTERN_LEARNING_RATE = 0.005f;

// |t| < 2 is treated as noise: with 50 trials that is roughly the 95%
// confidence level (two-sided).
constexpr float SIGNIFICANCE_T = 2.0f;

struct PatternStream
{
    std::vector<float> values;
    std::vector<int> target;    // +1 during the PATTERN_HOLD ticks after a completed pattern, else -1
    std::vector<bool> afterC;   // tick lies within PATTERN_HOLD ticks after a C (valid or not)
};

// Random stream of scalar values with embedded patterns and decoys.
// Most decoys end in C, so that "respond after any C" is not enough: on the
// ticks after a C, only the A..B..C history tells positive from negative.
// Labels come from the pattern definition applied to the whole stream,
// so patterns formed by chance (e.g. a decoy next to a real pattern) count too.
inline PatternStream makePatternStream(std::uint32_t seed, size_t length)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::uniform_real_distribution<float> distractor_value(0.0f, DISTRACTOR_MAX);
    std::uniform_int_distribution<int> gap(0, PATTERN_MAX_GAP);
    std::uniform_int_distribution<int> long_gap(PATTERN_MAX_GAP + 1, PATTERN_MAX_GAP + 2);
    std::uniform_int_distribution<int> filler(1, 4);
    std::uniform_int_distribution<int> decoy_kind(0, 6);

    PatternStream s;
    auto distractor = [&]() {
        for (;;) {
            const float v = distractor_value(rng);
            if (std::abs(v - PATTERN_A) > SYMBOL_MARGIN && std::abs(v - PATTERN_B) > SYMBOL_MARGIN &&
                std::abs(v - PATTERN_C) > SYMBOL_MARGIN)
                return v;
        }
    };
    auto push = [&](float v) { s.values.push_back(v); };
    auto pushDistractors = [&](int count) { for (int i = 0; i < count; ++i) push(distractor()); };
    auto A = [&] { push(PATTERN_A); };
    auto B = [&] { push(PATTERN_B); };
    auto C = [&] { push(PATTERN_C); };
    auto g = [&] { pushDistractors(gap(rng)); };
    auto G = [&] { pushDistractors(long_gap(rng)); };

    while (s.values.size() < length) {
        const float event = chance(rng);
        if (event < 0.3f) {             // full pattern
            A(); g(); B(); g(); C();
        } else if (event < 0.8f) {      // decoy
            switch (decoy_kind(rng)) {
            case 0: A(); g(); B(); break;               // no C
            case 1: B(); g(); A(); g(); C(); break;     // wrong order
            case 2: A(); G(); B(); g(); C(); break;     // first gap too long
            case 3: A(); g(); B(); G(); C(); break;     // second gap too long
            case 4: A(); g(); C(); break;               // no B
            case 5: B(); g(); C(); break;               // no A
            case 6: C(); break;                         // C alone
            }
        }
        pushDistractors(filler(rng));
    }
    s.values.resize(length);

    s.target.assign(length, -1);
    s.afterC.assign(length, false);
    for (size_t t = 0; t < length; ++t) {
        if (s.values[t] != PATTERN_C)
            continue;
        for (size_t k = t + 1; k <= t + PATTERN_HOLD && k < length; ++k)
            s.afterC[k] = true;
        bool found = false;
        for (int g2 = 0; g2 <= PATTERN_MAX_GAP && !found; ++g2) {
            const long j = static_cast<long>(t) - 1 - g2;
            if (j < 0 || s.values[j] != PATTERN_B)
                continue;
            for (int g1 = 0; g1 <= PATTERN_MAX_GAP && !found; ++g1) {
                const long i = j - 1 - g1;
                found = i >= 0 && s.values[i] == PATTERN_A;
            }
        }
        if (found)
            for (size_t k = t + 1; k <= t + PATTERN_HOLD && k < length; ++k)
                s.target[k] = +1;
    }
    return s;
}

// Scores of one prediction sequence against a stream.
struct PatternScore
{
    float balanced = 0.0f;  // over all ticks: 0.5 * (hits on positive + hits on negative)
    float afterC = 0.0f;    // over ticks after a C only: valid pattern vs decoy. The sequence measure.
    bool diverged = false;
};

// `predict(t)` is the prediction (true = positive) for tick t.
inline PatternScore scorePredictions(const PatternStream& s, const std::function<bool(size_t)>& predict)
{
    int pos = 0, neg = 0, tp = 0, tn = 0;
    int c_pos = 0, c_neg = 0, c_tp = 0, c_tn = 0;
    for (size_t t = 0; t < s.values.size(); ++t) {
        const bool p = predict(t);
        if (s.target[t] > 0) { ++pos; tp += p; } else { ++neg; tn += !p; }
        if (s.afterC[t]) {
            if (s.target[t] > 0) { ++c_pos; c_tp += p; } else { ++c_neg; c_tn += !p; }
        }
    }
    PatternScore score;
    score.balanced = 0.5f * (static_cast<float>(tp) / pos + static_cast<float>(tn) / neg);
    score.afterC = 0.5f * (static_cast<float>(c_tp) / c_pos + static_cast<float>(c_tn) / c_neg);
    return score;
}

using NetworkBuilder = std::function<std::unique_ptr<network>(bool hasER)>;

// One tick: feed x to input 0, hold any further inputs at 1.0 (bias inputs),
// step, return the mean of the output values.
inline float patternStep(network& net, float x)
{
    net.setInput(0, x);
    for (size_t i = 1; i < net.inputCount(); ++i)
        net.setInput(i, 1.0f);
    net.step();
    const std::vector<float> out = net.outputs();
    float sum = 0.0f;
    for (float v : out) sum += v;
    return sum / static_cast<float>(out.size());
}

// --- Frozen building blocks ------------------------------------------------

inline dense& asDense(network& net, network::LayerId id) { return dynamic_cast<dense&>(net.getLayer(id)); }

// Hand-wired value detectors, frozen. Adds two layers fed by the network's
// first inputs (the sensor x, plus a bias input held at 1.0):
//
//   ramps (7): for each symbol s, clamp(k * (x - (s - h))) and
//              clamp(k * (x - (s + h))), which saturate at -10 / +10 thanks to
//              max_output; plus one constant +10.
//   bands (4): per symbol, half the difference of its two ramps minus 5:
//              +5 when |x - s| < h, -5 otherwise; plus a constant +5 (bias).
//
// Features are +-5 rather than 0/10 because the learning rule only uses the
// sign of each input, and a 0 input never teaches anything.
// E-R and habituation are off: habituation would silence the constant.
constexpr size_t BAND_FEATURES = 4;  // A, B, C, bias

struct ValueDetectors
{
    network::LayerId ramps, bands;
};

// Frozen band detectors for any set of symbol values: one +-5 feature per
// symbol (+5 when |x - symbol| < halfWidth), plus a +5 bias feature. Layers
// "ramps" (2 per symbol + a constant) and "bands" (symbols + 1), fed by two
// new inputs (the sensor and a bias held at 1.0).
inline ValueDetectors addSymbolDetectors(network& net, const std::vector<float>& symbols, float halfWidth)
{
    const float h = halfWidth;
    const float k = 20.0f / h;  // gain: saturated well before the band edges
    const size_t n = symbols.size();

    ValueDetectors d;
    d.ramps = net.addLayer("ramps", {LayerType::Dense, 2 * n + 1, false, false, true});
    d.bands = net.addLayer("bands", {LayerType::Dense, n + 1, false, false, true});
    net.addInputs(d.ramps, 2);  // x, bias
    net.connect(d.ramps, d.bands);

    dense& ramps = asDense(net, d.ramps);
    for (size_t i = 0; i < n; ++i) {
        ramps.setWeights(2 * i, {k, -k * (symbols[i] - h)});
        ramps.setWeights(2 * i + 1, {k, -k * (symbols[i] + h)});
    }
    ramps.setWeights(2 * n, {0.0f, 100.0f});  // constant +10

    dense& bands = asDense(net, d.bands);
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> w(2 * n + 1, 0.0f);
        w[2 * i] = 0.5f;
        w[2 * i + 1] = -0.5f;
        w[2 * n] = -0.5f;
        bands.setWeights(i, w);
    }
    std::vector<float> bias(2 * n + 1, 0.0f);
    bias[2 * n] = 0.5f;
    bands.setWeights(n, bias);  // constant +5
    return d;
}

// Detectors for the benchmark's A, B, C.
inline ValueDetectors addValueDetectors(network& net)
{
    return addSymbolDetectors(net, {PATTERN_A, PATTERN_B, PATTERN_C}, SYMBOL_MARGIN / 2.0f);
}

// Frozen delay window over `source` (width `width`): a new layer holding
// source's values at lags 0..depth, (depth + 1) * width neurons, lag-major.
// Built from a chain of copy layers tap1..tapN. For the lags to be right,
// the update order must run taps deepest first, then source, then the
// window: `order` receives the taps in that order; the caller places source
// and window after them.
inline network::LayerId addDelayWindow(network& net, network::LayerId source, size_t width, int depth,
                                       std::vector<network::LayerId>& tapOrder)
{
    auto identity = [&](network::LayerId id, size_t firstNeuron) {
        dense& target = asDense(net, id);
        for (size_t i = 0; i < width; ++i) {
            std::vector<float> w(width, 0.0f);
            w[i] = 1.0f;
            target.setWeights(firstNeuron + i, w);
        }
    };

    std::vector<network::LayerId> taps;
    network::LayerId previous = source;
    for (int lag = 1; lag <= depth; ++lag) {
        const auto tap = net.addLayer("tap" + std::to_string(lag), {LayerType::Dense, width, false, false, true});
        net.connect(previous, tap);
        identity(tap, 0);
        taps.push_back(tap);
        previous = tap;
    }

    // Each lag is its own wiring group, so every window neuron reads one source only.
    const auto window = net.addLayer("window", {LayerType::Dense, 0, false, false, true});
    net.addFeedback(source, window, width);
    identity(window, 0);
    for (int lag = 1; lag <= depth; ++lag) {
        net.addFeedback(taps[lag - 1], window, width);
        identity(window, static_cast<size_t>(lag) * width);
    }

    tapOrder.insert(tapOrder.end(), taps.rbegin(), taps.rend());
    return window;
}

// Frozen random reservoir (echo-state style) reading `source` (`sourceWidth`
// outputs): nothing in it is designed for the task. One layer with two
// populations, because a dense neuron sums one wiring group only:
//
//   input neurons (inputNeurons):         read `source`, random weights
//                                         scaled by inputScale / sqrt(sourceWidth)
//   recurrent neurons (recurrentNeurons): read the whole reservoir, themselves
//                                         included, scaled by recurrentScale /
//                                         sqrt(inputNeurons + recurrentNeurons)
//
// The input neurons bring in the current tick; the recurrent neurons carry
// the past forward (they see the input neurons' current values and their own
// previous ones). With recurrentNeurons = 0 there is no recurrence at all:
// any memory then comes from the neurons' own state (E-R thresholds).
// recoveryJitter spreads the neurons' E-R relaxation rates (see
// neuron::randomizeDynamics), i.e. their memory timescales.
// Weights come from the neuron random streams, so exr::reseed controls them.
// The caller places the returned layer after `source` in the update order.
inline network::LayerId addReservoir(network& net, network::LayerId source, size_t sourceWidth,
                                     size_t inputNeurons, size_t recurrentNeurons,
                                     float inputScale, float recurrentScale, bool hasER,
                                     const Jitter& recoveryJitter = {}, const Jitter& alphaJitter = {})
{
    const auto res = net.addLayer("reservoir", {LayerType::Dense, inputNeurons, false, hasER, true,
                                                recoveryJitter, {}, alphaJitter});
    net.connect(source, res);
    if (recurrentNeurons > 0)
        net.addFeedback(res, res, recurrentNeurons);

    dense& reservoir = asDense(net, res);
    const float in_factor = inputScale / std::sqrt(static_cast<float>(sourceWidth));
    const float rec_factor = recurrentScale / std::sqrt(static_cast<float>(inputNeurons + recurrentNeurons));
    for (size_t i = 0; i < reservoir.size(); ++i) {
        std::vector<float> weights = reservoir.weights(i);
        for (float& w : weights)
            w *= i < inputNeurons ? in_factor : rec_factor;
        reservoir.setWeights(i, weights);
    }
    return res;
}

// With E-R on, a runaway network does not stay non-finite: once thresholds
// become inf nothing fires and the output sits at exactly 0. So divergence
// is detected from the response magnitude, not only from NaN/inf.
inline bool isDivergent(float response) { return !std::isfinite(response) || std::abs(response) > 1e6f; }

// How the reward is computed from the target on each training tick.
enum class RewardMode
{
    Target,  // always the desired sign: weights keep growing even when already right
    Error    // the desired sign only when the output's sign is wrong, else 0
             // (a reward prediction error; lets thresholds/bias be learned)
};

// Trains on one stream, then scores on a fresh one with learning frozen.
// learningRate = 0 gives the control: same network seed and training stream, no learning.
inline PatternScore runPatternTrial(const NetworkBuilder& build, std::uint32_t seed, bool hasER,
                                    float learningRate, size_t trainTicks = PATTERN_TRAIN_TICKS,
                                    RewardMode mode = RewardMode::Target)
{
    reseed(seed);
    const std::unique_ptr<network> net = build(hasER);

    // The reward is the desired sign of the response (see the Pavlovian test).
    // Positive ticks are much rarer, so their reward is scaled up to give both
    // classes the same total weight.
    const PatternStream train = makePatternStream(1000 + seed, trainTicks);
    const auto positives = std::count(train.target.begin(), train.target.end(), +1);
    const float pos_weight = static_cast<float>(trainTicks - positives) / static_cast<float>(positives);
    bool diverged = false;
    for (size_t t = 0; t < train.values.size(); ++t) {
        const float response = patternStep(*net, train.values[t]);
        diverged = diverged || isDivergent(response);
        const bool positive = train.target[t] > 0;
        if (mode == RewardMode::Error && (response > 0.0f) == positive)
            continue;  // already right: no reward
        net->applyReward(positive ? pos_weight : -1.0f, learningRate);
    }

    const PatternStream test = makePatternStream(2000 + seed, PATTERN_TEST_TICKS);
    std::vector<float> responses;
    for (float x : test.values) {
        responses.push_back(patternStep(*net, x));
        diverged = diverged || isDivergent(responses.back());
    }
    PatternScore score = scorePredictions(test, [&](size_t t) { return responses[t] > 0.0f; });
    score.diverged = diverged;
    return score;
}

// Mean and standard error of a per-trial measure, plus paired difference to the control.
struct Measure
{
    float mean = 0.0f, mean_control = 0.0f;
    float stderr_mean = 0.0f;
    float diff = 0.0f, stderr_diff = 0.0f;  // trained - control, paired by seed
};

inline float standardError(const std::vector<float>& values, float mean)
{
    float var = 0.0f;
    for (float v : values) var += (v - mean) * (v - mean) / static_cast<float>(values.size() - 1);
    return std::sqrt(var / static_cast<float>(values.size()));
}

inline float meanOf(const std::vector<float>& values)
{
    float sum = 0.0f;
    for (float v : values) sum += v;
    return sum / static_cast<float>(values.size());
}

inline Measure summarize(const std::vector<float>& trained, const std::vector<float>& control)
{
    Measure m;
    std::vector<float> diffs;
    for (size_t i = 0; i < trained.size(); ++i) diffs.push_back(trained[i] - control[i]);
    m.mean = meanOf(trained);
    m.mean_control = meanOf(control);
    m.stderr_mean = standardError(trained, m.mean);
    m.diff = meanOf(diffs);
    m.stderr_diff = standardError(diffs, m.diff);
    return m;
}

struct PatternStats
{
    Measure balanced;
    Measure afterC;
    int diverged = 0, diverged_control = 0;
};

inline PatternStats measurePattern(const NetworkBuilder& build, bool hasER, float learningRate,
                                   int trials = PATTERN_TRIALS, size_t trainTicks = PATTERN_TRAIN_TICKS,
                                   RewardMode mode = RewardMode::Target)
{
    PatternStats st;
    std::vector<float> bal, bal_c, aft, aft_c;
    for (int trial = 0; trial < trials; ++trial) {
        const auto seed = static_cast<std::uint32_t>(trial);
        const PatternScore r = runPatternTrial(build, seed, hasER, learningRate, trainTicks, mode);
        const PatternScore c = runPatternTrial(build, seed, hasER, 0.0f, trainTicks, mode);
        bal.push_back(r.balanced);
        bal_c.push_back(c.balanced);
        aft.push_back(r.afterC);
        aft_c.push_back(c.afterC);
        st.diverged += r.diverged;
        st.diverged_control += c.diverged;
    }
    st.balanced = summarize(bal, bal_c);
    st.afterC = summarize(aft, aft_c);
    return st;
}

inline float tStatistic(float effect, float stderr_value)
{
    if (stderr_value == 0.0f)
        return effect == 0.0f ? 0.0f : std::copysign(INFINITY, effect);
    return effect / stderr_value;
}

inline const char* verdict(float t, const char* better, const char* worse)
{
    if (t >= SIGNIFICANCE_T) return better;
    if (t <= -SIGNIFICANCE_T) return worse;
    return "NOISE - no detectable difference";
}

// Shortcut rules that ignore (part of) the sequence, scored on the same test
// streams: a network has only learned the sequence if it beats them on the
// after-C measure.
// Prints results with 3 significant digits within a scope, then restores
// std::cout, so the format never depends on which tests ran before.
struct ThreeDigits
{
    std::streamsize saved = std::cout.precision(3);
    ~ThreeDigits() { std::cout.precision(saved); }
};

inline void printShortcutBaselines(int trials = PATTERN_TRIALS)
{
    ThreeDigits digits;
    struct Rule { const char* name; std::function<bool(const std::vector<float>&, size_t)> fires; };
    const Rule rules[] = {
        {"respond after any C", [](const std::vector<float>& v, size_t t) {
             for (int k = 1; k <= PATTERN_HOLD; ++k)
                 if (t >= static_cast<size_t>(k) && v[t - k] == PATTERN_C) return true;
             return false; }},
        {"respond after C with B 1..n+1 ticks before it", [](const std::vector<float>& v, size_t t) {
             for (int k = 1; k <= PATTERN_HOLD; ++k) {
                 if (t < static_cast<size_t>(k) || v[t - k] != PATTERN_C) continue;
                 for (int g = 0; g <= PATTERN_MAX_GAP; ++g) {
                     const long j = static_cast<long>(t) - k - 1 - g;
                     if (j >= 0 && v[j] == PATTERN_B) return true;
                 }
             }
             return false; }},
    };
    std::cout << " Shortcut rules (" << trials << " test streams):\n";
    for (const Rule& rule : rules) {
        float bal = 0.0f, aft = 0.0f;
        for (int trial = 0; trial < trials; ++trial) {
            const PatternStream s = makePatternStream(2000 + trial, PATTERN_TEST_TICKS);
            const PatternScore sc = scorePredictions(s, [&](size_t t) { return rule.fires(s.values, t); });
            bal += sc.balanced / trials;
            aft += sc.afterC / trials;
        }
        std::cout << "   balanced " << bal << ", after C " << aft << "  <- " << rule.name << "\n";
    }
}

inline void printMeasure(const char* name, const Measure& m, const char* chance_good)
{
    ThreeDigits digits;
    const float t_control = tStatistic(m.diff, m.stderr_diff);
    const float t_chance = tStatistic(m.mean - 0.5f, m.stderr_mean);
    std::cout << "   " << name << ": " << m.mean << " (control " << m.mean_control << "), trained - control "
              << m.diff << " +- " << m.stderr_diff << "\n"
              << "     vs control: t = " << t_control << " -> "
              << verdict(t_control, "LEARNING - trained beats the no-learning control",
                         "HARMFUL - learning makes it worse than the control") << "\n"
              << "     vs chance:  t = " << t_chance << " -> "
              << verdict(t_chance, chance_good, "BELOW CHANCE - systematically wrong") << "\n";
}

inline void printPatternStats(const char* title, const PatternStats& st, int trials = PATTERN_TRIALS)
{
    ThreeDigits digits;
    std::cout << " " << title << ": diverged " << st.diverged << "/" << trials
              << " (control " << st.diverged_control << ")\n";
    printMeasure("balanced accuracy (all ticks)", st.balanced, "ABOVE CHANCE - reacts to the pattern");
    printMeasure("accuracy after C (valid vs decoy)", st.afterC, "ABOVE CHANCE - uses the A..B..C history");
    std::cout << "   (|t| < " << SIGNIFICANCE_T << " is treated as noise, ~95% confidence)\n";
}

// One-line topology for result tables: layers in update order as
// name(neurons + flags), then feedback edges. Flags: * frozen, E E-R on,
// h habituation on, r recovery jitter, l learning jitter, a alpha jitter. Runs of numbered layers with identical settings
// (tap5, tap4, ...) are collapsed.
inline std::string topologySummary(const network& net)
{
    struct Item { std::string prefix, first, last, flags; size_t size; };
    std::vector<Item> items;
    for (network::LayerId id : net.updateOrder()) {
        const std::string& name = net.layerName(id);
        const LayerSpec& spec = net.layerSpec(id);
        std::string flags = std::string(spec.frozen ? "*" : "") + (spec.hasER ? "E" : "") + (spec.hasHabituation ? "h" : "")
                            + (spec.recoveryJitter.enabled() ? "r" : "") + (spec.learningJitter.enabled() ? "l" : "")
                            + (spec.alphaJitter.enabled() ? "a" : "");
        const size_t size = net.getLayer(id).size();
        const size_t digits = name.find_last_not_of("0123456789") + 1;
        const std::string prefix = digits < name.size() ? name.substr(0, digits) : name + "#";
        if (!items.empty() && items.back().prefix == prefix && items.back().flags == flags && items.back().size == size)
            items.back().last = name;
        else
            items.push_back({prefix, name, name, flags, size});
    }
    std::string out;
    for (const Item& item : items) {
        if (!out.empty()) out += " > ";
        out += item.first == item.last ? item.first : item.first + ".." + item.last;
        out += "(" + std::to_string(item.size) + item.flags + ")";
    }
    for (const network::Edge& e : net.edges())
        if (e.kind == network::EdgeKind::Feedback && net.layerName(e.to) != "window")
            out += "; fb " + net.layerName(e.from) + "->" + net.layerName(e.to) + "(" + std::to_string(e.width) + ")";
    return out;
}

// Builds one instance of the topology and prints its full description.
inline void printTopology(const NetworkBuilder& build, bool hasER)
{
    reseed(0);
    const std::unique_ptr<network> net = build(hasER);
    std::cout << " Topology:\n";
    net->describe(std::cout);
}

inline void printPatternHeader(const char* title, const NetworkBuilder& build, bool hasER,
                               int trials = PATTERN_TRIALS)
{
    const PatternStream sample = makePatternStream(2000, PATTERN_TEST_TICKS);
    const auto positives = std::count(sample.target.begin(), sample.target.end(), +1);
    const auto after_c = std::count(sample.afterC.begin(), sample.afterC.end(), true);
    std::cout << "\n==========================================\n"
              << " [Gapped pattern A,{0.." << PATTERN_MAX_GAP << "},B,{0.." << PATTERN_MAX_GAP
              << "},C, " << title << " - " << trials << " trials]\n"
              << " test stream: " << positives << " positive ticks, " << after_c << " ticks after a C, of "
              << PATTERN_TEST_TICKS << "\n"
              << "==========================================\n";
    printTopology(build, hasER);
    printShortcutBaselines(trials);
}

} // namespace pattern_benchmark
