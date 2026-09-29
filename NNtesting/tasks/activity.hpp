#pragma once
// Shared by the activity-economy experiments (er_economy, er_paths,
// er_fatigue, er_history): a classification task, a network with several
// alternative hidden paths between its input and its readouts, and meters
// for how much of the network is active and which path drives the output.
//
//              +-- path A (12 neurons) --+
//   input -----+-- path B (20 neurons) --+-- one readout per class
//              +-- path C (35 neurons) --+
//   stim_A ----^   stim_B ----^   stim_C ----^   (per-path stimulation, silent during the task)
//
// Only the hidden paths' neuron dynamics differ between the compared models:
//   er      E-R neurons: an adaptive threshold that rises with firing and relaxes back
//   gate    a fixed threshold (neuron::gate): all-or-nothing like E-R, but nothing adapts
//   linear  no threshold: the output is the weighted sum
// Readouts are linear in every model. No habituation (by default), and no
// activity or sparsity term anywhere: learning sees only task errors.
#include <algorithm>
#include <array>
#include "er_options.hpp"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "experiment.hpp"
#include "layers/dense.hpp"
#include "network.hpp"

namespace activity {

using namespace exr;

// K classes, each a fixed random +-1 pattern over the sensors; a sample is
// its pattern plus uniform +-noise on every sensor.
struct Prototypes
{
    size_t inputs = 32, classes = 4;
    double noise = 0.5;
    std::vector<std::vector<float>> patterns;

    Prototypes(size_t inputs_, size_t classes_, double noise_, std::uint32_t seed)
        : inputs(inputs_), classes(classes_), noise(noise_)
    {
        std::mt19937 g(seed);
        std::uniform_int_distribution<int> coin(0, 1);
        patterns.assign(classes, std::vector<float>(inputs));
        for (auto& p : patterns)
            for (float& v : p)
                v = coin(g) ? 1.0f : -1.0f;
    }

    // A sample of class `k` into x.
    void sample(std::mt19937& g, size_t k, std::vector<float>& x) const
    {
        std::uniform_real_distribution<double> unit(-1.0, 1.0);
        x = patterns[k];
        for (float& v : x)
            v += static_cast<float>(noise * unit(g));
    }
    size_t draw(std::mt19937& g, std::vector<float>& x) const
    {
        const size_t k = std::uniform_int_distribution<size_t>(0, classes - 1)(g);
        sample(g, k, x);
        return k;
    }
};

// Parameters every activity experiment takes.
inline std::vector<nnt::ParamSpec> commonParams(std::vector<nnt::ParamSpec> extra = {})
{
    std::vector<nnt::ParamSpec> p = {
        {"inputs", "32", "sensors"},
        {"classes", "4", "classes (one readout each)"},
        {"noise", "1.0", "uniform +- noise on every sensor"},
        {"hold", "4", "ticks each sample is shown (the readouts' outputs are summed over them)"},
        {"paths", "12,20,35", "neurons in each alternative path (named A, B, C, ...)"},
        {"stim", "8", "stimulation sensors per path"},
        {"gate", "match", "gate model: fixed firing threshold, or match: the threshold at which the untrained "
                          "linear network is as active as the trained E-R one"},
        {"recovery", "0.9", "E-R threshold decay per tick without firing (larger: slower recovery)"},
        {"habituation", "false", "habituation in the hidden paths"},
        {"habituation_steps", "100", "habituation: ticks of the same input before it is suppressed"},
        {"habituation_tolerance", "0", "habituation: relative change still counted as the same input (0: exact)"},
        {"habituation_decay", "0", "habituation: suppressed input scaled by decay per tick (0: cut at once)"},
        {"spontaneous_below", "1e-10", "E-R: a silent neuron fires spontaneously once its threshold is at or below this"},
        {"spontaneous_amplitude", "0.1", "E-R: spontaneous output drawn uniformly in +- this"},
        {"spontaneous_rate", "0", "E-R: extra probability of a spontaneous firing on any silent tick"},
        {"habituation_fade_after", "2", "habituation with a decay: repeats before fading starts"},
        {"normalize", "true", "hidden paths divide each weighted sum by the length of the neuron's weights"},
        {"readout_normalize", "false", "the class readouts divide their weighted sums by the length of their weights"},
        {"resting_threshold", "0.2", "E-R resting threshold of the paths, or auto: 0.2 times the paths' mean 1/|w| "
                                     "(recalibrated for normalised sums)"},
        {"growth", "linear", "E-R threshold growth on firing: linear (default), log (original), fixed, multiplicative"},
        {"growth_amount", "0.5", "E-R threshold growth amount (linear, fixed, multiplicative)"},
        {"learning", "fa", "fa: paths and readouts learn from the task errors (feedback alignment, delta rule); "
                           "readout: paths frozen, readouts learn (sign rule, error-driven)"},
        {"lr", "0.003", "learning rate (0.0003 before the normalised default: a normalised sum moves about 10x less per update)"},
        {"train", "1500", "training samples"},
        {"test", "300", "test samples"},
    };
    p.insert(p.end(), extra.begin(), extra.end());
    return p;
}

inline std::vector<size_t> parseSizes(const std::string& text)
{
    std::vector<size_t> sizes;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ','))
        sizes.push_back(static_cast<size_t>(std::stoul(item)));
    if (sizes.empty() || sizes.size() > 26 || std::ranges::find(sizes, size_t{0}) != sizes.end())
        throw std::invalid_argument("paths: a comma-separated list of 1 to 26 positive sizes");
    return sizes;
}

// The network: paths between one input and the readouts.
class PathNet
{
public:
    // model: "er", "gate" or "linear". `gate`: the gate model's threshold
    // (default: the gate parameter, which must then be a number).
    PathNet(const nnt::Params& p, const std::string& model, float gate = -1.0f) : model_(model)
    {
        if (model != "er" && model != "gate" && model != "linear")
            throw std::invalid_argument("model must be er, gate or linear");
        const size_t inputs = static_cast<size_t>(p.getInt("inputs"));
        classes_ = static_cast<size_t>(p.getInt("classes"));
        const size_t stim = static_cast<size_t>(p.getInt("stim"));
        learning_ = p.getString("learning");
        if (learning_ != "fa" && learning_ != "readout")
            throw std::invalid_argument("learning must be fa or readout");
        const bool fa = learning_ == "fa";
        const bool habituation = p.getBool("habituation");

        net_ = std::make_unique<network>();
        const std::vector<size_t> sizes = parseSizes(p.getString("paths"));
        for (size_t i = 0; i < sizes.size(); ++i) {
            LayerSpec spec = LayerSpec::Dense(sizes[i], habituation, model == "er");
            spec.habituationRule = {static_cast<std::uint32_t>(p.getInt("habituation_steps")),
                                    static_cast<float>(p.getDouble("habituation_tolerance")),
                                    static_cast<float>(p.getDouble("habituation_decay")),
                                    static_cast<std::uint32_t>(p.getInt("habituation_fade_after"))};
            spec.normalize = p.getBool("normalize");
            if (model == "gate")
                spec.gate = gate >= 0.0f ? gate : static_cast<float>(p.getDouble("gate"));
            if (model == "er") {
                spec.thresholdGrowth = er_options::thresholdGrowth(p.getString("growth"), p.getDouble("growth_amount"));
                spec.spontaneous = er_options::spontaneous(p.getDouble("spontaneous_below"),
                                                           p.getDouble("spontaneous_amplitude"),
                                                           p.getDouble("spontaneous_rate"));
            }
            const float recovery = static_cast<float>(p.getDouble("recovery"));
            if (model == "er" && recovery != recovery_factor)  // a spread too small to matter: every neuron gets it
                spec.recoveryJitter = Jitter::uniform(1e-7f).around(recovery);
            if (fa)
                spec.learningRule = LearningRule::feedbackAlignment();
            spec.frozen = !fa;
            const std::string name(1, static_cast<char>('A' + i));
            names_.push_back(name);
            paths_.push_back(net_->addLayer(name, spec));
            first_.push_back(hidden_);
            hidden_ += sizes[i];
        }
        net_->addInputs(paths_[0], inputs, "input");
        for (size_t i = 1; i < paths_.size(); ++i)
            net_->connectInputs("input", paths_[i]);
        for (size_t i = 0; i < paths_.size(); ++i)
            net_->addInputs(paths_[i], stim, "stim_" + names_[i]);
        LayerSpec readout = LayerSpec::Dense(1, false, false);
        readout.normalize = p.getBool("readout_normalize");  // the builder turns it on
        if (fa)
            readout.learningRule = LearningRule::feedbackAlignment();
        for (size_t k = 0; k < classes_; ++k) {
            const auto r = net_->addLayer("class" + std::to_string(k), readout);
            for (auto path : paths_)
                net_->connect(path, r);
            net_->addOutput(r);
            readouts_.push_back(r);
        }
        stimValues_.assign(stim, 0.0f);
        pathOf_.resize(hidden_);
        for (size_t i = 0; i < paths_.size(); ++i)
            std::fill_n(pathOf_.begin() + static_cast<std::ptrdiff_t>(first_[i]), sizes[i], i);
        if (model == "er")
            for (auto path : paths_)
                restingFactor_ = er_options::calibrateRestingThreshold(net_->layerAs<neuron_layer>(path),
                                                                       p.getString("resting_threshold"));
        refreshWeights();
    }
    // Resting threshold / the default after calibration (last path; 1 without E-R).
    float restingFactor() const { return restingFactor_; }

    const std::string& model() const { return model_; }
    size_t pathCount() const { return paths_.size(); }
    const std::string& pathName(size_t i) const { return names_[i]; }
    size_t pathSize(size_t i) const { return net_->getLayer(paths_[i]).size(); }
    size_t hidden() const { return hidden_; }
    size_t classes() const { return classes_; }
    size_t pathOf(size_t neuron) const { return pathOf_[neuron]; }
    network& net() { return *net_; }

    // The output of hidden neuron `i` (paths in order).
    float hiddenOutput(size_t i) const
    {
        const size_t p = pathOf_[i];
        return net_->getLayer(paths_[p]).output()[i - first_[p]];
    }
    float threshold(size_t i) const
    {
        const size_t p = pathOf_[i];
        return net_->layerAs<neuron_layer>(paths_[p]).neurons()[i - first_[p]].threshold();
    }

    // Readout k's weight on hidden neuron i (the readouts read the paths in order).
    float readoutWeight(size_t k, size_t i) const { return weights_[k * hidden_ + i]; }
    void refreshWeights()
    {
        weights_.assign(classes_ * hidden_, 0.0f);
        for (size_t k = 0; k < classes_; ++k) {
            const std::vector<float> w = net_->layerAs<dense>(readouts_[k]).weights(0);
            std::copy_n(w.begin(), std::min(w.size(), hidden_), weights_.begin() + static_cast<std::ptrdiff_t>(k * hidden_));
        }
    }

    // One tick with sensor values x (empty: silence); `stimulated` paths get
    // random +-amplitude on their stimulation sensors, the others silence.
    std::vector<float> tick(std::span<const float> x, const std::vector<bool>& stimulated = {}, float amplitude = 0.0f,
                            std::mt19937* g = nullptr)
    {
        if (x.empty()) {
            silence_.assign(net_->inputSource("input").size(), 0.0f);
            net_->setInputs("input", silence_);
        } else {
            net_->setInputs("input", x);
        }
        for (size_t i = 0; i < paths_.size(); ++i) {
            const bool on = i < stimulated.size() && stimulated[i];
            for (float& v : stimValues_)
                v = on ? (std::uniform_int_distribution<int>(0, 1)(*g) ? amplitude : -amplitude) : 0.0f;
            net_->setInputs("stim_" + names_[i], stimValues_);
        }
        net_->step();
        return net_->outputs();
    }

    // Learning from one tick's outputs y for class k (targets +1 / -1).
    void learn(const std::vector<float>& y, size_t k, float lr)
    {
        if (learning_ == "fa") {
            std::vector<float> errors(classes_);
            for (size_t r = 0; r < classes_; ++r)
                errors[r] = (r == k ? 1.0f : -1.0f) - y[r];
            net_->applyError(errors, lr);
        } else {
            for (size_t r = 0; r < classes_; ++r) {
                const float target = r == k ? 1.0f : -1.0f;
                if (y[r] * target <= 0.0f)
                    net_->getLayer(readouts_[r]).applyReward(target, lr);
            }
        }
    }

    // A copy with the whole state (weights, thresholds, traces).
    std::string snapshot() const
    {
        std::ostringstream os(std::ios::binary);
        net_->save(os);
        return os.str();
    }
    void restore(const std::string& state)
    {
        std::istringstream is(state, std::ios::binary);
        net_ = network::load(is);
        refreshWeights();
    }

private:
    float restingFactor_ = 1.0f;
    std::string model_, learning_;
    std::unique_ptr<network> net_;
    std::vector<network::LayerId> paths_, readouts_;
    std::vector<std::string> names_;
    std::vector<size_t> first_, pathOf_;
    size_t hidden_ = 0, classes_ = 0;
    std::vector<float> weights_, stimValues_, silence_;
};

// A neuron counts as active in a tick when its output is non-zero; E-R's
// spontaneous firings (uniform in +-0.01, after a long silence) are counted
// separately: any active output that small (task sums are far larger).
inline bool active(float y) { return std::abs(y) > firing_epsilon; }
inline bool spontaneous(float y) { return active(y) && std::abs(y) <= spontaneous_min_amplitude; }

// Activity over a window of ticks.
struct Summary
{
    double ticks = 0;
    double activePerTick = 0;       // active hidden neurons per tick
    double activeFraction = 0;      // ... as a fraction of the hidden neurons
    double spontaneousPerTick = 0;  // of which spontaneous E-R firings
    double massPerTick = 0;         // sum of |output| per tick
    double uniqueActive = 0;        // fraction of hidden neurons active at least once
    double neverActive = 0;         // 1 - uniqueActive
    double meanRun = 0;             // mean length of an unbroken active period, in ticks
    double entropy = 0;             // of the neurons' share of all activity, / log(neurons): 1 = even use
    double switchRate = 0;          // changes of the leading path (most drive per neuron) per tick
    double internalSwitchRate = 0;  // ... of which while the input stayed the same (not on a new sample's first tick)
    double meanDominance = 0;       // mean ticks a path stays in the lead
    std::vector<double> pathActive; // active neurons per tick, per path
    std::vector<double> pathActiveFraction;
    std::vector<double> pathDrive;  // share of the readouts' input |w y| from each path
    std::vector<double> pathThreshold;  // mean E-R threshold per path (baseline_threshold without E-R)
    std::vector<double> pathDominant;   // fraction of ticks each path leads (most drive per neuron)
};

class Meter
{
public:
    explicit Meter(const PathNet& net) : net_(net) { reset(); }

    void reset()
    {
        const size_t n = net_.hidden(), P = net_.pathCount();
        ticks_ = 0;
        activeTicks_.assign(n, 0);
        run_.assign(n, 0);
        runs_ = runTicks_ = 0;
        spont_ = mass_ = 0;
        pathActive_.assign(P, 0);
        pathDrive_.assign(P, 0);
        pathThreshold_.assign(P, 0);
        pathDominant_.assign(P, 0);
        dominant_ = SIZE_MAX;
        switches_ = internalSwitches_ = 0;
        dominanceRuns_ = 0;
        sampleStart_ = true;
    }

    // Call before the first tick of each new input: lead changes on that
    // tick are the input's doing, the others the network's own.
    void newSample() { sampleStart_ = true; }

    // Call after every tick of the window.
    void observe()
    {
        const size_t n = net_.hidden(), P = net_.pathCount();
        std::vector<double> drive(P, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const float y = net_.hiddenOutput(i);
            const size_t p = net_.pathOf(i);
            pathThreshold_[p] += net_.threshold(i);
            if (active(y)) {
                ++activeTicks_[i];
                ++run_[i];
                ++pathActive_[p];
                spont_ += spontaneous(y);
                mass_ += std::abs(y);
                for (size_t k = 0; k < net_.classes(); ++k)
                    drive[p] += std::abs(net_.readoutWeight(k, i) * y);
            } else if (run_[i] > 0) {
                ++runs_;
                runTicks_ += run_[i];
                run_[i] = 0;
            }
        }
        const double total = std::accumulate(drive.begin(), drive.end(), 0.0);
        if (total > 0) {
            for (size_t p = 0; p < P; ++p)
                pathDrive_[p] += drive[p] / total;
            // The leading path: the most drive per neuron, so a small path can lead.
            size_t top = 0;
            for (size_t p = 1; p < P; ++p)
                if (drive[p] / static_cast<double>(net_.pathSize(p)) >
                    drive[top] / static_cast<double>(net_.pathSize(top)))
                    top = p;
            ++pathDominant_[top];
            if (top != dominant_) {
                if (dominant_ != SIZE_MAX) {
                    ++switches_;
                    internalSwitches_ += !sampleStart_;
                }
                ++dominanceRuns_;
                dominant_ = top;
            }
        }
        sampleStart_ = false;
        ++ticks_;
    }

    Summary summary() const
    {
        Summary s;
        const size_t n = net_.hidden(), P = net_.pathCount();
        const double T = std::max<double>(1, static_cast<double>(ticks_));
        s.ticks = static_cast<double>(ticks_);
        double active_sum = 0, unique = 0, entropy = 0;
        for (size_t i = 0; i < n; ++i) {
            active_sum += static_cast<double>(activeTicks_[i]);
            unique += activeTicks_[i] > 0;
        }
        for (size_t i = 0; i < n; ++i)
            if (activeTicks_[i] > 0) {
                const double q = static_cast<double>(activeTicks_[i]) / active_sum;
                entropy -= q * std::log(q);
            }
        // Unfinished runs count as runs of their current length.
        double runs = static_cast<double>(runs_), run_ticks = static_cast<double>(runTicks_);
        for (size_t r : run_)
            if (r > 0) {
                ++runs;
                run_ticks += static_cast<double>(r);
            }
        s.activePerTick = active_sum / T;
        s.activeFraction = s.activePerTick / static_cast<double>(n);
        s.spontaneousPerTick = spont_ / T;
        s.massPerTick = mass_ / T;
        s.uniqueActive = unique / static_cast<double>(n);
        s.neverActive = 1.0 - s.uniqueActive;
        s.meanRun = runs > 0 ? run_ticks / runs : 0.0;
        s.entropy = n > 1 ? entropy / std::log(static_cast<double>(n)) : 0.0;
        s.switchRate = static_cast<double>(switches_) / T;
        s.internalSwitchRate = static_cast<double>(internalSwitches_) / T;
        s.meanDominance = dominanceRuns_ > 0 ? T / static_cast<double>(dominanceRuns_) : 0.0;
        for (size_t p = 0; p < P; ++p) {
            s.pathActive.push_back(pathActive_[p] / T);
            s.pathActiveFraction.push_back(pathActive_[p] / T / static_cast<double>(net_.pathSize(p)));
            s.pathDrive.push_back(pathDrive_[p] / T);
            s.pathThreshold.push_back(pathThreshold_[p] / T / static_cast<double>(net_.pathSize(p)));
            s.pathDominant.push_back(pathDominant_[p] / T);
        }
        return s;
    }

private:
    const PathNet& net_;
    size_t ticks_ = 0;
    std::vector<size_t> activeTicks_, run_;
    size_t runs_ = 0, runTicks_ = 0;
    double spont_ = 0, mass_ = 0;
    std::vector<double> pathActive_, pathDrive_, pathThreshold_, pathDominant_;
    size_t dominant_ = SIZE_MAX, switches_ = 0, internalSwitches_ = 0, dominanceRuns_ = 0;
    bool sampleStart_ = true;
};

// One sample shown for `hold` ticks: the decision is the class whose readout
// summed the most over the ticks; latency is the tick (1-based) from which
// the running decision no longer changed.
struct Response
{
    size_t decision = 0;
    size_t latency = 0;
    std::vector<float> evidence;  // summed readout outputs
};

// Shows sample x for `hold` ticks, observing each tick with `meter` (if any)
// and learning at every tick when lr > 0.
inline Response present(PathNet& net, std::span<const float> x, size_t k, size_t hold, float lr, Meter* meter)
{
    Response r;
    r.evidence.assign(net.classes(), 0.0f);
    size_t last = SIZE_MAX;
    if (meter)
        meter->newSample();
    for (size_t t = 0; t < hold; ++t) {
        const std::vector<float> y = net.tick(x);
        if (meter)
            meter->observe();
        for (size_t c = 0; c < y.size(); ++c)
            r.evidence[c] += y[c];
        const auto now = static_cast<size_t>(std::ranges::max_element(r.evidence) - r.evidence.begin());
        if (now != last) {
            r.latency = t + 1;
            last = now;
        }
        if (lr > 0.0f)
            net.learn(y, k, lr);
    }
    r.decision = last;
    return r;
}

// Plays `count` samples; returns the accuracy. Learns when lr > 0 (the
// readout weights the meter uses are refreshed every `refresh` samples).
inline double stream(PathNet& net, const Prototypes& task, std::mt19937& g, int count, size_t hold, float lr,
                     Meter* meter, double* latency = nullptr, int refresh = 50)
{
    std::vector<float> x;
    int correct = 0;
    double latency_sum = 0;
    for (int i = 0; i < count; ++i) {
        if (lr > 0.0f && meter && i % refresh == 0)
            net.refreshWeights();
        const size_t k = task.draw(g, x);
        const Response r = present(net, x, k, hold, lr, meter);
        correct += r.decision == k;
        latency_sum += static_cast<double>(r.latency);
    }
    if (lr > 0.0f)
        net.refreshWeights();
    if (latency)
        *latency = latency_sum / std::max(1, count);
    return static_cast<double>(correct) / std::max(1, count);
}

// Records a summary's metrics with `prefix` (e.g. "er_").
inline void record(nnt::Trial& t, const std::string& prefix, const PathNet& net, const Summary& s)
{
    t.record(prefix + "active_per_tick", s.activePerTick);
    t.record(prefix + "active_fraction", s.activeFraction);
    t.record(prefix + "spontaneous_per_tick", s.spontaneousPerTick);
    t.record(prefix + "mass_per_tick", s.massPerTick);
    t.record(prefix + "unique_active", s.uniqueActive);
    t.record(prefix + "never_active", s.neverActive);
    t.record(prefix + "mean_active_run", s.meanRun);
    t.record(prefix + "entropy", s.entropy);
    t.record(prefix + "switch_rate", s.switchRate);
    t.record(prefix + "internal_switch_rate", s.internalSwitchRate);
    t.record(prefix + "mean_lead_ticks", s.meanDominance);
    for (size_t p = 0; p < net.pathCount(); ++p) {
        const std::string name = prefix + "path" + net.pathName(p) + "_";
        t.record(name + "active_fraction", s.pathActiveFraction[p]);
        t.record(name + "drive", s.pathDrive[p]);
        // Drive share over size share: 1 = the path drives as much as its size.
        t.record(name + "drive_per_size",
                 s.pathDrive[p] * static_cast<double>(net.hidden()) / static_cast<double>(net.pathSize(p)));
        t.record(name + "lead", s.pathDominant[p]);
        t.record(name + "threshold", s.pathThreshold[p]);
    }
}

// The fixed threshold at which the linear version of the network is active
// in `fraction` of its neuron-ticks on the samples from `seed`.
inline float matchingGate(const nnt::Params& p, const Prototypes& task, std::uint32_t weightSeed,
                          std::uint32_t sampleSeed, double fraction)
{
    exr::reseed(weightSeed);
    PathNet net(p, "linear");
    std::mt19937 g(sampleSeed);
    std::vector<float> x, magnitudes;
    const size_t hold = static_cast<size_t>(p.getInt("hold"));
    for (int i = 0; i < 200; ++i) {
        task.draw(g, x);
        for (size_t t = 0; t < hold; ++t) {
            net.tick(x);
            for (size_t n = 0; n < net.hidden(); ++n)
                magnitudes.push_back(std::abs(net.hiddenOutput(n)));
        }
    }
    const auto q = static_cast<size_t>(std::clamp(1.0 - fraction, 0.0, 1.0) * static_cast<double>(magnitudes.size() - 1));
    std::nth_element(magnitudes.begin(), magnitudes.begin() + static_cast<std::ptrdiff_t>(q), magnitudes.end());
    return magnitudes[q];
}

// Conditions that set up a network state before a probe, for `ticks`
// ticks, with the task input silent unless the condition is "busy":
//   rest     nothing
//   busy     ordinary task samples (natural heavy use)
//   all      every path stimulated
//   A, B...  that path stimulated (random +-amplitude on its stimulation sensors)
inline std::vector<std::string> conditionNames(const PathNet& net)
{
    std::vector<std::string> names = {"rest", "busy", "all"};
    for (size_t p = 0; p < net.pathCount(); ++p)
        names.push_back(net.pathName(p));
    return names;
}

inline void condition(PathNet& net, const std::string& name, int ticks, float amplitude, const Prototypes& task,
                      std::mt19937& g, size_t hold)
{
    std::vector<bool> stimulated(net.pathCount(), false);
    if (name == "all")
        stimulated.assign(net.pathCount(), true);
    for (size_t p = 0; p < net.pathCount(); ++p)
        if (name == net.pathName(p))
            stimulated[p] = true;
    std::vector<float> x;
    for (int t = 0; t < ticks; ++t) {
        if (name == "busy") {
            if (t % static_cast<int>(hold) == 0)
                task.draw(g, x);
            net.tick(x);
        } else {
            net.tick({}, stimulated, amplitude, &g);
        }
    }
}

inline void stateParams(std::vector<nnt::ParamSpec>& p)
{
    p.push_back({"condition_ticks", "20", "ticks of each condition before the probe"});
    p.push_back({"amplitude", "10", "stimulation: +- value on each stimulation sensor (task sums are about 5)"});
}

// Optional per-block time course: one CSV row per call.
class Trace
{
public:
    Trace(const std::string& path, int trial) : enabled_(!path.empty())
    {
        if (!enabled_)
            return;
        const bool fresh = trial == 0;
        file_.open(path, fresh ? std::ios::trunc : std::ios::app);
        if (!file_)
            throw std::runtime_error("cannot write trace file " + path);
    }
    bool enabled() const { return enabled_; }
    void header(const PathNet& net, const std::string& extra, int trial)
    {
        if (!enabled_ || trial != 0)
            return;
        file_ << "trial,model,condition,block,tick," << extra << "accuracy,active_fraction,mass_per_tick,switch_rate";
        for (size_t p = 0; p < net.pathCount(); ++p)
            file_ << ",drive_" << net.pathName(p) << ",active_" << net.pathName(p) << ",threshold_" << net.pathName(p);
        file_ << "\n";
    }
    void row(int trial, const std::string& model, const std::string& condition, int block, double tick,
             const std::string& extra, double accuracy, const Summary& s)
    {
        if (!enabled_)
            return;
        file_ << trial << "," << model << "," << condition << "," << block << "," << tick << "," << extra << accuracy
              << "," << s.activeFraction << "," << s.massPerTick << "," << s.switchRate;
        for (size_t p = 0; p < s.pathDrive.size(); ++p)
            file_ << "," << s.pathDrive[p] << "," << s.pathActiveFraction[p] << "," << s.pathThreshold[p];
        file_ << "\n";
    }

private:
    bool enabled_;
    std::ofstream file_;
};

} // namespace activity
