// Dynamic nonlinearity substitution, milestone 1: for known static
// functions of increasing complexity, how large must a network be to reach
// a fixed test error, with conventional (ReLU), E-R, static-threshold or
// plain clamped hidden neurons? See doc/nonlinearity.md.
//
// One trial is one model, one architecture (depth x width) and one seed;
// sweep them with --set (e.g. --set model=relu,er,gate --set depth=1,2,3
// --set width=4,8,16). The trial seed draws the weights, so the same seed
// gives every model the same initial weights for an architecture. Task
// data comes from data_seed and the l4 coefficients from task_seed: both
// are the same for every model and seed.
//
// Protocol (identical for every model): each sample is held for
// depth + 1 + settle ticks and the output is read on the last one; while
// training, that output's error is applied once (feedback alignment, a
// learned bias per neuron, learning rate lr). Every eval_every samples the
// validation MSE is measured; training stops when it reaches target_mse
// or after `train` samples. Then the test set is presented once.
// success = test MSE <= target_mse.
//
// Metrics: the learning curve (curve_<samples>: validation MSE after that
// many training samples; best_validation_mse and where it was reached),
// test/validation MSE, normalized MSE (MSE / target variance),
// success, training samples used; depth, width, neurons, parameters;
// activity on the test set (active hidden neurons per tick, spikes per
// sample, unique neurons per sample, fraction used, never active); cost
// proxies (dense and event-driven synaptic operations per sample, wall time
// per inference); E-R threshold mean and quantiles; with state_probes, the
// state test (state_variance, state_bias2, state_share: the part of the error
// that comes from the history before a stimulus, state_output_range,
// state_spike_sd); the E-R constants and
// the l4 coefficients. No activity term is used anywhere in learning.
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include "experiment.hpp"
#include "nonlinearity.hpp"

namespace {

using namespace nonlinearity;

nnt::Register experiment({
    .name = "nl_static",
    .description = "static function approximation (l0..l4) with relu, E-R, static-threshold or clamped hidden "
                   "neurons on a depth x width grid: test MSE, success at a fixed error, size, activity, cost",
    .tags = {"er", "nonlinearity", "learning"},
    .params = {
        {"task", "l1", "l0 x1+x2, l1 x1*x2, l2 sin(x1*x2), l3 sin(x1*x2)+exp(-x3^2), l4 sum of k sines"},
        {"k", "1", "l4: number of sine components"},
        {"l4_inputs", "2", "l4: input dimension"},
        {"task_seed", "12345", "l4: seed of the coefficients"},
        {"data_seed", "1", "seed of the training stream (validation +1, test +2)"},
        {"model", "relu", "hidden neurons: relu, er, gate (fixed threshold), clamp (plain neuron; alias linear)"},
        {"gate", "0.2", "gate model: the fixed threshold (default: E-R's resting threshold, baseline_threshold)"},
        {"growth", "linear", "E-R threshold growth on firing: linear (default), log (original), fixed, multiplicative"},
        {"growth_amount", "0.5", "E-R threshold growth amount (linear, fixed, multiplicative)"},
        {"spontaneous_below", "1e-10", "E-R: a silent neuron fires spontaneously once its threshold is at or below this"},
        {"spontaneous_amplitude", "0.1", "E-R: spontaneous output drawn uniformly in +- this"},
        {"spontaneous_rate", "0", "E-R: extra probability of a spontaneous firing on any silent tick"},
        {"pretrain_model", "", "train first with these hidden neurons (relu, gate, clamp, er), then copy the weights into `model` (empty: no pretraining)"},
        {"pretrain", "0", "samples (steps) of pretraining with pretrain_model"},
        {"habituation", "false", "habituation in the hidden layers (every model)"},
        {"habituation_steps", "100", "habituation, cut mode: repeats before the input is cut"},
        {"habituation_tolerance", "0", "habituation: relative change still counted as a repeat (0: exact)"},
        {"habituation_decay", "0", "habituation: 0 cuts; a value in (0, 1] fades the input by that factor per repeat"},
        {"habituation_fade_after", "2", "habituation with a decay: repeats before fading starts"},
        {"normalize", "true", "hidden layers divide each weighted sum by the length of the neuron's weights"},
        {"readout_normalize", "false", "the readout divides its weighted sum by the length of its weights"},
        {"resting_threshold", "0.2", "E-R resting threshold of the hidden layers, or auto: 0.2 times their mean "
                                     "1/|w| at initialization, with the gate model's gate scaled the same way"},
        {"learn_ticks", "last", "training: learn from the last tick's error (last) or from every tick at lr/ticks (all)"},
        {"depth", "2", "hidden layers"},
        {"width", "16", "neurons per hidden layer"},
        {"lr", "0.01", "learning rate (the same for every model)"},
        {"settle", "1", "extra ticks per sample after the input reaches the output"},
        {"train", "20000", "maximum training samples"},
        {"eval_every", "1000", "training samples between validation checks"},
        {"validation", "500", "validation samples"},
        {"test", "1000", "test samples"},
        {"target_mse", "0.001", "success: test MSE at or below this; also the early-stopping target"},
        {"early_stop", "true", "stop training once the validation MSE reaches target_mse (false: use the whole budget)"},
        {"state_probes", "0", "state test: stimuli, each presented after state_histories different random histories (0: off)"},
        {"state_histories", "20", "state test: histories per stimulus"},
        {"state_history_len", "10", "state test: random samples in each history"},
        {"trace", "", "trace mode: CSV of every hidden neuron's output and threshold per tick (optional)"},
        {"trace_samples", "5", "trace mode: test samples to trace"},
    },
    .trials = 10,
    .run = [](nnt::Trial& t) {
        const auto& p = t.params();
        const StaticTask task(p.getString("task"), static_cast<size_t>(p.getInt("k")),
                              static_cast<size_t>(p.getInt("l4_inputs")),
                              static_cast<std::uint32_t>(p.getInt("task_seed")));
        const std::uint32_t dataSeed = static_cast<std::uint32_t>(p.getInt("data_seed"));
        const std::string model = p.getString("model");
        const size_t depth = static_cast<size_t>(p.getInt("depth")), width = static_cast<size_t>(p.getInt("width"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const size_t train = static_cast<size_t>(p.getInt("train"));
        const size_t evalEvery = std::max<size_t>(1, static_cast<size_t>(p.getInt("eval_every")));
        const double target = p.getDouble("target_mse");

        std::optional<Habituation> habituationRule;
        if (p.getBool("habituation"))
            habituationRule = Habituation{static_cast<std::uint32_t>(p.getInt("habituation_steps")),
                                          static_cast<float>(p.getDouble("habituation_tolerance")),
                                          static_cast<float>(p.getDouble("habituation_decay")),
                                          static_cast<std::uint32_t>(p.getInt("habituation_fade_after"))};
        const Spontaneous spontaneousSetting = er_options::spontaneous(
            p.getDouble("spontaneous_below"), p.getDouble("spontaneous_amplitude"), p.getDouble("spontaneous_rate"));
        Mlp net(model, task.inputs, depth, width, static_cast<float>(p.getDouble("gate")),
                er_options::thresholdGrowth(p.getString("growth"), p.getDouble("growth_amount")), spontaneousSetting,
                habituationRule, p.getBool("normalize"), p.getBool("readout_normalize"));
        const float restingFactor = net.calibrateThresholds(p.getString("resting_threshold"));
        const std::string learnTicks = p.getString("learn_ticks");
        if (learnTicks != "last" && learnTicks != "all")
            throw std::invalid_argument("learn_ticks must be last or all");
        const size_t hold = net.hold(static_cast<size_t>(p.getInt("settle")));
        const Dataset validation(task, static_cast<size_t>(p.getInt("validation")), dataSeed + 1);
        const Dataset test(task, static_cast<size_t>(p.getInt("test")), dataSeed + 2);

        // The network being trained or measured: `net`, or while pretraining
        // the pretraining network.
        Mlp* cur = &net;
        auto mseOf = [&](const Dataset& d, Activity* meter, std::ostream* trace, size_t traced) {
            double sum = 0.0;
            for (size_t i = 0; i < d.y.size(); ++i) {
                std::ostream* tr = i < traced ? trace : nullptr;
                const std::string prefix = tr ? model + ',' + std::to_string(depth) + ',' + std::to_string(width) + ',' +
                                                    std::to_string(t.seed()) + ',' + std::to_string(i)
                                              : std::string();
                const float y = cur->present(d.x[i], hold, meter, tr, prefix);
                const double e = static_cast<double>(y) - d.y[i];
                sum += e * e;
            }
            return d.y.empty() ? 0.0 : sum / static_cast<double>(d.y.size());
        };

        // Training stream: the same samples, in the same order, for every model.
        std::mt19937 g(dataSeed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> x(task.inputs);
        auto trainSamples = [&](size_t n) {
            for (size_t i = 0; i < n; ++i) {
                for (float& v : x)
                    v = u(g);
                if (learnTicks == "all")
                    cur->presentLearningEveryTick(x, hold, task(x), lr);
                else
                    cur->learn(task(x), cur->present(x, hold), lr);
            }
        };
        // Optional pretraining with other hidden neurons, on the start of the
        // same training stream; then `net` takes over the weights and biases.
        const std::string pretrainModel = p.getString("pretrain_model");
        const size_t pretrain = pretrainModel.empty() ? 0 : static_cast<size_t>(p.getInt("pretrain"));
        if (pretrain > 0) {
            Mlp pre(pretrainModel, task.inputs, depth, width, static_cast<float>(p.getDouble("gate")),
                    er_options::thresholdGrowth(p.getString("growth"), p.getDouble("growth_amount")), spontaneousSetting,
                habituationRule, p.getBool("normalize"), p.getBool("readout_normalize"));
            pre.calibrateThresholds(p.getString("resting_threshold"));
            cur = &pre;
            trainSamples(pretrain);
            const double preMse = mseOf(validation, nullptr, nullptr, 0);
            t.record("pretrain_validation_mse", std::isfinite(preMse) ? preMse : 1e30);
            net.copyFrom(pre);
            cur = &net;
            const double switched = mseOf(validation, nullptr, nullptr, 0);
            t.record("switch_validation_mse", std::isfinite(switched) ? switched : 1e30);
        }
        t.record("pretrain_samples", static_cast<double>(pretrain));
        double validationMse = mseOf(validation, nullptr, nullptr, 0);
        t.record("initial_validation_mse", validationMse);
        size_t trained = 0;
        const bool earlyStop = p.getBool("early_stop");
        double bestValidation = validationMse;
        size_t bestAt = 0;
        while (trained < train && !(earlyStop && validationMse <= target) && std::isfinite(validationMse)) {
            const size_t block = std::min(evalEvery, train - trained);
            trainSamples(block);
            trained += block;
            validationMse = mseOf(validation, nullptr, nullptr, 0);
            // The learning curve: validation MSE after every block.
            t.record("curve_" + std::to_string(trained), std::isfinite(validationMse) ? validationMse : 1e30);
            if (validationMse < bestValidation) {
                bestValidation = validationMse;
                bestAt = trained;
            }
        }
        t.record("best_validation_mse", std::isfinite(bestValidation) ? bestValidation : 1e30);
        t.record("best_validation_at", static_cast<double>(bestAt));

        std::ofstream traceFile;
        const std::string tracePath = p.getString("trace");
        if (!tracePath.empty()) {
            traceFile.open(tracePath, std::ios::app);
            if (traceFile.tellp() == 0)
                traceFile << "model,depth,width,seed,sample,tick,layer,neuron,output,threshold\n";
        }
        Activity meter;
        const auto start = std::chrono::steady_clock::now();
        const double testMse = mseOf(test, &meter, traceFile.is_open() ? &traceFile : nullptr,
                                     static_cast<size_t>(p.getInt("trace_samples")));
        const std::chrono::duration<double, std::micro> elapsed = std::chrono::steady_clock::now() - start;

        const double samples = static_cast<double>(test.y.size());
        t.record("test_mse", std::isfinite(testMse) ? testMse : 1e30);
        t.record("test_nmse", std::isfinite(testMse) && test.variance > 0 ? testMse / test.variance : 1e30);
        t.record("validation_mse", std::isfinite(validationMse) ? validationMse : 1e30);
        t.record("success", std::isfinite(testMse) && testMse <= target ? 1.0 : 0.0);
        t.record("training_samples", static_cast<double>(trained));
        t.record("target_variance", test.variance);
        t.record("task_complexity", task.complexity());
        t.record("inputs", static_cast<double>(task.inputs));
        t.record("hidden_neurons", static_cast<double>(net.hiddenNeurons()));
        t.record("neurons", static_cast<double>(net.hiddenNeurons() + 1));
        t.record("parameters", static_cast<double>(net.parameters()));
        t.record("hold_ticks", static_cast<double>(hold));

        const double hidden = static_cast<double>(net.hiddenNeurons());
        t.record("active_neurons_mean", meter.activePerTick());
        t.record("active_fraction", meter.activePerTick() / hidden);
        t.record("spikes_per_sample", meter.active / samples);
        t.record("unique_neurons_per_sample", meter.uniquePerSample());
        t.record("fraction_of_neurons_used", meter.uniquePerSample() / hidden);
        t.record("never_active", static_cast<double>(meter.neverActive()));
        t.record("dense_synops_per_sample", static_cast<double>(net.denseSynops() * hold));
        t.record("event_synops_per_sample", meter.eventSynops / samples);
        t.record("inference_us", elapsed.count() / samples);

        // State test: the same stimulus after different histories. Per
        // stimulus, the outputs' mean m and variance v over the histories;
        // the error splits into bias^2 = (m - y)^2 and the state's share v.
        const size_t probes = static_cast<size_t>(p.getInt("state_probes"));
        if (probes > 0) {
            const size_t histories = static_cast<size_t>(p.getInt("state_histories"));
            const size_t historyLen = static_cast<size_t>(p.getInt("state_history_len"));
            const Dataset stimuli(task, probes, dataSeed + 3);
            std::mt19937 h(dataSeed + 4);
            double variance = 0, bias2 = 0, spread = 0, spikeVar = 0;
            for (size_t i = 0; i < probes; ++i) {
                std::vector<double> ys, spikes;
                for (size_t r = 0; r < histories; ++r) {
                    for (size_t j = 0; j < historyLen; ++j) {
                        for (float& v : x)
                            v = u(h);
                        net.present(x, hold);
                    }
                    Activity a;
                    ys.push_back(net.present(stimuli.x[i], hold, &a));
                    spikes.push_back(a.active);
                }
                double m = 0, ms = 0;
                for (size_t r = 0; r < histories; ++r) {
                    m += ys[r];
                    ms += spikes[r];
                }
                m /= static_cast<double>(histories);
                ms /= static_cast<double>(histories);
                double v = 0, vs = 0, lo = ys[0], hi = ys[0];
                for (size_t r = 0; r < histories; ++r) {
                    v += (ys[r] - m) * (ys[r] - m);
                    vs += (spikes[r] - ms) * (spikes[r] - ms);
                    lo = std::min(lo, ys[r]);
                    hi = std::max(hi, ys[r]);
                }
                variance += v / static_cast<double>(histories);
                spikeVar += vs / static_cast<double>(histories);
                bias2 += (m - stimuli.y[i]) * (m - stimuli.y[i]);
                spread += hi - lo;
            }
            const double np = static_cast<double>(probes);
            t.record("state_variance", variance / np);
            t.record("state_bias2", bias2 / np);
            t.record("state_share", variance + bias2 > 0 ? variance / (variance + bias2) : 0.0);
            t.record("state_output_range", spread / np);
            t.record("state_spike_sd", std::sqrt(spikeVar / np));
        }
        if (!meter.thresholds.empty()) {
            double mean = 0.0;
            for (float v : meter.thresholds)
                mean += v;
            t.record("threshold_mean", mean / static_cast<double>(meter.thresholds.size()));
            t.record("threshold_p10", quantile(meter.thresholds, 0.1));
            t.record("threshold_p50", quantile(meter.thresholds, 0.5));
            t.record("threshold_p90", quantile(meter.thresholds, 0.9));
        }

        // The neuron and learning constants in force (the same library build
        // for every model; E-R's apply to model=er only).
        t.record("er_alpha", default_alpha);
        t.record("er_resting_threshold", baseline_threshold * restingFactor);
        t.record("er_threshold_floor_after_firing", 2.0 * baseline_threshold * restingFactor);
        t.record("er_recovery", recovery_factor);
        t.record("er_spontaneous_below_threshold", min_threshold);
        t.record("er_spontaneous_amplitude", p.getDouble("spontaneous_amplitude"));
        t.record("habituation", habituationRule ? 1.0 : 0.0);
        t.record("normalize", p.getBool("normalize") ? 1.0 : 0.0);
        t.record("readout_normalize", p.getBool("readout_normalize") ? 1.0 : 0.0);
        t.record("learning_gain", default_learning_gain);
        t.record("max_output", max_output);
        t.record("max_weight", max_weight);
        for (size_t c = 0; c < task.k && task.level == "l4"; ++c) {
            for (size_t j = 0; j < task.inputs; ++j)
                t.record("task_a" + std::to_string(c) + "_" + std::to_string(j), task.a[c * task.inputs + j]);
            t.record("task_b" + std::to_string(c), task.b[c]);
        }
    },
});

} // namespace
