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
// Metrics: test/validation MSE, normalized MSE (MSE / target variance),
// success, training samples used; depth, width, neurons, parameters;
// activity on the test set (active hidden neurons per tick, spikes per
// sample, unique neurons per sample, fraction used, never active); cost
// proxies (dense and event-driven synaptic operations per sample, wall time
// per inference); E-R threshold mean and quantiles; the E-R constants and
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
        {"depth", "2", "hidden layers"},
        {"width", "16", "neurons per hidden layer"},
        {"lr", "0.01", "learning rate (the same for every model)"},
        {"settle", "1", "extra ticks per sample after the input reaches the output"},
        {"train", "20000", "maximum training samples"},
        {"eval_every", "1000", "training samples between validation checks"},
        {"validation", "500", "validation samples"},
        {"test", "1000", "test samples"},
        {"target_mse", "0.001", "success: test MSE at or below this; also the early-stopping target"},
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

        Mlp net(model, task.inputs, depth, width, static_cast<float>(p.getDouble("gate")));
        const size_t hold = net.hold(static_cast<size_t>(p.getInt("settle")));
        const Dataset validation(task, static_cast<size_t>(p.getInt("validation")), dataSeed + 1);
        const Dataset test(task, static_cast<size_t>(p.getInt("test")), dataSeed + 2);

        auto mseOf = [&](const Dataset& d, Activity* meter, std::ostream* trace, size_t traced) {
            double sum = 0.0;
            for (size_t i = 0; i < d.y.size(); ++i) {
                std::ostream* tr = i < traced ? trace : nullptr;
                const std::string prefix = tr ? model + ',' + std::to_string(depth) + ',' + std::to_string(width) + ',' +
                                                    std::to_string(t.seed()) + ',' + std::to_string(i)
                                              : std::string();
                const float y = net.present(d.x[i], hold, meter, tr, prefix);
                const double e = static_cast<double>(y) - d.y[i];
                sum += e * e;
            }
            return d.y.empty() ? 0.0 : sum / static_cast<double>(d.y.size());
        };

        // Training stream: the same samples, in the same order, for every model.
        std::mt19937 g(dataSeed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> x(task.inputs);
        double validationMse = mseOf(validation, nullptr, nullptr, 0);
        t.record("initial_validation_mse", validationMse);
        size_t trained = 0;
        while (trained < train && !(validationMse <= target) && std::isfinite(validationMse)) {
            const size_t block = std::min(evalEvery, train - trained);
            for (size_t i = 0; i < block; ++i) {
                for (float& v : x)
                    v = u(g);
                const float y = net.present(x, hold);
                net.learn(task(x), y, lr);
            }
            trained += block;
            validationMse = mseOf(validation, nullptr, nullptr, 0);
        }

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
        t.record("er_resting_threshold", baseline_threshold);
        t.record("er_threshold_floor_after_firing", 2.0 * baseline_threshold);
        t.record("er_recovery", recovery_factor);
        t.record("er_spontaneous_below_threshold", min_threshold);
        t.record("er_spontaneous_amplitude", spontaneous_min_amplitude);
        t.record("habituation", 0.0);
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
