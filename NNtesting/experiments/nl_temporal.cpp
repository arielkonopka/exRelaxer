// Dynamic nonlinearity substitution, temporal tasks: when the target
// depends on earlier inputs, can E-R's internal state stand in for memory
// the network does not otherwise have? See doc/nonlinearity.md.
//
// One trial is one model, one architecture (depth x width) and one seed,
// as in nl_static. The input is a stream: each step x(t) is held for
// depth + 1 + settle ticks and the output is read on the last one, so the
// feed-forward pipeline has flushed the previous step. A network whose
// neurons have no state (relu, gate, clamp) therefore sees x(t) only, plus
// x(t-1) .. x(t-window) when window > 0. E-R keeps its thresholds across
// steps; er_memoryless resets them before every step (the ablation
// control). While training, the error of every step is applied once
// (feedback alignment, learned bias, rate lr). No term counts activity.
//
// Evaluation runs a fresh stream continuously (validation and test each
// from their own seed); the first `warmup` steps are not scored. Success is
// fixed in advance: accuracy >= target_accuracy for binary tasks (t1-t3),
// MSE <= target_mse for t4.
//
// Metrics: test_accuracy (binary), test_mse, test_nmse, success,
// input_ceiling_accuracy (binary: the best accuracy any function of the
// inputs the network is shown can reach on the test stream; for models
// without state this bounds what they can do), chance_accuracy (always
// the majority answer); the learning curve (curve_<steps>, validation
// accuracy for binary tasks, MSE for t4); size, activity and cost as in
// nl_static; E-R thresholds; the neuron constants.
#include <chrono>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include "experiment.hpp"
#include "nonlinearity.hpp"

namespace {

using namespace nonlinearity;

struct Score
{
    double mse = 0.0, correct = 0.0, variance = 0.0;
    size_t scored = 0;
    std::map<std::vector<float>, std::pair<size_t, size_t>> patterns;  // input -> (count, ones)
    size_t ones = 0;

    double accuracy() const { return scored ? correct / static_cast<double>(scored) : 0.0; }
    double meanSquared() const { return scored ? mse / static_cast<double>(scored) : 0.0; }
    // The best any function of the shown inputs can do: majority per pattern.
    double ceiling() const
    {
        size_t best = 0;
        for (const auto& [x, c] : patterns)
            best += std::max(c.second, c.first - c.second);
        return scored ? static_cast<double>(best) / static_cast<double>(scored) : 0.0;
    }
    double chance() const
    {
        return scored ? static_cast<double>(std::max(ones, scored - ones)) / static_cast<double>(scored) : 0.0;
    }
};

nnt::Register experiment({
    .name = "nl_temporal",
    .description = "temporal tasks (delayed XOR, delayed AND-NOT, parity of n, sin(x(t)*x(t-2))) with relu, E-R, "
                   "memoryless E-R, static-threshold or clamped hidden neurons: accuracy, size, activity, cost",
    .tags = {"er", "nonlinearity", "temporal", "learning"},
    .params = {
        {"task", "t1", "t1 x(t) XOR x(t-1), t2 x(t) AND NOT x(t-3), t3 parity of the last n, t4 sin(x(t)*x(t-2))"},
        {"n", "2", "t3: parity length"},
        {"window", "0", "extra past inputs x(t-1)..x(t-window) given to every model (0: x(t) only)"},
        {"data_seed", "1", "seed of the training stream (validation +1, test +2)"},
        {"model", "er", "hidden neurons: relu, er, er_memoryless (state reset every step), gate, clamp (alias linear)"},
        {"gate", "0.2", "gate model: the fixed threshold (default: E-R's resting threshold)"},
        {"growth", "linear", "E-R threshold growth on firing: linear (default), log (original), fixed, multiplicative"},
        {"growth_amount", "0.5", "E-R threshold growth amount (linear, fixed, multiplicative)"},
        {"spontaneous_below", "1e-10", "E-R: a silent neuron fires spontaneously once its threshold is at or below this"},
        {"spontaneous_amplitude", "0.01", "E-R: spontaneous output drawn uniformly in +- this"},
        {"spontaneous_rate", "0", "E-R: extra probability of a spontaneous firing on any silent tick"},
        {"pretrain_model", "", "train first with these hidden neurons (relu, gate, clamp, er), then copy the weights into `model` (empty: no pretraining)"},
        {"pretrain", "0", "samples (steps) of pretraining with pretrain_model"},
        {"habituation", "false", "habituation in the hidden layers (every model)"},
        {"habituation_steps", "100", "habituation, cut mode: repeats before the input is cut"},
        {"habituation_tolerance", "0", "habituation: relative change still counted as a repeat (0: exact)"},
        {"habituation_decay", "0", "habituation: 0 cuts; a value in (0, 1] fades the input by that factor per repeat"},
        {"habituation_fade_after", "2", "habituation with a decay: repeats before fading starts"},
        {"normalize", "false", "hidden layers divide each weighted sum by the length of the neuron's weights"},
        {"learn_ticks", "last", "training: learn from the last tick's error (last) or from every tick at lr/ticks (all)"},
        {"depth", "1", "hidden layers"},
        {"width", "16", "neurons per hidden layer"},
        {"lr", "0.003", "learning rate (the same for every model)"},
        {"settle", "1", "extra ticks per step after the input reaches the output"},
        {"train", "20000", "maximum training steps"},
        {"eval_every", "1000", "training steps between validation checks"},
        {"validation", "500", "validation steps (scored)"},
        {"test", "2000", "test steps (scored)"},
        {"warmup", "20", "unscored steps at the start of every evaluation stream"},
        {"target_accuracy", "0.95", "success for binary tasks: test accuracy at or above this"},
        {"target_mse", "0.001", "success for t4: test MSE at or below this"},
        {"early_stop", "true", "stop training once validation reaches the success target"},
        {"trace", "", "trace mode: CSV of every hidden neuron's output and threshold per tick (optional)"},
        {"trace_steps", "20", "trace mode: test steps to trace"},
    },
    .trials = 10,
    .run = [](nnt::Trial& t) {
        const auto& p = t.params();
        const TemporalTask task(p.getString("task"), static_cast<size_t>(p.getInt("n")),
                                static_cast<size_t>(p.getInt("window")));
        const std::uint32_t dataSeed = static_cast<std::uint32_t>(p.getInt("data_seed"));
        const std::string model = p.getString("model");
        const size_t depth = static_cast<size_t>(p.getInt("depth")), width = static_cast<size_t>(p.getInt("width"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const size_t train = static_cast<size_t>(p.getInt("train"));
        const size_t evalEvery = std::max<size_t>(1, static_cast<size_t>(p.getInt("eval_every")));
        const size_t warmup = std::max(static_cast<size_t>(p.getInt("warmup")), task.lag() + task.window);
        const double targetAccuracy = p.getDouble("target_accuracy"), targetMse = p.getDouble("target_mse");

        std::optional<Habituation> habituationRule;
        if (p.getBool("habituation"))
            habituationRule = Habituation{static_cast<std::uint32_t>(p.getInt("habituation_steps")),
                                          static_cast<float>(p.getDouble("habituation_tolerance")),
                                          static_cast<float>(p.getDouble("habituation_decay")),
                                          static_cast<std::uint32_t>(p.getInt("habituation_fade_after"))};
        const Spontaneous spontaneousSetting = er_options::spontaneous(
            p.getDouble("spontaneous_below"), p.getDouble("spontaneous_amplitude"), p.getDouble("spontaneous_rate"));
        Mlp net(model, task.inputs(), depth, width, static_cast<float>(p.getDouble("gate")),
                er_options::thresholdGrowth(p.getString("growth"), p.getDouble("growth_amount")), spontaneousSetting,
                habituationRule, p.getBool("normalize"));
        const std::string learnTicks = p.getString("learn_ticks");
        if (learnTicks != "last" && learnTicks != "all")
            throw std::invalid_argument("learn_ticks must be last or all");
        const size_t hold = net.hold(static_cast<size_t>(p.getInt("settle")));
        const size_t keep = task.lag() + task.window + 1;

        auto good = [&](const Score& s) {
            return task.binary() ? s.accuracy() >= targetAccuracy : s.meanSquared() <= targetMse;
        };
        auto push = [&](std::deque<float>& h, float v) {
            h.push_back(v);
            if (h.size() > keep)
                h.pop_front();
        };

        // Runs `steps` scored steps of a stream (after `skip` unscored ones)
        // on `cur`: `net`, or while pretraining the pretraining network.
        Mlp* cur = &net;
        std::vector<float> x, hv;
        auto run = [&](std::mt19937& g, size_t skip, size_t steps, bool learn, Activity* meter, std::ostream* trace,
                       size_t traced) {
            Score s;
            std::deque<float> h;
            double mean = 0.0, meanSq = 0.0;
            for (size_t i = 0; i < skip + steps; ++i) {
                push(h, task.draw(g));
                hv.assign(h.begin(), h.end());
                task.present(hv, x);
                const bool scored = i >= skip && h.size() > task.lag();
                std::ostream* tr = scored && i - skip < traced ? trace : nullptr;
                const std::string prefix = tr ? model + ',' + std::to_string(depth) + ',' + std::to_string(width) + ',' +
                                                    std::to_string(t.seed()) + ',' + std::to_string(i - skip)
                                              : std::string();
                const bool known = h.size() > task.lag();
                const float target = known ? task.target(hv) : 0.0f;
                float y;
                if (learn && known && learnTicks == "all") {
                    y = cur->presentLearningEveryTick(x, hold, target, lr);
                } else {
                    y = cur->present(x, hold, scored ? meter : nullptr, tr, prefix);
                    if (learn && known)
                        cur->learn(target, y, lr);
                }
                if (!known)
                    continue;
                if (!scored)
                    continue;
                const double e = static_cast<double>(y) - target;
                s.mse += e * e;
                ++s.scored;
                mean += target;
                meanSq += static_cast<double>(target) * target;
                if (task.binary()) {
                    s.correct += (y > 0.5f) == (target > 0.5f);
                    s.ones += target > 0.5f;
                    auto& c = s.patterns[x];
                    ++c.first;
                    c.second += target > 0.5f;
                }
            }
            if (s.scored) {
                mean /= static_cast<double>(s.scored);
                s.variance = meanSq / static_cast<double>(s.scored) - mean * mean;
            }
            return s;
        };
        auto metricOf = [&](const Score& s) {
            const double v = task.binary() ? s.accuracy() : s.meanSquared();
            return std::isfinite(v) ? v : 1e30;
        };

        const size_t validationSteps = static_cast<size_t>(p.getInt("validation"));
        auto validate = [&] {
            std::mt19937 g(dataSeed + 1);
            return run(g, warmup, validationSteps, false, nullptr, nullptr, 0);
        };
        std::mt19937 trainStream(dataSeed);
        // Optional pretraining with other hidden neurons, on the start of the
        // same training stream; then `net` takes over the weights and biases.
        const std::string pretrainModel = p.getString("pretrain_model");
        const size_t pretrain = pretrainModel.empty() ? 0 : static_cast<size_t>(p.getInt("pretrain"));
        if (pretrain > 0) {
            Mlp pre(pretrainModel, task.inputs(), depth, width, static_cast<float>(p.getDouble("gate")),
                    er_options::thresholdGrowth(p.getString("growth"), p.getDouble("growth_amount")), spontaneousSetting,
                habituationRule, p.getBool("normalize"));
            cur = &pre;
            run(trainStream, 0, pretrain, true, nullptr, nullptr, 0);
            t.record("pretrain_validation", metricOf(validate()));
            net.copyFrom(pre);
            cur = &net;
            t.record("switch_validation", metricOf(validate()));
        }
        t.record("pretrain_steps", static_cast<double>(pretrain));
        Score validation = validate();
        t.record("initial_validation", metricOf(validation));
        size_t trained = 0;
        const bool earlyStop = p.getBool("early_stop");
        double best = metricOf(validation);
        size_t bestAt = 0;
        while (trained < train && !(earlyStop && good(validation)) && std::isfinite(validation.mse)) {
            const size_t block = std::min(evalEvery, train - trained);
            // One continuous training stream; each block resumes where the last stopped.
            run(trainStream, 0, block, true, nullptr, nullptr, 0);
            trained += block;
            validation = validate();
            const double m = metricOf(validation);
            t.record("curve_" + std::to_string(trained), m);
            if (task.binary() ? m > best : m < best) {
                best = m;
                bestAt = trained;
            }
        }
        t.record("best_validation", best);
        t.record("best_validation_at", static_cast<double>(bestAt));

        std::ofstream traceFile;
        const std::string tracePath = p.getString("trace");
        if (!tracePath.empty()) {
            traceFile.open(tracePath, std::ios::app);
            if (traceFile.tellp() == 0)
                traceFile << "model,depth,width,seed,step,tick,layer,neuron,output,threshold\n";
        }
        Activity meter;
        std::mt19937 testStream(dataSeed + 2);
        const auto start = std::chrono::steady_clock::now();
        const Score test = run(testStream, warmup, static_cast<size_t>(p.getInt("test")), false, &meter,
                               traceFile.is_open() ? &traceFile : nullptr, static_cast<size_t>(p.getInt("trace_steps")));
        const std::chrono::duration<double, std::micro> elapsed = std::chrono::steady_clock::now() - start;

        const double steps = static_cast<double>(std::max<size_t>(test.scored, 1));
        const double mse = test.meanSquared();
        t.record("test_mse", std::isfinite(mse) ? mse : 1e30);
        t.record("test_nmse", std::isfinite(mse) && test.variance > 0 ? mse / test.variance : 1e30);
        if (task.binary()) {
            t.record("test_accuracy", test.accuracy());
            t.record("input_ceiling_accuracy", test.ceiling());
            t.record("chance_accuracy", test.chance());
        }
        t.record("success", std::isfinite(mse) && good(test) ? 1.0 : 0.0);
        t.record("training_steps", static_cast<double>(trained));
        t.record("target_variance", test.variance);
        t.record("task_complexity", task.complexity());
        t.record("task_lag", static_cast<double>(task.lag()));
        t.record("inputs", static_cast<double>(task.inputs()));
        t.record("hidden_neurons", static_cast<double>(net.hiddenNeurons()));
        t.record("neurons", static_cast<double>(net.hiddenNeurons() + 1));
        t.record("parameters", static_cast<double>(net.parameters()));
        t.record("hold_ticks", static_cast<double>(hold));

        const double hidden = static_cast<double>(net.hiddenNeurons());
        t.record("active_neurons_mean", meter.activePerTick());
        t.record("active_fraction", meter.activePerTick() / hidden);
        t.record("spikes_per_sample", meter.active / steps);
        t.record("unique_neurons_per_sample", meter.uniquePerSample());
        t.record("fraction_of_neurons_used", meter.uniquePerSample() / hidden);
        t.record("never_active", static_cast<double>(meter.neverActive()));
        t.record("dense_synops_per_sample", static_cast<double>(net.denseSynops() * hold));
        t.record("event_synops_per_sample", meter.eventSynops / steps);
        t.record("inference_us", elapsed.count() / steps);
        if (!meter.thresholds.empty()) {
            double mean = 0.0;
            for (float v : meter.thresholds)
                mean += v;
            t.record("threshold_mean", mean / static_cast<double>(meter.thresholds.size()));
            t.record("threshold_p10", quantile(meter.thresholds, 0.1));
            t.record("threshold_p50", quantile(meter.thresholds, 0.5));
            t.record("threshold_p90", quantile(meter.thresholds, 0.9));
        }

        t.record("er_alpha", default_alpha);
        t.record("er_resting_threshold", baseline_threshold);
        t.record("er_threshold_floor_after_firing", 2.0 * baseline_threshold);
        t.record("er_recovery", recovery_factor);
        t.record("er_spontaneous_below_threshold", min_threshold);
        t.record("er_spontaneous_amplitude", spontaneous_min_amplitude);
        t.record("er_state_reset_every_step", net.memoryless() ? 1.0 : 0.0);
        t.record("habituation", habituationRule ? 1.0 : 0.0);
        t.record("normalize", p.getBool("normalize") ? 1.0 : 0.0);
        t.record("learning_gain", default_learning_gain);
        t.record("max_output", max_output);
        t.record("max_weight", max_weight);
    },
});

} // namespace
