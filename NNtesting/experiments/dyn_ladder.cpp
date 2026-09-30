// Dynamic ladder: tasks whose answer lies in how the input changes over
// time, from reading motion on a 1-D retina up to a closed-loop catch game
// (see NNtesting/tasks/dynamic.hpp and doc/dynamic.md). Is E-R's state
// useful on changing input, compared with networks that have no state and
// with networks that are simply shown the last frames (a frame window, the
// standard trick for games)?
//
// One trial is one model, one architecture and one seed, with the same
// network, training and measurements as nl_temporal: each frame is held for
// depth + 1 + settle ticks and the output is read on the last one, so a
// model without state sees only the current frame, plus the previous
// `window` frames when window > 0. The error of every step is applied once
// (feedback alignment, learned bias, rate lr). No term counts activity.
//
// Score (higher is better): accuracy for dir and change, R^2 = 1 - NMSE for
// vel, the catch rate for catch. Also: input_ceiling_accuracy (binary
// tasks: the best any function of the shown inputs can do), for catch the
// catch rate of chasing the ball's current column (chase_catch_rate) and
// agreement with the oracle's actions; activity and cost as in nl_temporal.
#include <chrono>
#include <deque>
#include <map>
#include <string>
#include "dynamic.hpp"
#include "experiment.hpp"
#include "nonlinearity.hpp"

namespace {

using namespace nonlinearity;

struct Score
{
    double mse = 0.0, correct = 0.0, mean = 0.0, meanSq = 0.0;
    size_t scored = 0, games = 0, caught = 0, chaseCaught = 0;
    std::map<std::vector<float>, std::pair<size_t, size_t>> patterns;  // input -> (count, ones)

    double accuracy() const { return scored ? correct / static_cast<double>(scored) : 0.0; }
    double meanSquared() const { return scored ? mse / static_cast<double>(scored) : 0.0; }
    double variance() const
    {
        if (!scored)
            return 0.0;
        const double m = mean / static_cast<double>(scored);
        return meanSq / static_cast<double>(scored) - m * m;
    }
    double nmse() const { return variance() > 0 ? meanSquared() / variance() : 1e30; }
    double catchRate() const { return games ? static_cast<double>(caught) / static_cast<double>(games) : 0.0; }
    double ceiling() const
    {
        size_t best = 0;
        for (const auto& [x, c] : patterns)
            best += std::max(c.second, c.first - c.second);
        return scored ? static_cast<double>(best) / static_cast<double>(scored) : 0.0;
    }
};

int actionOf(float y) { return y > 0.33f ? 1 : y < -0.33f ? -1 : 0; }

nnt::Register experiment({
    .name = "dyn_ladder",
    .description = "time-varying input ladder (motion direction, change detection, velocity, closed-loop catch) "
                   "with relu, E-R, memoryless E-R or static-threshold hidden neurons, optional frame window",
    .tags = {"er", "temporal", "dynamic", "control", "learning"},
    .params = {
        {"task", "dir", "dir (motion direction), change (pattern changed), vel (velocity, regression), catch (game)"},
        {"size", "0", "retina pixels (catch: field width and height); 0 = 16, catch 8"},
        {"switch_p", "-1", "per-step probability of a new velocity / pattern; -1 = task default (dir 0.1, change "
                           "0.3, vel 0.05)"},
        {"window", "0", "extra past frames given to every model (0: the current frame only)"},
        {"data_seed", "1", "seed of the training stream (validation +1, test +2)"},
        {"model", "er", "hidden neurons: relu, er, er_memoryless (state reset every step), gate, clamp"},
        {"gate", "0.2", "gate model: the fixed threshold"},
        {"growth", "linear", "E-R threshold growth: linear (default), log, fixed, multiplicative"},
        {"growth_amount", "0.5", "E-R threshold growth amount"},
        {"depth", "1", "hidden layers"},
        {"width", "32", "neurons per hidden layer"},
        {"lr", "0.003", "learning rate"},
        {"state_readout", "false", "the readout also reads the last hidden layer's E-R thresholds through a State "
                                   "layer (doc/model.md)"},
        {"settle", "1", "extra ticks per step after the input reaches the output"},
        {"train", "20000", "maximum training steps"},
        {"eval_every", "2000", "training steps between validation checks"},
        {"validation", "1000", "validation steps (catch: games)"},
        {"test", "2000", "test steps (catch: games)"},
        {"warmup", "20", "unscored steps at the start of every evaluation stream (not catch)"},
    },
    .trials = 5,
    .run = [](nnt::Trial& t) {
        const auto& p = t.params();
        const std::string taskName = p.getString("task");
        const bool isCatch = taskName == "catch";
        size_t size = static_cast<size_t>(p.getInt("size"));
        if (size == 0)
            size = isCatch ? 8 : 16;
        double switchP = p.getDouble("switch_p");
        if (switchP < 0)
            switchP = taskName == "dir" ? 0.1 : taskName == "change" ? 0.3 : 0.05;
        const std::uint32_t dataSeed = static_cast<std::uint32_t>(p.getInt("data_seed"));
        dynamic::World world(taskName, size, switchP, 777);
        const size_t window = static_cast<size_t>(p.getInt("window"));
        const size_t frame = world.frameSize(), inputs = frame * (window + 1);
        const std::string model = p.getString("model");
        const size_t depth = static_cast<size_t>(p.getInt("depth")), width = static_cast<size_t>(p.getInt("width"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        Mlp net(model, inputs, depth, width, static_cast<float>(p.getDouble("gate")),
                er_options::thresholdGrowth(p.getString("growth"), p.getDouble("growth_amount")), {}, std::nullopt,
                false, false, p.getBool("state_readout"));
        const size_t hold = net.hold(static_cast<size_t>(p.getInt("settle")));
        const size_t warmup = isCatch ? 0 : std::max<size_t>(static_cast<size_t>(p.getInt("warmup")), window + 1);

        std::vector<float> f, x(inputs);
        std::deque<std::vector<float>> frames;
        auto input = [&] {
            world.frame(f);
            frames.push_front(f);
            if (frames.size() > window + 1)
                frames.pop_back();
            for (size_t w = 0; w <= window; ++w)
                for (size_t i = 0; i < frame; ++i)
                    x[w * frame + i] = w < frames.size() ? frames[w][i] : 0.0f;
        };

        // Streams: `steps` scored steps after `skip`. Catch: `steps` games
        // (training: until `steps` steps have been played).
        auto run = [&](std::mt19937& g, size_t skip, size_t steps, bool learn, Activity* meter) {
            Score s;
            frames.clear();
            world.reset(g);
            if (!isCatch) {
                for (size_t i = 0; i < skip + steps; ++i) {
                    input();
                    const float target = world.target();
                    const bool scored = i >= skip;
                    const float y = net.present(x, hold, scored ? meter : nullptr);
                    if (learn)
                        net.learn(target, y, lr);
                    if (scored) {
                        const double e = static_cast<double>(y) - target;
                        s.mse += e * e;
                        s.mean += target;
                        s.meanSq += static_cast<double>(target) * target;
                        ++s.scored;
                        if (world.binary()) {
                            s.correct += (y > 0.5f) == (target > 0.5f);
                            auto& c = s.patterns[x];
                            ++c.first;
                            c.second += target > 0.5f;
                        }
                    }
                    world.advance(g);
                }
                return s;
            }
            size_t played = 0;
            while (learn ? played < steps : s.games < steps) {
                world.reset(g);
                frames.clear();
                // The chase policy plays a copy of the same game.
                dynamic::World chase = world;
                while (!chase.over())
                    chase.advance(g, static_cast<int>(chase.chase()));
                while (!world.over()) {
                    input();
                    const float target = world.target();
                    const float y = net.present(x, hold, learn ? nullptr : meter);
                    if (learn)
                        net.learn(target, y, lr);
                    const int a = actionOf(y);
                    s.correct += a == static_cast<int>(target);
                    const double e = static_cast<double>(y) - target;
                    s.mse += e * e;
                    ++s.scored;
                    world.advance(g, a);
                    ++played;
                }
                ++s.games;
                s.caught += world.caught();
                s.chaseCaught += chase.caught();
            }
            return s;
        };
        auto score = [&](const Score& s) {
            const double v = isCatch ? s.catchRate() : world.binary() ? s.accuracy() : 1.0 - s.nmse();
            return std::isfinite(v) ? v : -1e30;
        };

        const size_t validationSteps = static_cast<size_t>(p.getInt("validation"));
        auto validate = [&] {
            std::mt19937 g(dataSeed + 1);
            return score(run(g, warmup, validationSteps, false, nullptr));
        };
        const size_t train = static_cast<size_t>(p.getInt("train"));
        const size_t evalEvery = std::max<size_t>(1, static_cast<size_t>(p.getInt("eval_every")));
        double validation = validate(), best = validation;
        size_t bestAt = 0;
        t.record("initial_validation", validation);
        for (size_t trained = 0; trained < train;) {
            const size_t block = std::min(evalEvery, train - trained);
            // Each block is a fresh stream (or fresh games) from its own seed;
            // neuron state carries on.
            std::mt19937 g(dataSeed * 7919u + static_cast<std::uint32_t>(trained));
            run(g, 0, block, true, nullptr);
            trained += block;
            validation = validate();
            t.record("curve_" + std::to_string(trained), validation);
            if (validation > best) {
                best = validation;
                bestAt = trained;
            }
        }
        t.record("validation_score", validation);
        t.record("best_validation", best);
        t.record("best_validation_at", static_cast<double>(bestAt));

        Activity meter;
        std::mt19937 testStream(dataSeed + 2);
        const auto start = std::chrono::steady_clock::now();
        const Score test = run(testStream, warmup, static_cast<size_t>(p.getInt("test")), false, &meter);
        const std::chrono::duration<double, std::micro> elapsed = std::chrono::steady_clock::now() - start;
        const double steps = static_cast<double>(std::max<size_t>(test.scored, 1));

        t.record("test_score", score(test));
        t.record("test_mse", std::isfinite(test.meanSquared()) ? test.meanSquared() : 1e30);
        if (world.binary()) {
            t.record("test_accuracy", test.accuracy());
            t.record("input_ceiling_accuracy", test.ceiling());
        } else if (!isCatch) {
            t.record("test_nmse", std::isfinite(test.nmse()) ? test.nmse() : 1e30);
        } else {
            t.record("catch_rate", test.catchRate());
            t.record("chase_catch_rate", static_cast<double>(test.chaseCaught) / static_cast<double>(test.games));
            t.record("oracle_agreement", test.accuracy());
        }
        t.record("switch_p", switchP);
        t.record("inputs", static_cast<double>(inputs));
        t.record("hidden_neurons", static_cast<double>(net.hiddenNeurons()));
        t.record("parameters", static_cast<double>(net.parameters()));
        t.record("hold_ticks", static_cast<double>(hold));
        const double hidden = static_cast<double>(net.hiddenNeurons());
        t.record("active_neurons_mean", meter.activePerTick());
        t.record("active_fraction", meter.activePerTick() / hidden);
        t.record("spikes_per_step", meter.active / steps);
        t.record("fraction_of_neurons_used", meter.uniquePerSample() / hidden);
        t.record("never_active", static_cast<double>(meter.neverActive()));
        t.record("dense_synops_per_step", static_cast<double>(net.denseSynops() * hold));
        t.record("event_synops_per_step", meter.eventSynops / steps);
        t.record("inference_us", elapsed.count() / steps);
        if (!meter.thresholds.empty()) {
            double mean = 0.0;
            for (float v : meter.thresholds)
                mean += v;
            t.record("threshold_mean", mean / static_cast<double>(meter.thresholds.size()));
        }
        t.record("er_state_reset_every_step", net.memoryless() ? 1.0 : 0.0);
    },
});

} // namespace
