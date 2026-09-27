// Is habituation an activity-saving mechanism, and what does it cost?
//
// The three-path network of tasks/activity.hpp learns the prototype task
// with short presentations (hold ticks each, where habituation never
// triggers). It is then tested with every sample held unchanged for longer
// (test_holds). Habituation zeroes a neuron's input once it has been
// identical for habituation_steps (100) ticks, until it changes. Compare
// habituation=false,true for each model (er, gate matched to E-R's
// activity, linear). No activity term anywhere.
//
// Metrics per model and test hold H (prefix er_h<H>_ ...):
//   spikes_per_sample       active hidden neuron-ticks per sample
//   late_active_fraction    active fraction after the first 100 ticks of a sample (H > 100)
//   accuracy_onset          decision from the readouts summed over the first `hold` ticks
//                           (how fast a new stimulus is recognized after the previous one)
//   accuracy_sum            decision from the readouts summed over the whole sample
//   accuracy_end            decision from the readouts on the sample's last tick
//                           (is the stimulus still represented at the end?)
//   accuracy_per_100_spikes accuracy_sum per 100 spikes
#include <random>
#include <string>
#include "activity.hpp"
#include "experiment.hpp"

namespace {

using namespace activity;

nnt::Register experiment({
    .name = "er_habituation",
    .description = "hold each stimulus for long: does habituation save activity, and does the network still "
                   "recognize the stimulus and notice a new one?",
    .tags = {"er", "activity", "habituation"},
    .params = commonParams({
        {"models", "er,gate,linear", "models to compare (gate=match needs er first)"},
        {"test_holds", "4,50,200,500", "ticks each test sample is held"},
        {"test_samples", "60", "test samples per hold"},
        {"hold_noise", "0", "uniform +- noise redrawn every tick of a held sample (a real sensor's flicker)"},
    }),
    .trials = 10,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Prototypes task(static_cast<size_t>(p.getInt("inputs")), static_cast<size_t>(p.getInt("classes")),
                              p.getDouble("noise"), 77 + t.seed());
        const size_t hold = static_cast<size_t>(p.getInt("hold"));
        const int train = static_cast<int>(p.getInt("train")), samples = static_cast<int>(p.getInt("test_samples"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const std::vector<size_t> holds = parseSizes(p.getString("test_holds"));
        const float holdNoise = static_cast<float>(p.getDouble("hold_noise"));
        std::uniform_real_distribution<float> flicker(-holdNoise, holdNoise);
        std::stringstream models(p.getString("models"));
        std::string model;
        double er_active = -1;
        while (std::getline(models, model, ',')) {
            float gate = -1.0f;
            if (model == "gate" && p.getString("gate") == "match") {
                if (er_active < 0)
                    throw std::invalid_argument("gate=match needs the er model first");
                gate = matchingGate(p, task, t.seed(), 1000 + t.seed(), er_active);
            }
            exr::reseed(t.seed());
            PathNet net(p, model, gate);
            std::mt19937 train_samples(2000 + t.seed());
            stream(net, task, train_samples, train, hold, lr, nullptr);
            Meter meter(net);
            std::mt19937 check(1000 + t.seed());
            stream(net, task, check, 100, hold, 0.0f, &meter);
            if (model == "er")
                er_active = meter.summary().activeFraction;
            const std::string trained = net.snapshot();

            const size_t n = net.hidden(), K = net.classes();
            for (size_t H : holds) {
                net.restore(trained);
                std::mt19937 g(5000 + t.seed());
                std::vector<float> x;
                double spikes = 0, lateActive = 0, lateTicks = 0;
                int onset = 0, sum = 0, end = 0;
                for (int s = 0; s < samples; ++s) {
                    const size_t k = task.draw(g, x);
                    std::vector<double> evidence(K, 0.0), early(K, 0.0);
                    std::vector<float> y, xt = x;
                    for (size_t tick = 0; tick < H; ++tick) {
                        if (holdNoise > 0)
                            for (size_t j = 0; j < x.size(); ++j)
                                xt[j] = x[j] + flicker(g);
                        y = net.tick(xt);
                        size_t on = 0;
                        for (size_t i = 0; i < n; ++i)
                            on += active(net.hiddenOutput(i));
                        spikes += static_cast<double>(on);
                        if (tick >= 100) {
                            lateActive += static_cast<double>(on) / static_cast<double>(n);
                            ++lateTicks;
                        }
                        for (size_t c = 0; c < K; ++c) {
                            evidence[c] += y[c];
                            if (tick < hold)
                                early[c] += y[c];
                        }
                    }
                    auto argmax = [](const auto& v) {
                        return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin());
                    };
                    onset += argmax(early) == k;
                    sum += argmax(evidence) == k;
                    // All-zero readouts (the stimulus no longer represented) count as wrong.
                    const bool silent = std::all_of(y.begin(), y.end(), [](float v) { return v == 0.0f; });
                    end += !silent && argmax(y) == k;
                }
                const std::string pre = model + "_h" + std::to_string(H) + "_";
                const double spikesPerSample = spikes / samples;
                t.record(pre + "spikes_per_sample", spikesPerSample);
                if (lateTicks > 0)
                    t.record(pre + "late_active_fraction", lateActive / lateTicks);
                t.record(pre + "accuracy_onset", static_cast<double>(onset) / samples);
                t.record(pre + "accuracy_sum", static_cast<double>(sum) / samples);
                t.record(pre + "accuracy_end", static_cast<double>(end) / samples);
                t.record(pre + "accuracy_per_100_spikes",
                         spikesPerSample > 0 ? 100.0 * sum / samples / spikesPerSample : 0.0);
            }
        }
    },
});

} // namespace
