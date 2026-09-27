// Experiments 1 and 6: does E-R make a network use less activity for the
// same task, without any activity cost?
//
// Three networks with the same architecture (tasks/activity.hpp: three
// alternative paths of 12, 20 and 35 neurons between the input and the
// readouts), the same initial weights, the same learning rule and the same
// samples. Only the hidden neurons differ:
//   er      adaptive E-R threshold
//   gate    a fixed threshold at E-R's resting value (all-or-nothing, nothing adapts)
//   linear  no threshold (the E-R-free baseline)
// Nothing in the learning sees activity: the errors are the task's only.
//
// Each model is measured on the same test samples before and after
// training. Metrics per model (prefix er_, gate_, linear_): accuracy,
// latency (ticks until the decision settles), active neurons per tick,
// spikes per inference (active neuron-ticks per sample), accuracy per 100
// spikes, spontaneous firings, activity mass (sum of |output|), unique and
// never-active neurons, mean length of active periods, activity entropy,
// and per path its active fraction, share of the readouts' drive and mean
// threshold. `before_` metrics are the untrained network's; train_first_ /
// train_last_ are the first and last training blocks.
#include <random>
#include <string>
#include "activity.hpp"
#include "experiment.hpp"

namespace {

using namespace activity;

nnt::Register experiment({
    .name = "er_economy",
    .description = "activity economy: E-R vs fixed-threshold vs linear hidden neurons on the same task, no activity "
                   "cost; accuracy, active neurons, spikes per inference, path use",
    .tags = {"er", "activity", "learning", "quick"},
    .params = commonParams({
        {"models", "er,gate,linear", "models to compare"},
        {"block", "100", "training samples per trace block"},
        {"trace", "", "CSV file for the training time course (optional)"},
    }),
    .trials = 5,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Prototypes task(static_cast<size_t>(p.getInt("inputs")), static_cast<size_t>(p.getInt("classes")),
                              p.getDouble("noise"), 77 + t.seed());
        const size_t hold = static_cast<size_t>(p.getInt("hold"));
        const int train = static_cast<int>(p.getInt("train")), test = static_cast<int>(p.getInt("test"));
        const int block = static_cast<int>(p.getInt("block"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        Trace trace(p.getString("trace"), t.index());
        std::stringstream models(p.getString("models"));
        std::string model;
        bool header = false;
        double er_active = -1;  // the trained E-R network's active fraction, for gate=match
        while (std::getline(models, model, ',')) {
            float gate = -1.0f;
            if (model == "gate" && p.getString("gate") == "match") {
                if (er_active < 0)
                    throw std::invalid_argument("gate=match needs the er model first");
                gate = matchingGate(p, task, t.seed(), 1000 + t.seed(), er_active);
                t.record("gate_threshold", gate);
            }
            exr::reseed(t.seed());  // the same initial weights for every model
            PathNet net(p, model, gate);
            if (!header) {
                trace.header(net, "", t.index());
                header = true;
            }
            const std::string pre = model + "_";
            Meter meter(net);

            // Untrained, on the test samples (then back to the untrained state).
            const std::string untrained = net.snapshot();
            std::mt19937 before_samples(1000 + t.seed());
            t.record(pre + "before_accuracy", stream(net, task, before_samples, test, hold, 0.0f, &meter));
            const Summary before = meter.summary();
            t.record(pre + "before_active_fraction", before.activeFraction);
            t.record(pre + "before_mass_per_tick", before.massPerTick);
            net.restore(untrained);

            std::mt19937 train_samples(2000 + t.seed());
            for (int b = 0; b * block < train; ++b) {
                meter.reset();
                const int n = std::min(block, train - b * block);
                const double accuracy = stream(net, task, train_samples, n, hold, lr, &meter);
                const Summary s = meter.summary();
                trace.row(t.index(), model, "train", b, static_cast<double>((b + 1) * block * hold), "", accuracy, s);
                if (b == 0) {
                    t.record(pre + "train_first_accuracy", accuracy);
                    t.record(pre + "train_first_active_fraction", s.activeFraction);
                }
                t.record(pre + "train_last_accuracy", accuracy);
                t.record(pre + "train_last_active_fraction", s.activeFraction);
            }

            meter.reset();
            double latency = 0;
            std::mt19937 test_samples(1000 + t.seed());
            const double accuracy = stream(net, task, test_samples, test, hold, 0.0f, &meter, &latency);
            const Summary s = meter.summary();
            t.record(pre + "accuracy", accuracy);
            t.record(pre + "latency", latency);
            record(t, pre, net, s);
            const double spikes = s.activePerTick * static_cast<double>(hold);
            t.record(pre + "spikes_per_inference", spikes);
            t.record(pre + "accuracy_per_100_spikes", spikes > 0 ? 100.0 * accuracy / spikes : 0.0);
            if (model == "er")
                er_active = s.activeFraction;
        }
    },
});

} // namespace
