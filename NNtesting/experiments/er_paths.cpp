// Experiments 2 and 7: does a network with several alternative paths come
// to prefer the cheaper (smaller) one, and does it settle into a
// low-activity regime over a long run?
//
// The network of tasks/activity.hpp: three paths (12, 20, 35 neurons) from
// the same input to the readouts, each able to solve the task alone. Nothing
// tells the network what a path costs. Each model (er, gate, linear; see
// er_economy) trains for `train` samples, then runs `long` samples without
// learning (and, with online=true, with learning).
//
// A path's use is its share of the readouts' input |w * y|; divided by its
// share of the neurons it is 1 when the path is used in proportion to its
// size ("drive_per_size"). A preference for the small path shows as path A
// above 1 and C below. The leading path each tick is the one with the most
// drive per neuron; switch_rate counts changes of the lead per tick and
// mean_lead_ticks how long a lead lasts.
//
// Metrics per model (prefix er_ etc.): early_ (first training block) and
// late_ (last training block) drive_per_size per path; for the long run:
// accuracy, active fraction, spikes per sample, cumulative spikes, path use,
// lead fractions, switch rate (internal_: lead changes while the input stays
// the same), mean lead duration, activity entropy. Last, one sample is held
// for `dwell` ticks: dwell_ metrics (accuracy per tick, lead changes, active
// fraction in the first and last quarter) show what the network does on its
// own under a constant input.
#include <random>
#include <string>
#include "activity.hpp"
#include "experiment.hpp"

namespace {

using namespace activity;

double drivePerSize(const PathNet& net, const Summary& s, size_t p)
{
    return s.pathDrive[p] * static_cast<double>(net.hidden()) / static_cast<double>(net.pathSize(p));
}

nnt::Register experiment({
    .name = "er_paths",
    .description = "alternative paths of different sizes: does E-R come to prefer the cheaper path, and how does the "
                   "leading path change over a long run?",
    .tags = {"er", "activity", "paths", "learning"},
    .params = commonParams({
        {"models", "er,gate,linear", "models to compare (gate=match needs er first)"},
        {"long", "3000", "samples in the long run after training"},
        {"online", "false", "keep learning during the long run"},
        {"dwell", "400", "then one sample held this many ticks (0: none)"},
        {"block", "250", "samples per trace block"},
        {"trace", "", "CSV file for the time course (optional)"},
    }),
    .trials = 5,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Prototypes task(static_cast<size_t>(p.getInt("inputs")), static_cast<size_t>(p.getInt("classes")),
                              p.getDouble("noise"), 77 + t.seed());
        const size_t hold = static_cast<size_t>(p.getInt("hold"));
        const int train = static_cast<int>(p.getInt("train")), long_run = static_cast<int>(p.getInt("long"));
        const int block = static_cast<int>(p.getInt("block"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const float online = p.getBool("online") ? lr : 0.0f;
        Trace trace(p.getString("trace"), t.index());
        std::stringstream models(p.getString("models"));
        std::string model;
        bool header = false;
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
            if (!header) {
                trace.header(net, "", t.index());
                header = true;
            }
            const std::string pre = model + "_";
            Meter meter(net);

            std::mt19937 samples(2000 + t.seed());
            int blocks = 0;
            for (int done = 0; done < train; done += block, ++blocks) {
                meter.reset();
                const double accuracy = stream(net, task, samples, std::min(block, train - done), hold, lr, &meter);
                const Summary s = meter.summary();
                trace.row(t.index(), model, "train", blocks, static_cast<double>((done + block) * static_cast<int>(hold)),
                          "", accuracy, s);
                for (size_t q = 0; q < net.pathCount(); ++q) {
                    const std::string name = "path" + net.pathName(q) + "_drive_per_size";
                    if (done == 0)
                        t.record(pre + "early_" + name, drivePerSize(net, s, q));
                    t.record(pre + "late_" + name, drivePerSize(net, s, q));
                }
            }

            Meter whole(net);
            int correct_blocks = 0;
            double accuracy_sum = 0;
            for (int done = 0; done < long_run; done += block, ++blocks) {
                meter.reset();
                const int n = std::min(block, long_run - done);
                // The whole run is observed by `whole`; each block by `meter` for the trace.
                double accuracy = 0;
                {
                    std::vector<float> x;
                    int correct = 0;
                    for (int i = 0; i < n; ++i) {
                        const size_t k = task.draw(samples, x);
                        Response r;
                        r.evidence.assign(net.classes(), 0.0f);
                        size_t last = SIZE_MAX;
                        meter.newSample();
                        whole.newSample();
                        for (size_t tick = 0; tick < hold; ++tick) {
                            const std::vector<float> y = net.tick(x);
                            meter.observe();
                            whole.observe();
                            for (size_t c = 0; c < y.size(); ++c)
                                r.evidence[c] += y[c];
                            last = static_cast<size_t>(std::ranges::max_element(r.evidence) - r.evidence.begin());
                            if (online > 0.0f)
                                net.learn(y, k, online);
                        }
                        correct += last == k;
                    }
                    if (online > 0.0f)
                        net.refreshWeights();
                    accuracy = static_cast<double>(correct) / n;
                }
                accuracy_sum += accuracy;
                ++correct_blocks;
                trace.row(t.index(), model, online > 0 ? "long_online" : "long", blocks,
                          static_cast<double>((train + done + n) * static_cast<int>(hold)), "", accuracy, meter.summary());
            }
            const Summary s = whole.summary();
            t.record(pre + "accuracy", accuracy_sum / std::max(1, correct_blocks));
            record(t, pre, net, s);
            t.record(pre + "spikes_per_sample", s.activePerTick * static_cast<double>(hold));
            t.record(pre + "cumulative_spikes", s.activePerTick * s.ticks);
            if (model == "er")
                er_active = s.activeFraction;

            // One sample held for `dwell` ticks: with the input constant, any
            // change of the leading path is the network's own doing.
            const int dwell = static_cast<int>(p.getInt("dwell"));
            if (dwell > 0) {
                std::vector<float> x;
                const size_t k = task.draw(samples, x);
                Meter held(net), first(net), last(net);
                int correct = 0;
                for (int tick = 0; tick < dwell; ++tick) {
                    const std::vector<float> y = net.tick(x);
                    held.observe();
                    if (tick < dwell / 4)
                        first.observe();
                    if (tick >= dwell - dwell / 4)
                        last.observe();
                    correct += static_cast<size_t>(std::ranges::max_element(y) - y.begin()) == k;
                }
                const Summary h = held.summary();
                t.record(pre + "dwell_accuracy", static_cast<double>(correct) / dwell);
                t.record(pre + "dwell_switch_rate", h.switchRate);
                t.record(pre + "dwell_mean_lead_ticks", h.meanDominance);
                t.record(pre + "dwell_active_fraction", h.activeFraction);
                t.record(pre + "dwell_first_active_fraction", first.summary().activeFraction);
                t.record(pre + "dwell_last_active_fraction", last.summary().activeFraction);
                t.record(pre + "dwell_mean_active_run", h.meanRun);
                for (size_t q = 0; q < net.pathCount(); ++q)
                    t.record(pre + "dwell_path" + net.pathName(q) + "_lead", h.pathDominant[q]);
            }
        }
    },
});

} // namespace
