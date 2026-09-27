// Experiment 4: does the response to the same input depend on what the
// network did just before?
//
// From the same trained state, each condition runs for `condition_ticks`
// ticks (tasks/activity.hpp): rest, busy (ordinary task samples), all
// (every path stimulated), or one path stimulated (A, B, C). Then the same
// input X is shown for `hold` ticks. This is repeated for `inputs_tested`
// inputs X, each from the same trained state.
//
// A network whose output is f(input) responds identically after every
// condition; one whose output is f(input, state) does not. Per model and
// condition, relative to rest (so rest itself is 0):
//   pattern_change     fraction of neuron-ticks whose active/silent state differs from after rest
//   evidence_change    |evidence - evidence after rest| / |evidence after rest| (summed readout outputs)
//   decision_change    fraction of inputs whose decision differs from after rest
//   spike_change       spikes (active neuron-ticks) relative to after rest (1 = the same)
//   latency            ticks until the decision settles
//   accuracy           of the decisions
//   threshold_X        mean threshold of path X when X appears (E-R's internal state)
//   lead_X             fraction of ticks path X leads (most drive per neuron)
#include <cmath>
#include <random>
#include <string>
#include "activity.hpp"
#include "experiment.hpp"

namespace {

using namespace activity;

struct Probe
{
    std::vector<std::uint8_t> pattern;  // hold x hidden: active or not
    std::vector<float> evidence;
    size_t decision = 0, latency = 0;
    double spikes = 0;
    Summary summary;
    std::vector<double> threshold;  // per path, at the probe's first tick
};

Probe probeOnce(PathNet& net, std::span<const float> x, size_t hold)
{
    Probe r;
    r.threshold.assign(net.pathCount(), 0.0);
    for (size_t i = 0; i < net.hidden(); ++i)
        r.threshold[net.pathOf(i)] += net.threshold(i) / static_cast<double>(net.pathSize(net.pathOf(i)));
    Meter meter(net);
    r.evidence.assign(net.classes(), 0.0f);
    size_t last = SIZE_MAX;
    for (size_t t = 0; t < hold; ++t) {
        const std::vector<float> y = net.tick(x);
        meter.observe();
        for (size_t i = 0; i < net.hidden(); ++i) {
            const bool on = active(net.hiddenOutput(i));
            r.pattern.push_back(on);
            r.spikes += on;
        }
        for (size_t c = 0; c < y.size(); ++c)
            r.evidence[c] += y[c];
        const auto now = static_cast<size_t>(std::ranges::max_element(r.evidence) - r.evidence.begin());
        if (now != last) {
            r.latency = t + 1;
            last = now;
        }
    }
    r.decision = last;
    r.summary = meter.summary();
    return r;
}

nnt::Register experiment({
    .name = "er_history",
    .description = "history dependence: the same input after rest, heavy use or one path's stimulation; how much do "
                   "the activity pattern, output, spikes and latency change?",
    .tags = {"er", "activity", "state"},
    .params = [] {
        auto p = commonParams({
            {"models", "er,gate,linear", "models to compare (gate=match needs er first)"},
            {"inputs_tested", "20", "different inputs X, each probed after every condition"},
        });
        stateParams(p);
        return p;
    }(),
    .trials = 5,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Prototypes task(static_cast<size_t>(p.getInt("inputs")), static_cast<size_t>(p.getInt("classes")),
                              p.getDouble("noise"), 77 + t.seed());
        const size_t hold = static_cast<size_t>(p.getInt("hold"));
        const int train = static_cast<int>(p.getInt("train")), tested = static_cast<int>(p.getInt("inputs_tested"));
        const int ticks = static_cast<int>(p.getInt("condition_ticks"));
        const float amplitude = static_cast<float>(p.getDouble("amplitude")), lr = static_cast<float>(p.getDouble("lr"));
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
            if (model == "er") {
                Meter meter(net);
                std::mt19937 check(1000 + t.seed());
                stream(net, task, check, 100, hold, 0.0f, &meter);
                er_active = meter.summary().activeFraction;
            }
            const std::string trained = net.snapshot();
            const std::vector<std::string> conditions = conditionNames(net);

            struct Totals
            {
                double pattern = 0, evidence = 0, decision = 0, spikes = 0, latency = 0, correct = 0;
                std::vector<double> threshold, lead;
            };
            std::vector<Totals> totals(conditions.size());
            std::mt19937 inputs(5000 + t.seed());
            std::vector<float> x;
            for (int i = 0; i < tested; ++i) {
                const size_t k = task.draw(inputs, x);
                std::vector<Probe> probes;
                for (const std::string& c : conditions) {
                    net.restore(trained);
                    std::mt19937 g(3000 + t.seed() + static_cast<std::uint32_t>(i));
                    condition(net, c, ticks, amplitude, task, g, hold);
                    probes.push_back(probeOnce(net, x, hold));
                }
                const Probe& rest = probes[0];
                double rest_norm = 0;
                for (float e : rest.evidence)
                    rest_norm += static_cast<double>(e) * e;
                rest_norm = std::sqrt(rest_norm);
                for (size_t c = 0; c < conditions.size(); ++c) {
                    const Probe& r = probes[c];
                    Totals& tot = totals[c];
                    double differ = 0, diff = 0;
                    for (size_t j = 0; j < r.pattern.size(); ++j)
                        differ += r.pattern[j] != rest.pattern[j];
                    for (size_t j = 0; j < r.evidence.size(); ++j)
                        diff += static_cast<double>(r.evidence[j] - rest.evidence[j]) * (r.evidence[j] - rest.evidence[j]);
                    tot.pattern += differ / static_cast<double>(r.pattern.size());
                    tot.evidence += rest_norm > 0 ? std::sqrt(diff) / rest_norm : 0.0;
                    tot.decision += r.decision != rest.decision;
                    tot.spikes += rest.spikes > 0 ? r.spikes / rest.spikes : 1.0;
                    tot.latency += static_cast<double>(r.latency);
                    tot.correct += r.decision == k;
                    tot.threshold.resize(net.pathCount(), 0.0);
                    tot.lead.resize(net.pathCount(), 0.0);
                    for (size_t q = 0; q < net.pathCount(); ++q) {
                        tot.threshold[q] += r.threshold[q];
                        tot.lead[q] += r.summary.pathDominant[q];
                    }
                }
            }
            for (size_t c = 0; c < conditions.size(); ++c) {
                const std::string pre = model + "_" + conditions[c] + "_";
                const Totals& tot = totals[c];
                const double n = tested;
                t.record(pre + "pattern_change", tot.pattern / n);
                t.record(pre + "evidence_change", tot.evidence / n);
                t.record(pre + "decision_change", tot.decision / n);
                t.record(pre + "spike_change", tot.spikes / n);
                t.record(pre + "latency", tot.latency / n);
                t.record(pre + "accuracy", tot.correct / n);
                for (size_t q = 0; q < net.pathCount(); ++q) {
                    t.record(pre + "threshold_" + net.pathName(q), tot.threshold[q] / n);
                    t.record(pre + "lead_" + net.pathName(q), tot.lead[q] / n);
                }
            }
        }
    },
});

} // namespace
