// Does a network keep itself active when its inputs are zeroed?
//
// The three-path network of tasks/activity.hpp learns the prototype task,
// then its inputs are set to zero for `silence` ticks. Nothing drives it
// except its own state: E-R thresholds relaxing towards spontaneous firing
// (+-0.01 once a threshold falls below 1e-10), and, with recurrent=true,
// each path's connections to itself (random, frozen scale `gain` /
// sqrt(path size), learned with the rest). Models: er, gate (matched to
// E-R's activity), linear. No activity term anywhere.
//
// Metrics per model (prefix er_, gate_, linear_):
//   silence_active_w1..w4   active fraction of hidden neurons in ticks 1-10,
//                           11-100, 101-1000 and 1001-silence
//   time_to_quiet           first tick with no active hidden neuron (silence + 1 if never)
//   first_spontaneous       first tick with a spontaneous E-R firing (silence + 1 if none)
//   spontaneous_per_1000    spontaneous firings per 1000 ticks, whole network
//   spontaneous_period      mean ticks between bursts of spontaneous firing (0: fewer than two)
//   self_active_ticks       fraction of ticks after tick 100 with any active hidden neuron
//   reactivated             fraction of hidden neurons active at least once after tick 100
//   output_mass_w1..w4      mean sum of |readout outputs| per tick in the same windows
//   output_switches         changes of the readouts' leading class per tick after tick 100
//   end_threshold_mean/max  hidden thresholds when the silence ends
//   end_refractory          fraction of hidden neurons with a threshold above 1 then
//   after_accuracy          accuracy on the first `probe` samples after the silence
//   rest_accuracy           ... on the same samples straight after training (no silence)
//   after_spikes_ratio      spikes on the first sample after silence / without silence
// The trace file (optional) has the tick-by-tick time course.
#include <fstream>
#include <random>
#include <string>
#include "activity.hpp"
#include "experiment.hpp"

namespace {

using namespace activity;

void addRecurrence(PathNet& net, float gain)
{
    for (size_t q = 0; q < net.pathCount(); ++q) {
        const auto id = net.net().findLayer(net.pathName(q));
        net.net().connect(id, id);
        auto& d = net.net().layerAs<dense>(id);
        const size_t n = d.size();
        const float scale = gain / std::sqrt(static_cast<float>(n));
        for (size_t i = 0; i < n; ++i) {
            std::vector<float> w = d.weights(i);
            for (size_t j = w.size() - n; j < w.size(); ++j)  // the path's own outputs come last in the pool
                w[j] *= scale;
            d.setWeights(i, w);
        }
    }
    net.refreshWeights();
}

struct Probe
{
    double accuracy = 0, firstSpikes = 0;
};

Probe probeTask(PathNet& net, const Prototypes& task, std::uint32_t seed, int samples, size_t hold)
{
    Probe r;
    std::mt19937 g(seed);
    std::vector<float> x;
    Meter meter(net);
    for (int i = 0; i < samples; ++i) {
        meter.reset();
        const size_t k = task.draw(g, x);
        const Response resp = present(net, x, k, hold, 0.0f, &meter);
        r.accuracy += resp.decision == k;
        if (i == 0)
            r.firstSpikes = meter.summary().activePerTick * static_cast<double>(hold);
    }
    r.accuracy /= std::max(1, samples);
    return r;
}

nnt::Register experiment({
    .name = "er_silence",
    .description = "zero the inputs of a trained network: does activity continue on its own (spontaneous E-R "
                   "firing, recurrence), and how does it answer afterwards?",
    .tags = {"er", "activity", "state"},
    .params = commonParams({
        {"models", "er,gate,linear", "models to compare (gate=match needs er first)"},
        {"silence", "3000", "ticks with zero input"},
        {"recurrent", "false", "each path also reads its own outputs"},
        {"gain", "1.0", "recurrent weight scale (times 1/sqrt(path size))"},
        {"probe", "5", "task samples after the silence"},
        {"trace", "", "CSV file for the tick-by-tick time course (optional)"},
    }),
    .trials = 10,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Prototypes task(static_cast<size_t>(p.getInt("inputs")), static_cast<size_t>(p.getInt("classes")),
                              p.getDouble("noise"), 77 + t.seed());
        const size_t hold = static_cast<size_t>(p.getInt("hold"));
        const int train = static_cast<int>(p.getInt("train")), silence = static_cast<int>(p.getInt("silence"));
        const int samples = static_cast<int>(p.getInt("probe"));
        const float lr = static_cast<float>(p.getDouble("lr")), gain = static_cast<float>(p.getDouble("gain"));
        const bool recurrent = p.getBool("recurrent");
        std::ofstream trace;
        if (!p.getString("trace").empty()) {
            trace.open(p.getString("trace"), std::ios::app);
            if (trace.tellp() == 0)
                trace << "trial,model,tick,active_fraction,spontaneous,output_mass,leader\n";
        }
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
            if (recurrent)
                addRecurrence(net, gain);
            std::mt19937 train_samples(2000 + t.seed());
            stream(net, task, train_samples, train, hold, lr, nullptr);
            Meter meter(net);
            std::mt19937 check(1000 + t.seed());
            stream(net, task, check, 100, hold, 0.0f, &meter);
            if (model == "er")
                er_active = meter.summary().activeFraction;
            const std::string trained = net.snapshot();
            const Probe rest = probeTask(net, task, 4000 + t.seed(), samples, hold);
            net.restore(trained);

            const size_t n = net.hidden();
            const int edges[5] = {0, 10, 100, 1000, silence};
            double activeSum[4] = {}, mass[4] = {};
            int quiet = silence + 1, firstSpont = silence + 1, selfTicks = 0, switches = 0, late = 0;
            double spont = 0;
            std::vector<std::uint8_t> reactivated(n, 0);
            int leader = -1, lastSpontTick = -1, bursts = 0;
            double burstGaps = 0;
            for (int tick = 1; tick <= silence; ++tick) {
                const std::vector<float> y = net.tick({});
                size_t on = 0, sp = 0;
                for (size_t i = 0; i < n; ++i) {
                    const float h = net.hiddenOutput(i);
                    if (model == "er" && spontaneous(h))  // without E-R a small output is decaying activity
                        ++sp;
                    if (active(h)) {
                        ++on;
                        if (tick > 100)
                            reactivated[i] = 1;
                    }
                }
                double m = 0;
                int lead = 0;
                for (size_t k = 0; k < y.size(); ++k) {
                    m += std::abs(y[k]);
                    if (y[k] > y[static_cast<size_t>(lead)])
                        lead = static_cast<int>(k);
                }
                if (m == 0)
                    lead = -1;
                spont += static_cast<double>(sp);
                if (sp > 0) {
                    // A burst: spontaneous firing after at least one tick without.
                    if (lastSpontTick >= 0 && tick - lastSpontTick > 1) {
                        burstGaps += tick - lastSpontTick;
                        ++bursts;
                    }
                    lastSpontTick = tick;
                }
                if (sp > 0 && firstSpont > silence)
                    firstSpont = tick;
                if (on == 0 && quiet > silence)
                    quiet = tick;
                for (int w = 0; w < 4; ++w)
                    if (tick > edges[w] && tick <= edges[w + 1]) {
                        activeSum[w] += static_cast<double>(on) / static_cast<double>(n);
                        mass[w] += m;
                    }
                if (tick > 100) {
                    ++late;
                    selfTicks += on > 0;
                    switches += lead != leader;
                }
                leader = lead;
                if (trace.is_open())
                    trace << t.index() << ',' << model << ',' << tick << ',' << static_cast<double>(on) / static_cast<double>(n)
                          << ',' << sp << ',' << m << ',' << lead << '\n';
            }
            // Thresholds as the input returns (E-R: raised by spontaneous firings).
            double thresholdSum = 0, refractory = 0, thresholdMax = 0;
            for (size_t i = 0; i < n; ++i) {
                const double th = net.threshold(i);
                thresholdSum += th;
                thresholdMax = std::max(thresholdMax, th);
                refractory += th > 1.0;
            }
            const Probe after = probeTask(net, task, 4000 + t.seed(), samples, hold);

            const std::string pre = model + "_";
            for (int w = 0; w < 4; ++w) {
                const double span = std::max(1, std::min(edges[w + 1], silence) - edges[w]);
                t.record(pre + "silence_active_w" + std::to_string(w + 1), activeSum[w] / span);
                t.record(pre + "output_mass_w" + std::to_string(w + 1), mass[w] / span);
            }
            t.record(pre + "time_to_quiet", quiet);
            t.record(pre + "first_spontaneous", firstSpont);
            t.record(pre + "spontaneous_per_1000", 1000.0 * spont / std::max(1, silence));
            t.record(pre + "spontaneous_period", bursts ? burstGaps / bursts : 0.0);
            t.record(pre + "self_active_ticks", late ? static_cast<double>(selfTicks) / late : 0.0);
            t.record(pre + "reactivated",
                     static_cast<double>(std::count(reactivated.begin(), reactivated.end(), 1)) / static_cast<double>(n));
            t.record(pre + "output_switches", late ? static_cast<double>(switches) / late : 0.0);
            t.record(pre + "end_threshold_mean", thresholdSum / static_cast<double>(n));
            t.record(pre + "end_threshold_max", thresholdMax);
            t.record(pre + "end_refractory", refractory / static_cast<double>(n));
            t.record(pre + "rest_accuracy", rest.accuracy);
            t.record(pre + "after_accuracy", after.accuracy);
            t.record(pre + "after_spikes_ratio", rest.firstSpikes > 0 ? after.firstSpikes / rest.firstSpikes : 0.0);
        }
    },
});

} // namespace
