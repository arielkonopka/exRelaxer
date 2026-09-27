// Experiments 3 and 5: when one path is fatigued, does another take over,
// is the task kept, and does the fatigued path come back as it recovers?
//
// The network of tasks/activity.hpp (paths A, B, C of 12, 20 and 35
// neurons), trained on the task. From the same trained state, each
// condition runs for `condition_ticks` ticks with the task input silent:
//   rest    nothing happens (the reference)
//   A, B, C that path's stimulation sensors get random +-amplitude every tick,
//           so its neurons fire hard (E-R: their thresholds climb)
//   all     every path stimulated (no rested alternative left)
// Then the task resumes without learning for `probe` samples, and every
// sample is measured: accuracy, each path's share of the readouts' drive,
// each path's active fraction and mean threshold.
//
// Metrics per model and condition (e.g. er_A_...):
//   first_accuracy         accuracy over the first `early` probe samples (rest: rest_first_accuracy)
//   late_accuracy          ... over the last half of the probe
//   first_share            the stimulated path's drive share in the first sample,
//                          relative to the same path after rest (1 = no change)
//   others_share           the other paths' drive share, relative to rest (> 1: they took over)
//   first_threshold        the stimulated path's mean threshold in the first sample
//   recovery_ticks         ticks until the stimulated path's share is back to 90 % of rest
//                          (probe length if it never is)
// For `all`, first_share and recovery use the whole network's active
// fraction instead. The trace file has the sample-by-sample time course.
#include <random>
#include <string>
#include "activity.hpp"
#include "experiment.hpp"

namespace {

using namespace activity;

struct Course
{
    std::vector<double> correct, activeFraction;
    std::vector<std::vector<double>> share, threshold, pathActive;  // [sample][path]
};

Course probe(PathNet& net, const Prototypes& task, std::uint32_t seed, int samples, size_t hold)
{
    Course c;
    std::mt19937 g(seed);
    std::vector<float> x;
    Meter meter(net);
    for (int i = 0; i < samples; ++i) {
        meter.reset();
        const size_t k = task.draw(g, x);
        const Response r = present(net, x, k, hold, 0.0f, &meter);
        const Summary s = meter.summary();
        c.correct.push_back(r.decision == k);
        c.activeFraction.push_back(s.activeFraction);
        c.share.push_back(s.pathDrive);
        c.threshold.push_back(s.pathThreshold);
        c.pathActive.push_back(s.pathActiveFraction);
    }
    return c;
}

double mean(const std::vector<double>& v, size_t from, size_t to)
{
    double sum = 0;
    for (size_t i = from; i < to; ++i)
        sum += v[i];
    return to > from ? sum / static_cast<double>(to - from) : 0.0;
}

nnt::Register experiment({
    .name = "er_fatigue",
    .description = "fatigue one path (or all) by stimulating it, then resume the task: does another path take over, is "
                   "accuracy kept, how fast does the path come back?",
    .tags = {"er", "activity", "paths"},
    .params = [] {
        auto p = commonParams({
            {"models", "er,gate,linear", "models to compare (gate=match needs er first)"},
            {"probe", "40", "task samples after the condition"},
            {"early", "3", "samples counted as the first part of the probe"},
            {"trace", "", "CSV file for the sample-by-sample time course (optional)"},
        });
        stateParams(p);
        return p;
    }(),
    .trials = 10,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Prototypes task(static_cast<size_t>(p.getInt("inputs")), static_cast<size_t>(p.getInt("classes")),
                              p.getDouble("noise"), 77 + t.seed());
        const size_t hold = static_cast<size_t>(p.getInt("hold"));
        const int train = static_cast<int>(p.getInt("train")), samples = static_cast<int>(p.getInt("probe"));
        const size_t early = static_cast<size_t>(p.getInt("early"));
        const int ticks = static_cast<int>(p.getInt("condition_ticks"));
        const float amplitude = static_cast<float>(p.getDouble("amplitude")), lr = static_cast<float>(p.getDouble("lr"));
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
            std::mt19937 train_samples(2000 + t.seed());
            Meter meter(net);
            stream(net, task, train_samples, train, hold, lr, nullptr);
            meter.reset();
            std::mt19937 check(1000 + t.seed());
            stream(net, task, check, 100, hold, 0.0f, &meter);
            if (model == "er")
                er_active = meter.summary().activeFraction;
            const std::string trained = net.snapshot();

            std::vector<std::string> conditions = {"rest", "all"};
            for (size_t q = 0; q < net.pathCount(); ++q)
                conditions.push_back(net.pathName(q));
            Course rest;
            for (const std::string& c : conditions) {
                net.restore(trained);
                std::mt19937 g(3000 + t.seed());
                condition(net, c, ticks, amplitude, task, g, hold);
                const Course course = probe(net, task, 4000 + t.seed(), samples, hold);
                for (int i = 0; i < samples; ++i) {
                    Summary s;
                    s.activeFraction = course.activeFraction[static_cast<size_t>(i)];
                    s.pathDrive = course.share[static_cast<size_t>(i)];
                    s.pathActiveFraction = course.pathActive[static_cast<size_t>(i)];
                    s.pathThreshold = course.threshold[static_cast<size_t>(i)];
                    trace.row(t.index(), model, c, i, static_cast<double>((i + 1) * static_cast<int>(hold)), "",
                              course.correct[static_cast<size_t>(i)], s);
                }
                const std::string pre = model + "_" + c + "_";
                const size_t n = course.correct.size();
                t.record(pre + "first_accuracy", mean(course.correct, 0, std::min(early, n)));
                t.record(pre + "late_accuracy", mean(course.correct, n / 2, n));
                t.record(pre + "first_active_fraction", course.activeFraction[0]);
                if (c == "rest") {
                    rest = course;
                    continue;
                }
                // The stimulated path (all: the whole network).
                size_t path = SIZE_MAX;
                for (size_t q = 0; q < net.pathCount(); ++q)
                    if (c == net.pathName(q))
                        path = q;
                auto measure = [&](const Course& k, size_t i) {
                    return path == SIZE_MAX ? k.activeFraction[i] : k.share[i][path];
                };
                const double reference = measure(rest, 0);
                t.record(pre + "first_share", reference > 0 ? measure(course, 0) / reference : 0.0);
                if (path != SIZE_MAX) {
                    const double rest_others = 1.0 - rest.share[0][path], others = 1.0 - course.share[0][path];
                    t.record(pre + "others_share", rest_others > 0 ? others / rest_others : 0.0);
                    t.record(pre + "first_threshold", course.threshold[0][path]);
                    t.record(pre + "rest_threshold", rest.threshold[0][path]);
                }
                double recovery = static_cast<double>(n * hold);
                for (size_t i = 0; i < n; ++i)
                    if (measure(course, i) >= 0.9 * measure(rest, i)) {
                        recovery = static_cast<double>(i * hold);
                        break;
                    }
                t.record(pre + "recovery_ticks", recovery);
            }
        }
    },
});

} // namespace
