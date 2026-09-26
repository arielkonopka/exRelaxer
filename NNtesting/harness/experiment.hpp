// The interface every experiment file uses. An experiment is one task on
// one kind of network, with parameters (architecture, settings) that the
// runner can sweep. Each file in experiments/ registers one or more:
//
//   static nnt::Register barOrientation({
//       .name = "bar_orientation",
//       .description = "Frozen Gabor features + learned readout: vertical vs horizontal bars",
//       .tags = {"vision", "learning"},
//       .params = {{"mix", "64", "frozen random mixing neurons"}},
//       .trials = 10,
//       .expect = {{.metric = "accuracy", .min = 0.95}},
//       .run = [](nnt::Trial& t) {
//           const int mix = t.params().getInt("mix");
//           ...                              // exr::reseed(t.seed()) has already been called
//           t.record("accuracy", accuracy);
//       },
//   });
//
// The runner seeds every trial, runs the trials of each parameter
// combination, summarizes every recorded metric, checks the expectations
// and writes all results (with the machine and build they came from).
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nnt {

// A parameter an experiment understands, with its default value (as text).
struct ParamSpec
{
    std::string name;
    std::string defaultValue;
    std::string help;
};

// The parameter values of one run. Getters throw std::invalid_argument for
// an unknown name or a value that does not parse.
class Params
{
public:
    Params() = default;
    explicit Params(std::map<std::string, std::string> values) : values_(std::move(values)) {}

    const std::string& getString(const std::string& name) const;
    long long getInt(const std::string& name) const;
    double getDouble(const std::string& name) const;
    bool getBool(const std::string& name) const;  // true/false, on/off, yes/no, 1/0

    const std::map<std::string, std::string>& values() const { return values_; }

private:
    std::map<std::string, std::string> values_;
};

// A check on the mean of a metric over the trials, applied when the
// experiment runs with its default parameters.
struct Expect
{
    std::string metric;
    double min = -std::numeric_limits<double>::infinity();
    double max = std::numeric_limits<double>::infinity();
};

// One seeded run of an experiment. The runner calls exr::reseed(seed())
// before run(), so a trial is reproducible on its own.
class Trial
{
public:
    Trial(int index, std::uint32_t seed, const Params& params, std::ostream& log)
        : index_(index), seed_(seed), params_(params), log_(log) {}

    int index() const { return index_; }
    std::uint32_t seed() const { return seed_; }
    const Params& params() const { return params_; }

    // Records one value of a metric (e.g. an accuracy, a time in ms). A metric
    // recorded several times in a trial keeps its last value.
    void record(std::string_view metric, double value) { metrics_.insert_or_assign(std::string(metric), value); }
    const std::map<std::string, double>& metrics() const { return metrics_; }

    // Diagnostics, shown with --verbose.
    std::ostream& log() { return log_; }

    // Milliseconds per call of f(), best of `repeats` runs of `iterations`
    // calls: the least disturbed measurement on a busy machine.
    template <typename F>
    static double bestMs(int repeats, int iterations, F&& f)
    {
        double best = std::numeric_limits<double>::infinity();
        for (int r = 0; r < repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i)
                f();
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
            best = std::min(best, elapsed.count() / iterations);
        }
        return best;
    }

private:
    int index_;
    std::uint32_t seed_;
    const Params& params_;
    std::ostream& log_;
    std::map<std::string, double> metrics_;
};

struct Experiment
{
    std::string name = {};                 // unique; letters, digits, '_'
    std::string description = {};
    std::vector<std::string> tags = {};    // e.g. "vision", "learning", "performance"
    std::vector<ParamSpec> params = {};
    int trials = 10;                       // default number of trials
    std::vector<Expect> expect = {};
    std::function<void(Trial&)> run = {};
};

// Every registered experiment, in registration order.
const std::vector<Experiment>& experiments();

// Registers an experiment; use as a static object in its file. Throws
// std::logic_error for a duplicate name.
struct Register
{
    explicit Register(Experiment experiment);
};

} // namespace nnt
