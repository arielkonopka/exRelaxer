// nntest: runs the experiments in experiments/ (see experiment.hpp).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "experiment.hpp"
#include "random.hpp"
#include "results.hpp"

namespace {

using nnt::Experiment;

constexpr const char* usage = R"(nntest - benchmark harness for exRelaxer networks

usage:
  nntest list [--tag TAG]             experiments (optionally only those with a tag)
  nntest describe NAME                an experiment's parameters, defaults and checks
  nntest run SELECTOR... [options]    run experiments

selectors:
  NAME            one experiment
  PATTERN         '*' matches any text, e.g. '*_throughput'
  tag:TAG         every experiment with the tag
  all             every experiment

run options:
  --trials N            trials per parameter combination (default: the experiment's)
  --seed S              trial i uses seed S + i (default 0)
  --set NAME=V1,V2,...  parameter values; several --set give every combination
  --threads N           OpenMP threads (default: all)
  --out FILE            append every trial and summary to FILE (JSON Lines)
  --verbose             show the experiments' diagnostics

Checks (an experiment's expected results) apply to runs with default
parameters. Exit status: 0 ok, 1 a check failed or a trial threw, 2 usage.
)";

struct Options
{
    std::vector<std::string> selectors;
    std::optional<int> trials;
    std::uint32_t seed = 0;
    std::map<std::string, std::vector<std::string>> sets;  // parameter -> values
    std::optional<int> threads;
    std::string out;
    bool verbose = false;
    std::string tag;
};

[[noreturn]] void usageError(const std::string& message)
{
    std::cerr << "nntest: " << message << "\n\n" << usage;
    std::exit(2);
}

std::vector<std::string> split(const std::string& text, char separator)
{
    std::vector<std::string> parts;
    std::string part;
    std::istringstream in(text);
    while (std::getline(in, part, separator))
        parts.push_back(part);
    return parts;
}

// '*' matches any run of characters.
bool globMatch(std::string_view pattern, std::string_view text)
{
    if (pattern.empty())
        return text.empty();
    if (pattern.front() == '*')
        return globMatch(pattern.substr(1), text) || (!text.empty() && globMatch(pattern, text.substr(1)));
    return !text.empty() && pattern.front() == text.front() && globMatch(pattern.substr(1), text.substr(1));
}

bool hasTag(const Experiment& e, const std::string& tag) { return std::ranges::find(e.tags, tag) != e.tags.end(); }

std::vector<const Experiment*> select(const std::vector<std::string>& selectors)
{
    std::vector<const Experiment*> chosen;
    for (const std::string& selector : selectors) {
        bool any = false;
        for (const Experiment& e : nnt::experiments()) {
            const bool match = selector == "all" ||
                               (selector.rfind("tag:", 0) == 0 ? hasTag(e, selector.substr(4)) : globMatch(selector, e.name));
            if (match && std::ranges::find(chosen, &e) == chosen.end()) {
                chosen.push_back(&e);
                any = true;
            }
        }
        if (!any)
            usageError("no experiment matches '" + selector + "' (see nntest list)");
    }
    return chosen;
}

const Experiment& find(const std::string& name)
{
    for (const Experiment& e : nnt::experiments())
        if (e.name == name)
            return e;
    usageError("no experiment named '" + name + "' (see nntest list)");
}

Options parseRun(int argc, char** argv)
{
    Options o;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                usageError(arg + " needs a value");
            return argv[++i];
        };
        try {
            if (arg == "--trials") {
                o.trials = std::stoi(value());
                if (*o.trials < 1)
                    usageError("--trials must be at least 1");
            } else if (arg == "--seed") {
                o.seed = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (arg == "--threads") {
                o.threads = std::stoi(value());
            } else if (arg == "--out") {
                o.out = value();
            } else if (arg == "--verbose") {
                o.verbose = true;
            } else if (arg == "--set") {
                const std::string setting = value();
                const size_t eq = setting.find('=');
                if (eq == std::string::npos || eq == 0 || eq + 1 == setting.size())
                    usageError("--set needs NAME=VALUE[,VALUE...], got '" + setting + "'");
                o.sets[setting.substr(0, eq)] = split(setting.substr(eq + 1), ',');
            } else if (arg.rfind("--", 0) == 0) {
                usageError("unknown option " + arg);
            } else {
                o.selectors.push_back(arg);
            }
        } catch (const std::logic_error&) {
            usageError("bad value for " + arg);
        }
    }
    if (o.selectors.empty())
        usageError("run needs at least one experiment");
    return o;
}

// Every combination of the --set values for the parameters `e` declares.
std::vector<std::map<std::string, std::string>> grid(const Experiment& e, const Options& o)
{
    std::vector<std::map<std::string, std::string>> configs(1);
    for (const nnt::ParamSpec& p : e.params)
        configs[0][p.name] = p.defaultValue;
    for (const nnt::ParamSpec& p : e.params) {
        const auto set = o.sets.find(p.name);
        if (set == o.sets.end())
            continue;
        std::vector<std::map<std::string, std::string>> expanded;
        for (const auto& config : configs)
            for (const std::string& v : set->second) {
                auto c = config;
                c[p.name] = v;
                expanded.push_back(c);
            }
        configs = std::move(expanded);
    }
    return configs;
}

std::string describeParams(const Experiment& e, const std::map<std::string, std::string>& values)
{
    std::string text;
    for (const nnt::ParamSpec& p : e.params)
        text += (text.empty() ? "" : " ") + p.name + "=" + values.at(p.name);
    return text.empty() ? "(no parameters)" : text;
}

std::string paramsJson(const std::map<std::string, std::string>& values)
{
    std::string text = "{";
    for (const auto& [k, v] : values)
        text += (text.size() > 1 ? "," : "") + nnt::jsonString(k) + ":" + nnt::jsonString(v);
    return text + "}";
}

std::string envJson(const nnt::Environment& env)
{
    return std::string("{\"run\":") + nnt::jsonString(env.runId) + ",\"time\":" + nnt::jsonString(env.time) +
           ",\"host\":" + nnt::jsonString(env.host) + ",\"cpu\":" + nnt::jsonString(env.cpu) +
           ",\"compiler\":" + nnt::jsonString(env.compiler) + ",\"build\":" + nnt::jsonString(env.buildType) +
           ",\"native\":" + (env.native ? "true" : "false") + ",\"openmp\":" + (env.openmp ? "true" : "false") +
           ",\"threads\":" + std::to_string(env.threads) + ",\"git\":" + nnt::jsonString(env.gitCommit) +
           ",\"dirty\":" + (env.gitDirty ? "true" : "false") + "}";
}

int runCommand(const Options& o)
{
#ifdef _OPENMP
    if (o.threads)
        omp_set_num_threads(*o.threads);
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    const std::vector<const Experiment*> chosen = select(o.selectors);
    for (const auto& [name, values] : o.sets)
        if (std::ranges::none_of(chosen, [&](const Experiment* e) {
                return std::ranges::any_of(e->params, [&](const nnt::ParamSpec& p) { return p.name == name; });
            }))
            usageError("no selected experiment has a parameter '" + name + "'");

    const nnt::Environment env = nnt::captureEnvironment(threads);
    std::cout << "run " << env.runId << "  " << env.cpu << ", " << threads << " threads, " << env.buildType
              << (env.native ? " native" : "") << ", git " << env.gitCommit << (env.gitDirty ? " (dirty)" : "")
              << "\n";

    std::ofstream out;
    if (!o.out.empty()) {
        out.open(o.out, std::ios::app);
        if (!out)
            usageError("cannot open " + o.out);
    }
    const std::string env_json = envJson(env);
    std::ostream none(nullptr);
    std::ostream& log = o.verbose ? std::cerr : none;
    int status = 0;

    for (const Experiment* e : chosen) {
        const int trials = o.trials.value_or(e->trials);
        for (const auto& config : grid(*e, o)) {
            const bool defaults = std::ranges::all_of(e->params, [&](const nnt::ParamSpec& p) {
                return config.at(p.name) == p.defaultValue;
            });
            std::cout << "\n" << e->name << "  [" << describeParams(*e, config) << "]  " << trials << " trials\n";
            const nnt::Params params(config);
            std::map<std::string, std::vector<double>> values;  // metric -> one value per successful trial
            int failed = 0;
            double seconds_total = 0;
            for (int i = 0; i < trials; ++i) {
                const std::uint32_t seed = o.seed + static_cast<std::uint32_t>(i);
                exr::reseed(seed);
                nnt::Trial trial(i, seed, params, log);
                std::string error;
                const auto start = std::chrono::steady_clock::now();
                try {
                    e->run(trial);
                } catch (const std::exception& ex) {
                    error = ex.what();
                }
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                seconds_total += seconds;
                if (!error.empty()) {
                    ++failed;
                    std::cout << "  trial " << i << " (seed " << seed << ") failed: " << error << "\n";
                } else {
                    for (const auto& [metric, v] : trial.metrics())
                        values[metric].push_back(v);
                }
                if (out) {
                    out << "{\"type\":\"trial\",\"experiment\":" << nnt::jsonString(e->name) << ",\"params\":"
                        << paramsJson(config) << ",\"trial\":" << i << ",\"seed\":" << seed
                        << ",\"seconds\":" << nnt::jsonNumber(seconds) << ",\"metrics\":{";
                    bool first = true;
                    for (const auto& [metric, v] : trial.metrics()) {
                        out << (first ? "" : ",") << nnt::jsonString(metric) << ":" << nnt::jsonNumber(v);
                        first = false;
                    }
                    out << "},\"error\":" << (error.empty() ? "null" : nnt::jsonString(error))
                        << ",\"env\":" << env_json << "}\n";
                }
            }
            if (failed > 0)
                status = 1;

            // Summary table.
            size_t width = 6;
            for (const auto& [metric, v] : values)
                width = std::max(width, metric.size());
            std::cout << "  " << std::left << std::setw(static_cast<int>(width)) << "metric" << std::right
                      << std::setw(12) << "mean" << std::setw(11) << "stderr" << std::setw(11) << "sd"
                      << std::setw(11) << "min" << std::setw(11) << "max" << "\n";
            std::map<std::string, nnt::Summary> summaries;
            for (const auto& [metric, v] : values) {
                const nnt::Summary s = nnt::summarize(v);
                summaries[metric] = s;
                std::cout << "  " << std::left << std::setw(static_cast<int>(width)) << metric << std::right
                          << std::setprecision(5) << std::setw(12) << s.mean << std::setw(11) << s.stderr_mean
                          << std::setw(11) << s.sd << std::setw(11) << s.min << std::setw(11) << s.max << "\n";
            }
            std::cout << "  " << trials - failed << " of " << trials << " trials ok, " << std::setprecision(3)
                      << seconds_total << " s\n";

            // Checks, for the default configuration.
            std::string checks_json = "[";
            if (defaults)
                for (const nnt::Expect& check : e->expect) {
                    const auto s = summaries.find(check.metric);
                    const bool pass = s != summaries.end() && s->second.mean >= check.min && s->second.mean <= check.max;
                    if (!pass)
                        status = 1;
                    std::cout << "  check " << check.metric;
                    if (std::isfinite(check.min)) std::cout << " >= " << check.min;
                    if (std::isfinite(check.max)) std::cout << " <= " << check.max;
                    std::cout << ": " << (pass ? "pass" : "FAIL") << "\n";
                    checks_json += (checks_json.size() > 1 ? "," : "") + std::string("{\"metric\":") +
                                   nnt::jsonString(check.metric) + ",\"min\":" + nnt::jsonNumber(check.min) +
                                   ",\"max\":" + nnt::jsonNumber(check.max) + ",\"pass\":" + (pass ? "true" : "false") + "}";
                }
            checks_json += "]";

            if (out) {
                out << "{\"type\":\"summary\",\"experiment\":" << nnt::jsonString(e->name) << ",\"params\":"
                    << paramsJson(config) << ",\"defaults\":" << (defaults ? "true" : "false")
                    << ",\"trials\":" << trials << ",\"failed\":" << failed << ",\"seconds\":"
                    << nnt::jsonNumber(seconds_total) << ",\"metrics\":{";
                bool first = true;
                for (const auto& [metric, s] : summaries) {
                    out << (first ? "" : ",") << nnt::jsonString(metric) << ":{\"n\":" << s.n
                        << ",\"mean\":" << nnt::jsonNumber(s.mean) << ",\"sd\":" << nnt::jsonNumber(s.sd)
                        << ",\"stderr\":" << nnt::jsonNumber(s.stderr_mean) << ",\"min\":" << nnt::jsonNumber(s.min)
                        << ",\"max\":" << nnt::jsonNumber(s.max) << "}";
                    first = false;
                }
                out << "},\"checks\":" << checks_json << ",\"env\":" << env_json << "}\n";
            }
        }
    }
    return status;
}

void listCommand(const std::string& tag)
{
    size_t width = 4;
    for (const Experiment& e : nnt::experiments())
        width = std::max(width, e.name.size());
    for (const Experiment& e : nnt::experiments()) {
        if (!tag.empty() && !hasTag(e, tag))
            continue;
        std::string tags;
        for (const std::string& t : e.tags)
            tags += (tags.empty() ? "" : ",") + t;
        std::cout << std::left << std::setw(static_cast<int>(width) + 2) << e.name << e.description << "  [" << tags
                  << "]\n";
    }
}

void describeCommand(const std::string& name)
{
    const Experiment& e = find(name);
    std::cout << e.name << ": " << e.description << "\ndefault trials: " << e.trials << "\nparameters:\n";
    for (const nnt::ParamSpec& p : e.params)
        std::cout << "  " << std::left << std::setw(18) << p.name << std::setw(10) << p.defaultValue << p.help << "\n";
    if (!e.expect.empty()) {
        std::cout << "checks (default parameters):\n";
        for (const nnt::Expect& c : e.expect) {
            std::cout << "  " << c.metric;
            if (std::isfinite(c.min)) std::cout << " >= " << c.min;
            if (std::isfinite(c.max)) std::cout << " <= " << c.max;
            std::cout << "\n";
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || std::string(argv[1]) == "help" || std::string(argv[1]) == "--help") {
        std::cout << usage;
        return argc < 2 ? 2 : 0;
    }
    const std::string command = argv[1];
    if (command == "list") {
        std::string tag;
        if (argc == 4 && std::string(argv[2]) == "--tag")
            tag = argv[3];
        else if (argc != 2)
            usageError("list takes only --tag TAG");
        listCommand(tag);
        return 0;
    }
    if (command == "describe") {
        if (argc != 3)
            usageError("describe takes one experiment name");
        describeCommand(argv[2]);
        return 0;
    }
    if (command == "run")
        return runCommand(parseRun(argc, argv));
    usageError("unknown command '" + command + "'");
}
