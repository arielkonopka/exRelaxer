#include "results.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include "nnt_build_info.hpp"
#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

namespace nnt {

Summary summarize(const std::vector<double>& values)
{
    Summary s;
    s.n = values.size();
    if (values.empty())
        return s;
    double sum = 0;
    for (double v : values) sum += v;
    s.mean = sum / static_cast<double>(s.n);
    const auto [lo, hi] = std::ranges::minmax(values);
    s.min = lo;
    s.max = hi;
    if (s.n > 1) {
        double squares = 0;
        for (double v : values) squares += (v - s.mean) * (v - s.mean);
        s.sd = std::sqrt(squares / static_cast<double>(s.n - 1));
        s.stderr_mean = s.sd / std::sqrt(static_cast<double>(s.n));
    }
    return s;
}

namespace {

std::string cpuModel()
{
    std::ifstream info("/proc/cpuinfo");
    for (std::string line; std::getline(info, line);)
        if (line.rfind("model name", 0) == 0) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos)
                return line.substr(line.find_first_not_of(' ', colon + 1));
        }
    return "unknown";
}

std::string hostName()
{
#if __has_include(<unistd.h>)
    char name[256] = {};
    if (gethostname(name, sizeof(name) - 1) == 0)
        return name;
#endif
    return "unknown";
}

std::string utcNow(const char* format)
{
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream text;
    text << std::put_time(&utc, format);
    return text.str();
}

std::string compilerName()
{
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

} // namespace

Environment captureEnvironment(int threads)
{
    Environment env;
    std::random_device device;
    std::ostringstream id;
    id << utcNow("%Y%m%dT%H%M%SZ") << '-' << std::hex << (device() & 0xffffu);
    env.runId = id.str();
    env.time = utcNow("%Y-%m-%dT%H:%M:%SZ");
    env.host = hostName();
    env.cpu = cpuModel();
    env.compiler = compilerName();
    env.buildType = NNT_BUILD_TYPE;
    env.gitCommit = NNT_GIT_COMMIT;
    env.gitDirty = NNT_GIT_DIRTY;
    env.native = NNT_NATIVE;
#ifdef _OPENMP
    env.openmp = true;
#endif
    env.threads = threads;
    return env;
}

std::string jsonString(std::string_view text)
{
    std::string out = "\"";
    for (char c : text) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
                out += buffer;
            } else {
                out += c;
            }
        }
    }
    return out + "\"";
}

std::string jsonNumber(double value)
{
    if (!std::isfinite(value))
        return "null";
    std::ostringstream text;
    text << std::setprecision(10) << value;
    return text.str();
}

} // namespace nnt
