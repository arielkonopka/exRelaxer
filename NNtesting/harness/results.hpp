// Statistics over trials, the environment a run happened in, and the result
// file format (JSON Lines: one object per line).
#pragma once
#include <cstddef>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace nnt {

struct Summary
{
    size_t n = 0;
    double mean = 0, sd = 0, stderr_mean = 0, min = 0, max = 0;
};
Summary summarize(const std::vector<double>& values);

// Where results came from: needed to compare machines, builds and commits.
struct Environment
{
    std::string runId;      // unique per runner invocation
    std::string time;       // UTC, ISO 8601
    std::string host;
    std::string cpu;
    std::string compiler;
    std::string buildType;  // CMake build type
    std::string gitCommit;  // at build time; "unknown" outside a git checkout
    bool gitDirty = false;  // uncommitted changes at build time
    bool native = false;    // built with EXRELAXER_NATIVE (-march=native)
    bool openmp = false;
    int threads = 1;        // OpenMP threads available to the run
};
Environment captureEnvironment(int threads);

std::string jsonString(std::string_view text);  // quoted and escaped
std::string jsonNumber(double value);           // null for NaN / infinity

} // namespace nnt
