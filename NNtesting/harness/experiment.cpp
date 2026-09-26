#include "experiment.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace nnt {
namespace {

// Function-local so registration from other files' static objects works
// whatever the initialization order.
std::vector<Experiment>& registry()
{
    static std::vector<Experiment> all;
    return all;
}

[[noreturn]] void badValue(const std::string& name, const std::string& value, const char* type)
{
    throw std::invalid_argument("parameter " + name + " = '" + value + "' is not " + type);
}

} // namespace

const std::vector<Experiment>& experiments() { return registry(); }

Register::Register(Experiment experiment)
{
    if (std::ranges::any_of(registry(), [&](const Experiment& e) { return e.name == experiment.name; }))
        throw std::logic_error("nntest: two experiments named '" + experiment.name + "'");
    registry().push_back(std::move(experiment));
}

const std::string& Params::getString(const std::string& name) const
{
    const auto it = values_.find(name);
    if (it == values_.end())
        throw std::invalid_argument("unknown parameter '" + name + "'");
    return it->second;
}

long long Params::getInt(const std::string& name) const
{
    const std::string& text = getString(name);
    size_t used = 0;
    try {
        const long long value = std::stoll(text, &used);
        if (used == text.size())
            return value;
    } catch (const std::exception&) {
    }
    badValue(name, text, "an integer");
}

double Params::getDouble(const std::string& name) const
{
    const std::string& text = getString(name);
    size_t used = 0;
    try {
        const double value = std::stod(text, &used);
        if (used == text.size())
            return value;
    } catch (const std::exception&) {
    }
    badValue(name, text, "a number");
}

bool Params::getBool(const std::string& name) const
{
    std::string text = getString(name);
    std::ranges::transform(text, text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text == "true" || text == "on" || text == "yes" || text == "1")
        return true;
    if (text == "false" || text == "off" || text == "no" || text == "0")
        return false;
    badValue(name, text, "true or false");
}

} // namespace nnt
