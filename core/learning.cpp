#include "learning.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace exr {

void LearningRule::validate() const
{
    if (static_cast<std::uint8_t>(type) > static_cast<std::uint8_t>(LearningRuleType::Surrogate))
        throw std::invalid_argument("LearningRule: unknown type " + std::to_string(static_cast<int>(type)));
    if (!(decay >= 0.0f) || !std::isfinite(decay))
        throw std::invalid_argument("LearningRule: decay must be >= 0");
    if (!(trace >= 0.0f && trace < 1.0f))
        throw std::invalid_argument("LearningRule: trace must be in [0, 1)");
    if (!(baseline >= 0.0f && baseline <= 1.0f))
        throw std::invalid_argument("LearningRule: baseline must be in [0, 1]");
    if (!(noise > 0.0f) || !std::isfinite(noise))
        throw std::invalid_argument("LearningRule: noise must be > 0");
    if (!(bcmRate > 0.0f && bcmRate <= 1.0f))
        throw std::invalid_argument("LearningRule: bcmRate must be in (0, 1]");
    if (!(width > 0.0f) || !std::isfinite(width))
        throw std::invalid_argument("LearningRule: width must be > 0");
    if (window < 1 || window > 1024)
        throw std::invalid_argument("LearningRule: window must be in [1, 1024]");
}

const char* learningRuleName(LearningRuleType type)
{
    switch (type) {
    case LearningRuleType::Sign: return "sign";
    case LearningRuleType::Trace: return "trace";
    case LearningRuleType::FeedbackAlignment: return "feedback_alignment";
    case LearningRuleType::Perturbation: return "perturbation";
    case LearningRuleType::Oja: return "oja";
    case LearningRuleType::BCM: return "bcm";
    case LearningRuleType::Eligibility: return "eligibility";
    case LearningRuleType::EProp: return "eprop";
    case LearningRuleType::Surrogate: return "surrogate";
    }
    return "unknown";
}

std::string describeLearningRule(const LearningRule& rule)
{
    std::ostringstream os;
    switch (rule.type) {
    case LearningRuleType::Sign: os << "sign"; break;
    case LearningRuleType::Trace: os << "trace"; break;
    case LearningRuleType::FeedbackAlignment: os << "fa"; break;
    case LearningRuleType::Perturbation: os << "perturb(" << rule.noise << ")"; break;
    case LearningRuleType::Oja: os << "oja"; break;
    case LearningRuleType::BCM: os << "bcm(" << rule.bcmRate << ")"; break;
    case LearningRuleType::Eligibility: os << "elig"; break;
    case LearningRuleType::EProp: os << "eprop"; break;
    case LearningRuleType::Surrogate: os << "surrogate(" << rule.window << ")"; break;
    }
    if ((rule.usesTraces() || rule.type == LearningRuleType::Eligibility || rule.type == LearningRuleType::EProp) &&
        rule.trace > 0.0f)
        os << " tr" << rule.trace;
    if (rule.usesDerivatives() && rule.width != 0.5f)
        os << " w" << rule.width;
    if (rule.usesBaseline() && rule.baseline > 0.0f)
        os << " b" << rule.baseline;
    if (rule.unsupervised() && rule.winners > 0)
        os << " k" << rule.winners;
    if (rule.decay > 0.0f)
        os << " d" << rule.decay;
    if (rule.bias)
        os << " +bias";
    return os.str();
}

} // namespace exr
