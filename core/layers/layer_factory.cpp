#include "layer_factory.hpp"
#include "conv2d.hpp"
#include "dense.hpp"
#include "locally_connected2d.hpp"
#include "pool2d.hpp"
#include "retina.hpp"
#include <stdexcept>
#include <string>

namespace exr {

layer_factory::layer_factory()
{
    registerType(LayerType::Dense, [](const LayerSpec& spec) -> std::unique_ptr<layer> {
        return std::make_unique<dense>(spec.size, spec.hasHabituation, spec.hasER,
                                       spec.recoveryJitter, spec.learningJitter, spec.alphaJitter);
    });
    registerType(LayerType::Conv2D, [](const LayerSpec& spec) -> std::unique_ptr<layer> {
        return std::make_unique<conv2d>(spec.size, spec.window, spec.hasHabituation, spec.hasER,
                                        spec.recoveryJitter, spec.learningJitter, spec.alphaJitter);
    });
    registerType(LayerType::LocallyConnected2D, [](const LayerSpec& spec) -> std::unique_ptr<layer> {
        return std::make_unique<locally_connected2d>(spec.size, spec.window, spec.hasHabituation, spec.hasER,
                                                     spec.recoveryJitter, spec.learningJitter, spec.alphaJitter);
    });
    registerType(LayerType::Pool2D, [](const LayerSpec& spec) -> std::unique_ptr<layer> {
        return std::make_unique<pool2d>(spec.window, spec.pool);
    });
    registerType(LayerType::Retina, [](const LayerSpec& spec) -> std::unique_ptr<layer> {
        return std::make_unique<retina>(spec.retina, spec.hasHabituation, spec.hasER,
                                        spec.recoveryJitter, spec.learningJitter, spec.alphaJitter);
    });
}

layer_factory& layer_factory::instance()
{
    static layer_factory factory;
    return factory;
}

void layer_factory::registerType(LayerType type, Creator creator)
{
    this->creators[type] = std::move(creator);
}

bool layer_factory::isRegistered(LayerType type) const
{
    return this->creators.count(type) != 0;
}

std::unique_ptr<layer> layer_factory::create(const LayerSpec& spec) const
{
    const auto it = this->creators.find(spec.type);
    if (it == this->creators.end())
        throw std::invalid_argument("layer_factory: no creator registered for layer type " +
                                    std::to_string(static_cast<int>(spec.type)));
    return it->second(spec);
}

} // namespace exr
