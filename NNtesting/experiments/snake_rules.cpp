// Snake (tasks/snake.hpp) with a choice of learning rule for the readouts
// and for the mixing layer: the benchmark for comparing learning rules.
// Same game, state, network and rewards as the experiment `snake`; with the
// defaults (readout=sign, mix=frozen) it plays exactly the same games.
//
//   state (23 sensors) -> "eye" (pass-through) ----------------------+
//                           \-> dense mix (rule `mix`) ----------------+-> 3 readouts, one per action (rule `readout`)
//
// After each move the chosen action gets the reward r (`eat`, `die`, or
// +-`approach`). When it learns (error-driven: only when the readout's sign
// disagrees with r), each layer learns by its rule:
//
//   readout sign / trace                  applyReward(r) on the chosen readout
//   readout feedback_alignment            its own error r - y on the chosen readout (delta rule)
//   readout perturbation                  applyReward(-(r - y)^2) on the chosen readout
//   mix frozen                            nothing
//   mix sign / trace                      applyReward(r)
//   mix perturbation                      applyReward(-(r - y)^2)
//   mix feedback_alignment                a fixed random projection of the readouts' errors
//                                         (r - y for the chosen action, 0 for the others)
//   mix oja / bcm                         an unsupervised update (the reward is ignored)
//
// Metrics: apples and steps per test game (and while training), deaths,
// best game, and the training cost in microseconds per step.
#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
#include "experiment.hpp"
#include "layers/dense.hpp"
#include "network.hpp"
#include "snake.hpp"

namespace {

using namespace exr;

// The rule named `name`, with the experiment's rule options.
LearningRule ruleFor(const std::string& name, const nnt::Params& p)
{
    LearningRule rule;
    if (name == "sign")
        rule = LearningRule::sign();
    else if (name == "trace")
        rule = LearningRule::traced(static_cast<float>(p.getDouble("trace")),
                                    static_cast<float>(p.getDouble("baseline")));
    else if (name == "fa" || name == "feedback_alignment")
        rule = LearningRule::feedbackAlignment(static_cast<float>(p.getDouble("trace")));
    else if (name == "perturbation")
        rule = LearningRule::perturbation(static_cast<float>(p.getDouble("noise")),
                                          static_cast<float>(p.getDouble("trace")),
                                          static_cast<float>(p.getDouble("p_baseline")));
    else if (name == "oja")
        rule = LearningRule::oja(static_cast<std::uint32_t>(p.getInt("winners")));
    else if (name == "bcm")
        rule = LearningRule::bcm(static_cast<float>(p.getDouble("bcm_rate")),
                                 static_cast<std::uint32_t>(p.getInt("winners")));
    else
        throw std::invalid_argument("unknown learning rule '" + name +
                                    "' (sign, trace, fa, perturbation, oja, bcm; mix also frozen)");
    return rule.withDecay(static_cast<float>(p.getDouble("decay")));
}

struct Player
{
    std::unique_ptr<network> net = std::make_unique<network>();
    std::array<network::LayerId, 3> readouts{};
    network::LayerId mix = 0;
    bool mixLearns = false;
    LearningRule readoutRule, mixRule;

    explicit Player(const nnt::Params& p)
    {
        readoutRule = ruleFor(p.getString("readout"), p);
        if (readoutRule.unsupervised())
            throw std::invalid_argument("readout: an unsupervised rule cannot learn the actions");
        const bool bias = p.getBool("bias");
        if (bias)
            readoutRule = readoutRule.withBias();

        const Shape state{1, 1, snake::State::size};
        const auto eye = net->addLayer("eye", LayerSpec::Retina({state}, false, false));
        net->addInputs(eye, state);
        const size_t mix_size = static_cast<size_t>(p.getInt("mix_size"));
        const std::string mix_name = p.getString("mix");
        network::LayerId features = eye;
        if (mix_size > 0) {
            LayerSpec spec = LayerSpec::Dense(mix_size, false, p.getBool("mix_er"));
            mixLearns = mix_name != "frozen";
            if (mixLearns) {
                mixRule = ruleFor(mix_name, p);
                if (bias)
                    mixRule = mixRule.withBias();
                spec.learningRule = mixRule;
            }
            spec.frozen = !mixLearns;
            features = mix = net->addLayer("mix", spec);
            net->connect(eye, features);
        }
        const char* names[] = {"left", "straight", "right"};
        for (size_t a = 0; a < 3; ++a) {
            LayerSpec spec = LayerSpec::Dense(1, false, false);
            spec.learningRule = readoutRule;
            readouts[a] = net->addLayer(names[a], spec);
            net->connect(eye, readouts[a]);
            if (features != eye)
                net->connect(features, readouts[a]);
            net->addOutput(readouts[a]);
        }
    }

    std::vector<float> y;  // the readouts' outputs at the last choice

    // The action with the largest output; ties go to straight, then left.
    snake::Action choose(const snake::State& s)
    {
        net->setInputs(s.values);
        net->step();
        y = net->outputs();
        size_t best = 1;
        for (size_t a : {0u, 2u})
            if (y[a] > y[best])
                best = a;
        return static_cast<snake::Action>(best);
    }

    neuron_layer& layer(network::LayerId id) { return net->layerAs<neuron_layer>(id); }

    // One learning event: action `a` got reward r.
    void learn(size_t a, float r, float readoutRate, float mixRate)
    {
        const float error = r - y[a];
        const float fitness = -error * error;  // perturbation maximises it
        neuron_layer& readout = layer(readouts[a]);
        if (readoutRule.type == LearningRuleType::FeedbackAlignment)
            readout.applyModulators(std::span<const float>(&error, 1), readoutRate);
        else if (readoutRule.type == LearningRuleType::Perturbation)
            readout.applyReward(fitness, readoutRate);
        else
            readout.applyReward(r, readoutRate);
        if (!mixLearns)
            return;
        neuron_layer& m = layer(mix);
        switch (mixRule.type) {
        case LearningRuleType::FeedbackAlignment: {
            std::array<float, 3> errors{};
            errors[a] = error;
            m.applyFeedback(errors, mixRate);
            break;
        }
        case LearningRuleType::Perturbation:
            m.applyReward(fitness, mixRate);
            break;
        default:
            m.applyReward(r, mixRate);  // Oja / BCM ignore it
        }
    }
};

struct Rewards
{
    float eat, die, approach;
    bool errorDriven;  // learn only when the readout's sign disagrees with the reward
};

struct Totals
{
    double apples = 0, steps = 0, deaths = 0, best = 0;
    int games = 0;
    void add(const snake::Game& g, snake::Outcome last)
    {
        apples += static_cast<double>(g.score());
        steps += static_cast<double>(g.steps());
        deaths += last == snake::Outcome::Died;
        best = std::max(best, static_cast<double>(g.score()));
        ++games;
    }
};

// Plays `games` games from `seed`; learns when readoutRate > 0.
Totals play(Player& player, const nnt::Params& p, std::uint64_t seed, int games, float readoutRate, float mixRate,
            double explore, const Rewards& rewards)
{
    const int width = static_cast<int>(p.getInt("width")), height = static_cast<int>(p.getInt("height"));
    snake::Rng policy(seed ^ 0x5EEDull);
    Totals totals;
    for (int i = 0; i < games; ++i) {
        snake::Game game(width, height, seed * 1000003ull + static_cast<std::uint64_t>(i));
        snake::Outcome last = snake::Outcome::Moved;
        while (!game.over()) {
            snake::Action action = player.choose(game.state());
            if (explore > 0 && policy.uniform() < explore)
                action = static_cast<snake::Action>(policy.below(3));
            const int before = game.appleDistance();
            last = game.step(action);
            if (readoutRate > 0) {
                float r = 0;
                switch (last) {
                case snake::Outcome::Ate: r = rewards.eat; break;
                case snake::Outcome::Died: r = rewards.die; break;
                default: r = game.appleDistance() < before ? rewards.approach : -rewards.approach;
                }
                const size_t a = static_cast<size_t>(action);
                if (!rewards.errorDriven || player.y[a] * r <= 0.0f)
                    player.learn(a, r, readoutRate, mixRate);
            }
        }
        totals.add(game, last);
    }
    return totals;
}

void record(nnt::Trial& t, const std::string& suffix, const Totals& s)
{
    const double n = s.games;
    t.record("apples" + suffix, s.apples / n);
    t.record("steps" + suffix, s.steps / n);
    t.record("deaths" + suffix, s.deaths / n);  // the rest starved (or filled the field)
    t.record("best" + suffix, s.best);
    t.record("apples_per_100_steps" + suffix, s.steps > 0 ? 100.0 * s.apples / s.steps : 0.0);
}

nnt::Register experiment({
    .name = "snake_rules",
    .description = "snake with a learning rule per layer: readout rule (sign, trace, fa, perturbation) and mix "
                   "rule (frozen or any rule); apples and steps per game",
    .tags = {"control", "learning", "rules"},
    .params = {
        {"readout", "sign", "readout rule: sign, trace, fa, perturbation"},
        {"mix", "frozen", "mix layer: frozen, or a rule: sign, trace, fa, perturbation, oja, bcm"},
        {"mix_size", "64", "mixing neurons (0: readouts read the state only)"},
        {"mix_er", "false", "E-R in the mixing neurons"},
        {"width", "10", "field width in cells (inside the walls)"},
        {"height", "10", "field height in cells"},
        {"train", "200", "training games"},
        {"test", "50", "test games (no learning, no exploration)"},
        {"lr", "0.03", "readout learning rate"},
        {"mix_lr", "0.003", "mix learning rate"},
        {"explore", "0.05", "probability of a random action while training"},
        {"eat", "1", "reward for eating an apple"},
        {"die", "-1", "reward for hitting a wall or itself"},
        {"approach", "0.1", "reward for getting closer to the apple (its negative for moving away)"},
        {"reward", "error", "error: learn only when the readout's sign is wrong; target: learn every step"},
        {"bias", "false", "a learned bias per neuron (every layer that learns)"},
        {"decay", "0", "weight decay (times the learning rate)"},
        {"trace", "0", "trace decay per tick (trace, fa, perturbation)"},
        {"baseline", "0", "trace: reward-baseline rate"},
        {"p_baseline", "0.1", "perturbation: reward-baseline rate"},
        {"noise", "0.1", "perturbation: exploration noise"},
        {"bcm_rate", "0.01", "bcm: sliding threshold rate"},
        {"winners", "0", "oja / bcm: neurons that learn per update (0: all)"},
    },
    .trials = 10,
    .expect = {{.metric = "apples_control", .max = 1.0}},
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Rewards rewards{static_cast<float>(p.getDouble("eat")), static_cast<float>(p.getDouble("die")),
                              static_cast<float>(p.getDouble("approach")), p.getString("reward") == "error"};
        if (!rewards.errorDriven && p.getString("reward") != "target")
            throw std::invalid_argument("reward must be error or target");
        const int train_games = static_cast<int>(p.getInt("train")), test_games = static_cast<int>(p.getInt("test"));
        const std::uint64_t train_seed = 2000 + t.seed(), test_seed = 1000 + t.seed();
        const auto lr = static_cast<float>(p.getDouble("lr")), mix_lr = static_cast<float>(p.getDouble("mix_lr"));

        Player learner(p);
        const auto start = std::chrono::steady_clock::now();
        const Totals training = play(learner, p, train_seed, train_games, lr, mix_lr, p.getDouble("explore"), rewards);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        t.record("train_apples", training.apples / training.games);
        t.record("train_steps", training.steps / training.games);
        t.record("train_us_per_step", 1e6 * seconds / std::max(1.0, training.steps));
        record(t, "", play(learner, p, test_seed, test_games, 0.0f, 0.0f, 0.0, rewards));

        exr::reseed(t.seed());  // the same network, never trained
        Player control(p);
        record(t, "_control", play(control, p, test_seed, test_games, 0.0f, 0.0f, 0.0, rewards));
    },
});

} // namespace
