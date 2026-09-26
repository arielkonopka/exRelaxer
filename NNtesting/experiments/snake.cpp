// Snake, played by a network without a screen (tasks/snake.hpp): each step
// the network sees the 23-value state (the 8 cells around the head, the
// apple's direction, 4 points it sees and their distances, a bias) and
// picks turn left, straight or turn right.
//
//   state (23 sensors) -> "eye" (pass-through) ----------------------+
//                           \-> dense mix (frozen, random) ------------+-> 3 readouts, one per action (learn)
//
// The action is the readout with the largest output (with probability
// `explore` a random one while training). After the move only the chosen
// action's readout learns: reward `eat` for an apple, `die` for hitting a
// wall or itself, else +`approach` for getting closer to the apple and
// -`approach` for moving away. So each readout learns whether its action is
// good in this state: the sign of a one-step value. Error-driven (the
// default), a readout learns only when its sign disagrees with the reward.
//
// The Python version (experiments/snake/) plays the same games.
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>
#include "experiment.hpp"
#include "network.hpp"
#include "snake.hpp"

namespace {

using namespace exr;

struct Player
{
    std::unique_ptr<network> net = std::make_unique<network>();
    std::array<network::LayerId, 3> readouts{};

    explicit Player(const nnt::Params& p)
    {
        const Shape state{1, 1, snake::State::size};
        const auto eye = net->addLayer("eye", LayerSpec::Retina({state}, false, false));
        net->addInputs(eye, state);
        const size_t mix = static_cast<size_t>(p.getInt("mix"));
        network::LayerId features = eye;
        if (mix > 0) {
            features = net->addLayer("mix", LayerSpec::Dense(mix, false, false));
            net->connect(eye, features);
            net->freeze(features);
        }
        const char* names[] = {"left", "straight", "right"};
        for (size_t a = 0; a < 3; ++a) {
            readouts[a] = net->addLayer(names[a], LayerSpec::Dense(1, false, false));
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
};

struct Rewards
{
    float eat, die, approach;
    bool errorDriven;  // learn only when the readout's sign disagrees with the reward
};

struct Totals
{
    double apples = 0, steps = 0, deaths = 0, starved = 0, best = 0;
    int games = 0;
    void add(const snake::Game& g, snake::Outcome last)
    {
        apples += static_cast<double>(g.score());
        steps += static_cast<double>(g.steps());
        deaths += last == snake::Outcome::Died;
        starved += last == snake::Outcome::Starved;
        best = std::max(best, static_cast<double>(g.score()));
        ++games;
    }
};

// Plays `games` games from `seed`; learns when learningRate > 0.
Totals play(Player& player, const nnt::Params& p, std::uint64_t seed, int games, float learningRate,
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
            if (learningRate > 0) {
                float r = 0;
                switch (last) {
                case snake::Outcome::Ate: r = rewards.eat; break;
                case snake::Outcome::Died: r = rewards.die; break;
                default: r = game.appleDistance() < before ? rewards.approach : -rewards.approach;
                }
                const size_t a = static_cast<size_t>(action);
                if (!rewards.errorDriven || player.y[a] * r <= 0.0f)
                    player.net->getLayer(player.readouts[a]).applyReward(r, learningRate);
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
}

nnt::Register experiment({
    .name = "snake",
    .description = "snake without a screen: 23-value state (neighbourhood, apple direction, 4 seen points and "
                   "distances) -> frozen mix -> one learned readout per action",
    .tags = {"control", "learning", "quick"},
    .params = {
        {"width", "10", "field width in cells (inside the walls)"},
        {"height", "10", "field height in cells"},
        {"mix", "64", "frozen random mixing neurons (0: readouts read the state only)"},
        {"train", "200", "training games"},
        {"test", "50", "test games (no learning, no exploration)"},
        {"lr", "0.03", "learning rate"},
        {"explore", "0.05", "probability of a random action while training"},
        {"eat", "1", "reward for eating an apple"},
        {"die", "-1", "reward for hitting a wall or itself"},
        {"approach", "0.1", "reward for getting closer to the apple (its negative for moving away)"},
        {"reward", "error", "error: learn only when the readout's sign is wrong; target: learn every step"},
    },
    .trials = 10,
    .expect = {{.metric = "apples", .min = 10.0}, {.metric = "apples_control", .max = 1.0}},
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Rewards rewards{static_cast<float>(p.getDouble("eat")), static_cast<float>(p.getDouble("die")),
                              static_cast<float>(p.getDouble("approach")), p.getString("reward") == "error"};
        if (!rewards.errorDriven && p.getString("reward") != "target")
            throw std::invalid_argument("reward must be error or target");
        const int train_games = static_cast<int>(p.getInt("train")), test_games = static_cast<int>(p.getInt("test"));
        const std::uint64_t train_seed = 2000 + t.seed(), test_seed = 1000 + t.seed();

        Player learner(p);
        const Totals training = play(learner, p, train_seed, train_games, static_cast<float>(p.getDouble("lr")),
                                     p.getDouble("explore"), rewards);
        t.record("train_apples", training.apples / training.games);
        record(t, "", play(learner, p, test_seed, test_games, 0.0f, 0.0, rewards));

        exr::reseed(t.seed());  // the same network, never trained
        Player control(p);
        record(t, "_control", play(control, p, test_seed, test_games, 0.0f, 0.0, rewards));
    },
});

} // namespace
