// The headless snake game (NNtesting/tasks/snake.hpp): rules, the state
// vector, and reference values that the Python copy
// (NNtesting/experiments/snake/game.py) must reproduce exactly
// (EXrelaxer.py/tests/test_snake.py checks the same numbers).
#include <gtest/gtest.h>
#include <vector>
#include "snake.hpp"

using namespace snake;

namespace {

// Turns towards the apple when that is safe, else any safe way.
Action greedy(const Game& g)
{
    Action best = Action::Straight;
    int best_distance = 1 << 30;
    for (Action a : {Action::Straight, Action::Left, Action::Right}) {
        const Cell d = g.relative(static_cast<int>(a) - 1);
        const Cell next{g.head().x + d.x, g.head().y + d.y};
        if (g.blocked(next) && !(next == g.body().back()))
            continue;
        const int distance = std::abs(g.apple().x - next.x) + std::abs(g.apple().y - next.y);
        if (distance < best_distance) {
            best_distance = distance;
            best = a;
        }
    }
    return best;
}

} // namespace

TEST(SnakeTest, RandomNumbersMatchTheReference)
{
    Rng r(123);
    EXPECT_EQ(r.next(), 13032462758197477675ull);
    EXPECT_EQ(r.next(), 18015028434894305148ull);
}

TEST(SnakeTest, ReferenceGameRunsIntoTheWall)
{
    Game g(10, 10, 42);
    EXPECT_EQ(g.heading(), 1);  // right
    EXPECT_EQ(g.apple(), (Cell{6, 0}));
    EXPECT_EQ(g.body().size(), 3u);
    EXPECT_EQ(g.head(), (Cell{5, 5}));
    const std::vector<int> actions{1, 1, 2, 2, 0, 1, 1, 0, 0, 1, 2, 1, 1};
    for (size_t i = 0; i + 1 < actions.size(); ++i)
        EXPECT_EQ(g.step(static_cast<Action>(actions[i])), Outcome::Moved) << "step " << i;
    EXPECT_EQ(g.head(), (Cell{9, 7}));
    EXPECT_EQ(g.step(Action::Straight), Outcome::Died);  // into the right wall
    EXPECT_TRUE(g.over());
    EXPECT_EQ(g.steps(), 13u);
}

TEST(SnakeTest, EatingGrowsTheSnakeAndMovesTheApple)
{
    Game g(10, 10, 42);
    size_t length = g.body().size();
    for (int i = 0; i < 400 && g.score() < 5 && !g.over(); ++i) {
        const Cell apple = g.apple();
        const Outcome o = g.step(greedy(g));
        if (o == Outcome::Ate) {
            EXPECT_EQ(g.head(), apple);
            EXPECT_EQ(g.body().size(), ++length);
            EXPECT_FALSE(g.blocked(g.apple()));  // never on the snake
        } else {
            EXPECT_EQ(o, Outcome::Moved);
            EXPECT_EQ(g.body().size(), length);
        }
    }
    EXPECT_EQ(g.score(), 5u);
}

TEST(SnakeTest, TheHeadMayTakeTheTailsCellAndCirclingStarves)
{
    // Length 4 circling a 2 x 2 square: the head always enters the cell the
    // tail leaves in the same step.
    Game g(10, 10, 3, 4, 6);
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(g.step(Action::Right), Outcome::Moved) << "step " << i;
    EXPECT_EQ(g.body().size(), 4u);
    EXPECT_EQ(g.step(Action::Right), Outcome::Starved);  // 6 steps without an apple
    EXPECT_TRUE(g.over());

    Game h(10, 10, 3, 5);  // length 5 in the same circle bites itself
    Outcome o = Outcome::Moved;
    for (int i = 0; i < 4 && o == Outcome::Moved; ++i)
        o = h.step(Action::Right);
    EXPECT_EQ(o, Outcome::Died);
}

TEST(SnakeTest, StateMatchesTheReference)
{
    Game g(10, 10, 7);
    g.step(Action::Right);
    EXPECT_EQ(g.head(), (Cell{5, 4}));
    EXPECT_EQ(g.heading(), 0);
    EXPECT_EQ(g.apple(), (Cell{9, 8}));
    const std::vector<float> expected{-1, -1, -1, -1, -1, -1, 1, 1,                 // body behind
                                      -0.707106769f, 0.707106769f,                     // apple back-right
                                      0.0909090936f, -1, -1, -0.0909090936f,           // forward: top wall; left
                                      1, -0.0909090936f, 0.0909090936f, 0.0909090936f,  // right wall; back: own body
                                      -0.600000024f, -0.666666687f, -0.600000024f, 1, 1};
    const State s = g.state();
    ASSERT_EQ(s.values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(s.values[i], expected[i]) << "value " << i;
}
