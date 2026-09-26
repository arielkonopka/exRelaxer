"""The Python snake game (NNtesting/experiments/snake/game.py) against the
reference values of the C++ game's tests (tests/snake.cpp), and its
experiment's player."""
import importlib.util

import numpy as np
import pytest

from exrelaxer import datasets

NNTESTING = datasets.find_nntesting(__file__)
if NNTESTING is None:
    pytest.skip("not in the repository", allow_module_level=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snake = load("game", NNTESTING / "experiments" / "snake" / "game.py")


def test_random_numbers_match_the_cpp_reference():
    r = snake.Rng(123)
    assert r.next() == 13032462758197477675
    assert r.next() == 18015028434894305148


def test_reference_game_runs_into_the_wall():
    g = snake.Game(10, 10, 42)
    assert g.heading == 1 and g.apple == (6, 0) and g.head == (5, 5) and len(g.body) == 3
    for i, a in enumerate([1, 1, 2, 2, 0, 1, 1, 0, 0, 1, 2, 1]):
        assert g.step(a) == snake.MOVED, i
    assert g.head == (9, 7)
    assert g.step(snake.STRAIGHT) == snake.DIED
    assert g.over and g.steps == 13


def test_circling_starves_and_a_longer_snake_bites_itself():
    g = snake.Game(10, 10, 3, length=4, starve_after=6)
    for _ in range(5):
        assert g.step(snake.RIGHT) == snake.MOVED
    assert g.step(snake.RIGHT) == snake.STARVED
    h = snake.Game(10, 10, 3, length=5)
    outcomes = [h.step(snake.RIGHT) for _ in range(4)]
    assert outcomes[-1] == snake.DIED


def test_state_matches_the_cpp_reference():
    g = snake.Game(10, 10, 7)
    g.step(snake.RIGHT)
    assert g.head == (5, 4) and g.heading == 0 and g.apple == (9, 8)
    expected = np.array([-1, -1, -1, -1, -1, -1, 1, 1,
                         -0.707106769, 0.707106769,
                         0.0909090936, -1, -1, -0.0909090936, 1, -0.0909090936, 0.0909090936, 0.0909090936,
                         -0.600000024, -0.666666687, -0.600000024, 1, 1], dtype=np.float32)
    assert np.array_equal(g.state(), expected)


def test_eating_grows_the_snake():
    g = snake.Game(10, 10, 42)
    length = len(g.body)
    for _ in range(400):
        if g.score >= 5 or g.over:
            break
        # Greedy: the safe move that gets closest to the apple.
        best, best_distance = snake.STRAIGHT, None
        for a in (snake.STRAIGHT, snake.LEFT, snake.RIGHT):
            dx, dy = g.relative(a - 1)
            nxt = (g.head[0] + dx, g.head[1] + dy)
            if g.blocked(nxt) and nxt != g.body[-1]:
                continue
            d = abs(g.apple[0] - nxt[0]) + abs(g.apple[1] - nxt[1])
            if best_distance is None or d < best_distance:
                best, best_distance = a, d
        apple = g.apple
        if g.step(best) == snake.ATE:
            length += 1
            assert g.head == apple and not g.blocked(g.apple)
        assert len(g.body) == length
    assert g.score == 5


def test_the_player_learns(monkeypatch):
    monkeypatch.syspath_prepend(str(NNTESTING / "experiments" / "snake"))
    from exrelaxer import harness
    monkeypatch.setattr(harness, "_registry", [])
    experiment = load("snake_experiment", NNTESTING / "experiments" / "snake" / "experiment.py")
    import exrelaxer as exr
    p = {"width": 8, "height": 8, "mix": 32, "lr": 0.03, "explore": 0.05, "eat": 1.0, "die": -1.0,
         "approach": 0.1, "reward": "error"}
    exr.reseed(1)
    player = experiment.Player(p)
    before = experiment.play(player, p, 5, 20, 0.0, 0.0)[:, 0].mean()
    experiment.play(player, p, 6, 100, p["lr"], p["explore"])
    after = experiment.play(player, p, 5, 20, 0.0, 0.0)[:, 0].mean()
    assert after > before + 3
