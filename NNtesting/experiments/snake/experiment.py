"""Snake played by a network, from Python: the same game, state, network and
learning as the C++ experiment `snake` (NNtesting/experiments/snake.cpp).
See README.md here.

    state (23 sensors) -> "eye" (pass-through) -> frozen random mix
                                  \\---------------+-> 3 readouts (left, straight, right), learned
"""
import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt
import game as snake  # this folder's own module

NAMES = ("left", "straight", "right")


class Player:
    def __init__(self, p):
        self.net = net = exr.Network()
        state = exr.Shape(1, 1, snake.STATE_SIZE)
        eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(state), False, False))
        net.add_inputs(eye, state)
        features = eye
        if p["mix"] > 0:
            features = net.add_layer("mix", exr.LayerSpec.dense(p["mix"], False, False, frozen=True))
            net.connect(eye, features)
        self.readouts = []
        for name in NAMES:
            out = net.add_layer(name, exr.LayerSpec.dense(1, False, False))
            net.connect(eye, out)
            if features != eye:
                net.connect(features, out)
            net.add_output(out)
            self.readouts.append(out)
        self.y = None

    def choose(self, state):
        """The action with the largest output; ties go to straight, then left."""
        self.net.set_inputs(state)
        self.net.step()
        self.y = y = self.net.outputs()
        best = snake.STRAIGHT
        for a in (snake.LEFT, snake.RIGHT):
            if y[a] > y[best]:
                best = a
        return best


def play(player, p, seed, games, learning_rate, explore):
    """Plays `games` games; learns when learning_rate > 0. Returns per-game (score, steps, died)."""
    policy = snake.Rng(seed ^ 0x5EED)
    error_driven = p["reward"] == "error"
    results = []
    for i in range(games):
        g = snake.Game(p["width"], p["height"], (seed * 1000003 + i) & snake.MASK)
        last = snake.MOVED
        while not g.over:
            action = player.choose(g.state())
            if explore > 0 and policy.uniform() < explore:
                action = policy.below(3)
            before = g.apple_distance()
            last = g.step(action)
            if learning_rate > 0:
                if last == snake.ATE:
                    r = p["eat"]
                elif last == snake.DIED:
                    r = p["die"]
                else:
                    r = p["approach"] if g.apple_distance() < before else -p["approach"]
                if not error_driven or player.y[action] * r <= 0:
                    player.net.apply_reward_to(player.readouts[action], r, learning_rate)
        results.append((g.score, g.steps, last == snake.DIED))
    return np.array(results, dtype=np.float64)


def record(t, suffix, results):
    t.record("apples" + suffix, results[:, 0].mean())
    t.record("steps" + suffix, results[:, 1].mean())
    t.record("deaths" + suffix, results[:, 2].mean())
    t.record("best" + suffix, results[:, 0].max())


@nnt.experiment(
    name="snake_py",
    description="snake from Python: 23-value state -> frozen mix -> one learned readout per action "
                "(the C++ experiment snake, same games)",
    tags=["control", "learning", "quick"],
    params={
        "width": (10, "field width in cells (inside the walls)"),
        "height": (10, "field height in cells"),
        "mix": (64, "frozen random mixing neurons (0: readouts read the state only)"),
        "train": (200, "training games"),
        "test": (50, "test games (no learning, no exploration)"),
        "lr": (0.03, "learning rate"),
        "explore": (0.05, "probability of a random action while training"),
        "eat": (1.0, "reward for eating an apple"),
        "die": (-1.0, "reward for hitting a wall or itself"),
        "approach": (0.1, "reward for getting closer to the apple (its negative for moving away)"),
        "reward": ("error", "error: learn only when the readout's sign is wrong; target: learn every step"),
    },
    trials=5,
    expect={"apples": (10.0, None), "apples_control": (None, 1.0)},
)
def run(t):
    p = t.params
    if p["reward"] not in ("error", "target"):
        raise ValueError("reward must be error or target")
    train_seed, test_seed = 2000 + t.seed, 1000 + t.seed
    learner = Player(p)
    training = play(learner, p, train_seed, p["train"], p["lr"], p["explore"])
    t.record("train_apples", training[:, 0].mean())
    record(t, "", play(learner, p, test_seed, p["test"], 0.0, 0.0))

    exr.reseed(t.seed)  # the same network, never trained
    record(t, "_control", play(Player(p), p, test_seed, p["test"], 0.0, 0.0))
