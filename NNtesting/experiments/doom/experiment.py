"""Doom (ViZDoom) played from pixels by a small network that imitates an
oracle. The top rung of the dynamic ladder (dyn_ladder): does E-R's state
help when the right action depends on how the screen is changing? See
README.md here.

    screen (gray, pooled) [+ the last `window` screens] -> hidden (relu | er | gate) -> 3 action outputs

Scenarios:
  basic             a monster stands still on the far wall: strafe until it
                    is in front, then shoot. Reactive: one frame is enough.
  predict_position  a monster walks sideways across the far side of the
                    room; the agent turns and has one slow rocket, so it
                    must aim where the monster *will be*. Which way the
                    monster walks cannot be read from one frame.

Learning is imitation with the agent in control (DAgger-style): the agent's
own action is executed, and every step the outputs learn the oracle's
action for that state (one-hot target, feedback alignment, learned bias).
The oracle uses the game's object positions, which the network never sees.
No term counts activity.
"""
import math
import os

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

try:
    import vizdoom as vzd
except ImportError:  # the experiment reports it in run()
    vzd = None

ROCKET_SPEED = 17.0   # oracle: map units per tic used for the intercept
AIM_TOLERANCE = 4.0   # oracle: degrees; shoot when the aim is this close


def make_game(scenario, seed, frame_skip):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_objects_info_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_available_game_variables([vzd.GameVariable.ANGLE, vzd.GameVariable.POSITION_X,
                                       vzd.GameVariable.POSITION_Y])
    game.set_seed(seed)
    game.init()
    return game


class Oracle:
    """The action a player who knows the monster's position (and, for
    predict_position, its velocity) would take: 0 left, 1 right, 2 shoot."""

    def __init__(self, scenario, frame_skip):
        self.scenario, self.skip = scenario, frame_skip
        self.first = None
        self.steps = 0

    def reset(self):
        self.first, self.steps = None, 0

    def __call__(self, state):
        self.steps += 1
        if self.scenario == "basic":
            # Buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK. Centre the monster on screen.
            monsters = [l for l in state.labels if l.object_name != "DoomPlayer"]
            if not monsters:
                return 2
            m = monsters[0]
            dx = m.x + m.width / 2 - 80
            return 2 if abs(dx) <= 3 else (1 if dx > 0 else 0)
        # predict_position. Buttons: TURN_LEFT, TURN_RIGHT, ATTACK. Aim at the
        # intercept of a rocket and the monster's straight-line walk.
        angle, px, py = state.game_variables
        monsters = [o for o in state.objects if o.name == "Cacodemon"]
        if not monsters:
            return 2
        m = np.array([monsters[0].position_x, monsters[0].position_y])
        if self.first is None:
            self.first = (m, self.steps)
        elapsed = (self.steps - self.first[1]) * self.skip
        v = (m - self.first[0]) / elapsed if elapsed else np.zeros(2)
        d = m - np.array([px, py])
        a, b, c = v @ v - ROCKET_SPEED ** 2, 2 * d @ v, d @ d
        disc = max(b * b - 4 * a * c, 0.0)
        t = (-b - math.sqrt(disc)) / (2 * a)
        if t < 0:
            t = (-b + math.sqrt(disc)) / (2 * a)
        aim = d + v * max(t, 0.0)
        diff = (math.degrees(math.atan2(aim[1], aim[0])) - angle + 180) % 360 - 180
        if abs(diff) <= AIM_TOLERANCE and elapsed:
            return 2
        return 0 if diff > 0 else 1  # TURN_LEFT raises the angle


class Player:
    """screen window -> hidden -> 3 outputs; each frame is held for depth + 1 + settle ticks."""

    def __init__(self, p, frame_size):
        model, width = p["model"], p["width"]
        if model not in ("relu", "er", "gate", "clamp"):
            raise ValueError("model must be relu, er, gate or clamp")
        self.frame_size, self.window = frame_size, p["window"]
        inputs = frame_size * (self.window + 1)
        rule = exr.LearningRule.feedback_alignment().with_bias()
        self.net = net = exr.Network()
        spec = exr.LayerSpec.dense(width, False, model == "er", learning_rule=rule)
        spec.rectify = model == "relu"
        if model == "gate":
            spec.gate = p["gate"]
        self.hidden = net.add_layer("h", spec)
        self.out = net.add_layer("out", exr.LayerSpec.dense(3, False, False, learning_rule=rule))
        net.add_inputs(self.hidden, inputs, "x")
        net.connect(self.hidden, self.out)
        net.add_output(self.out)
        # The same initialization as nl_temporal: uniform, variance 1 / fan-in.
        for layer, fan_in, n in ((self.hidden, inputs, width), (self.out, width, 3)):
            scale = math.sqrt(3.0 / fan_in)
            for i in range(n):
                net.set_weights(layer, i, net.weights(layer, i) * scale)
        self.hold = 2 + p["settle"]
        self.frames = []
        self.spikes = 0.0
        self.ticks = 0

    def new_episode(self):
        self.frames = []

    def act(self, frame, meter=False):
        self.frames.insert(0, frame)
        del self.frames[self.window + 1:]
        x = np.zeros(self.frame_size * (self.window + 1), dtype=np.float32)
        for w, f in enumerate(self.frames):
            x[w * self.frame_size:(w + 1) * self.frame_size] = f
        self.net.set_inputs(x)
        for _ in range(self.hold):
            self.net.step()
            if meter:
                self.spikes += np.count_nonzero(np.abs(self.net.layer_output(self.hidden)) > 1e-6)
                self.ticks += 1
        self.y = np.asarray(self.net.outputs(), dtype=np.float32)
        return int(np.argmax(self.y))

    def learn(self, action, lr):
        target = np.zeros(3, dtype=np.float32)
        target[action] = 1.0
        self.net.apply_error(target - self.y, lr)


def pool(screen, factor):
    h, w = screen.shape
    s = screen[:h - h % factor, :w - w % factor].astype(np.float32) / 255.0
    return s.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)).ravel()


def won(game, p):
    """predict_position: the rocket hit (reward > 0). basic: the monster died before the timeout."""
    if p["scenario"] == "predict_position":
        return game.get_total_reward() > 0
    return game.get_episode_time() < game.get_episode_timeout()


def play(game, player, oracle, p, episodes, lr, meter=False):
    """Plays `episodes` episodes; learns when lr > 0. Returns (wins, rewards, agreement, steps)."""
    buttons = np.eye(3, dtype=int).tolist()
    wins, rewards, agree, steps = 0, [], 0, 0
    for _ in range(episodes):
        game.new_episode()
        player.new_episode()
        oracle.reset()
        while not game.is_episode_finished():
            state = game.get_state()
            want = oracle(state)
            a = player.act(pool(state.screen_buffer, p["pool"]), meter)
            if lr > 0:
                player.learn(want, lr)
            agree += a == want
            steps += 1
            game.make_action(buttons[a], p["frame_skip"])
        rewards.append(game.get_total_reward())
        wins += won(game, p)
    return wins / episodes, float(np.mean(rewards)), agree / max(steps, 1), steps


def play_oracle(game, oracle, p, episodes, lead=True):
    """The oracle's own score (lead=False: aims at where the monster is now)."""
    global ROCKET_SPEED
    saved = ROCKET_SPEED
    if not lead:
        ROCKET_SPEED = 1e9
    buttons = np.eye(3, dtype=int).tolist()
    wins = 0
    try:
        for _ in range(episodes):
            game.new_episode()
            oracle.reset()
            while not game.is_episode_finished():
                game.make_action(buttons[oracle(game.get_state())], p["frame_skip"])
            wins += won(game, p)
    finally:
        ROCKET_SPEED = saved
    return wins / episodes


@nnt.experiment(
    name="doom",
    description="ViZDoom from pooled pixels (basic, predict_position): relu, E-R, gate hidden neurons with "
                "an optional frame window, imitating an oracle; win rate, reward and spikes",
    tags=["er", "dynamic", "control", "learning"],
    params={
        "scenario": ("predict_position", "basic (reactive) or predict_position (must lead a moving target)"),
        "model": ("er", "hidden neurons: relu, er, gate or clamp"),
        "window": (0, "extra past frames given to the network (0: the current frame only)"),
        "width": (64, "hidden neurons"),
        "lr": (0.003, "learning rate"),
        "gate": (0.2, "gate model: the fixed threshold"),
        "settle": (1, "extra ticks per frame after the input reaches the output"),
        "pool": (4, "screen (160 x 120) pooling factor: 4 gives 40 x 30 inputs per frame"),
        "frame_skip": (4, "game tics per action"),
        "train": (600, "training episodes"),
        "test": (100, "test episodes (no learning)"),
        "oracle_episodes": (100, "episodes to measure the oracle with and without leading (0: skip)"),
    },
    trials=3,
)
def run(t):
    if vzd is None:
        raise RuntimeError("vizdoom is not installed: python3 -m pip install vizdoom")
    exr.set_threads(1)  # one game step is far too small for OpenMP; threads only add waiting
    p = t.params
    if p["scenario"] not in ("basic", "predict_position"):
        raise ValueError("scenario must be basic or predict_position")
    frame_size = (120 // p["pool"]) * (160 // p["pool"])
    oracle = Oracle(p["scenario"], p["frame_skip"])
    player = Player(p, frame_size)

    train_game = make_game(p["scenario"], 1000 + t.seed, p["frame_skip"])
    win, reward, agree, steps = play(train_game, player, oracle, p, p["train"], p["lr"])
    train_game.close()
    t.record("train_win_rate", win)
    t.record("train_steps", steps)

    test_game = make_game(p["scenario"], 5000 + t.seed, p["frame_skip"])
    win, reward, agree, steps = play(test_game, player, oracle, p, p["test"], 0.0, meter=True)
    t.record("win_rate", win)
    t.record("reward", reward)
    t.record("oracle_agreement", agree)
    t.record("steps_per_episode", steps / p["test"])
    t.record("spikes_per_step", player.spikes / max(steps, 1))
    t.record("active_fraction", player.spikes / max(player.ticks, 1) / p["width"])
    t.record("inputs", frame_size * (p["window"] + 1))
    t.record("hidden_neurons", p["width"])
    if p["oracle_episodes"] > 0:
        t.record("oracle_win_rate", play_oracle(test_game, oracle, p, p["oracle_episodes"]))
        if p["scenario"] == "predict_position":
            t.record("oracle_no_lead_win_rate", play_oracle(test_game, oracle, p, p["oracle_episodes"], lead=False))
    test_game.close()
