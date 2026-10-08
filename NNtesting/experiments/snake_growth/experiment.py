"""Snake with a network that grows while it learns (doc/development.md, doc/snake_growth_guide.md).

    state (23 sensors) -> eye (pass-through) -> er (E-R, `start` neurons) -> left, straight, right

Learning never stops: the readouts learn from their own error and the E-R layer through feedback
alignment, every step of every game. Meanwhile:

    width  an ActivityMonitor watches er; when the population is saturated (silent with input
           present and raised thresholds) for most of a window, `increment` new plastic neurons
           are added and the old ones are frozen (WidthGrowth)
    prune  neurons with invalid state, or grown neurons that never fire on their input, are removed
           newest first; growth that did not raise the score is undone, newest first (LIFO)
    depth  when the score is above the baseline and has stopped improving (Plateau), a new E-R layer
           is added behind the newest one, which is frozen (grow_depth)

Run it through the harness (`NNtesting/nntest.py run snake_growth`) or on its own, which prints
the development as it happens and can save the grown network:

    python NNtesting/experiments/snake_growth/experiment.py --games 1500 --save grown.exr
"""
import importlib.util
from pathlib import Path

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt
from exrelaxer.development import Plateau, Pruning, WidthGrowth, grow_depth

_game_file = Path(__file__).resolve().parent.parent / "snake" / "game.py"
_spec = importlib.util.spec_from_file_location("snake_game", _game_file)
snake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snake)

ACTIONS = ("left", "straight", "right")

DEFAULTS = {
    "width": 10, "height": 10,          # the board, inside the walls
    "start": 4,                         # E-R neurons at the start (1: start from one cell)
    "minimum": 0,                       # protected core; 0: the starting size
    "recovery": 0.9,                    # E-R threshold recovery of the adaptive layers
    "increment": 2,                     # neurons per width growth
    "freeze_old": True,                 # width growth freezes the neurons the layer had
    "max_width": 16,                    # no layer grows past this
    "window": 200,                      # ticks the saturation is judged over
    "saturated_share": 0.5,             # share of saturated ticks in the window that triggers growth
    "eval_games": 50,                   # games per evaluation window (one score = their mean apples)
    "games": 1500,                      # games played (and learned from) in all
    "patience": 4,                      # evaluations without a new best before a plateau
    "margin": 0.5,                      # the best must beat the baseline by this many apples
    "max_depth": 2,                     # adaptive E-R layers at most
    "deep_size": 4,                     # neurons of a layer added by depth growth; 0: as many as
                                        # the layer it grows behind
    "deep_minimum": 0,                  # protected size of such a layer; 0: deep_size
    "prune_base": False,                # pruning may also remove a layer's original neurons (never
                                        # below its minimum), not only grown ones
    "freeze_from": 2,                   # adding adaptive layer number >= this freezes every older
                                        # layer; earlier depth growth leaves them learning
    "undo": 3,                          # evaluations after a width growth before an unhelpful one is
                                        # undone (LIFO); 0: never undo
    "lr": 0.03, "explore": 0.05,
    "eat": 1.0, "die": -1.0, "approach": 0.1,
    "grow": True,                       # False: the same run with growth and pruning off (control)
}


class Agent:
    """The network, its development policies and their counters."""

    def __init__(self, p):
        self.p = p
        self.net = net = exr.Network()
        state = exr.Shape(1, 1, snake.STATE_SIZE)
        self.eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(state), False, False))
        net.add_inputs(self.eye, state, "state")
        self.er = net.add_layer("er", self.adaptive_spec(p["start"]))
        net.connect(self.eye, self.er)
        self.readouts = []
        for name in ACTIONS:
            out = net.add_layer(name, exr.LayerSpec.dense(1, False, False))
            net.connect(self.er, out)
            net.add_output(out)
            self.readouts.append(out)
        net.set_minimum_size(self.er, p["minimum"] or p["start"])
        self.layers = [self.er]   # adaptive E-R layers, oldest first; the newest one grows
        self.tune(self.er)
        self.monitor = self.new_monitor()
        self.width = WidthGrowth(increment=p["increment"], max_size=p["max_width"],
                                 freeze_existing=p["freeze_old"])
        self.pruning = Pruning(reasons=("invalid", "inactive"), inactive_after=5000,
                               only_grown=not p["prune_base"])
        self.depth_events, self.undo_events = [], []
        self.pending = None       # the last width growth: (evaluation, neurons added, score before,
                                  # neurons frozen before)
        self.hold_until = -1      # no width growth before this evaluation (after an undo)
        self.evaluation, self.last_score = 0, None
        self.ticks = 0

    def adaptive_spec(self, size):
        return exr.LayerSpec.dense(size, False, True,
                                   learning_rule=exr.LearningRule.feedback_alignment())

    def tune(self, layer):
        if abs(self.p["recovery"] - exr.constants.recovery_factor) > 1e-6:
            self.net.set_recovery_jitter(layer, exr.Jitter.uniform(1e-7).around(self.p["recovery"]))

    def new_monitor(self):
        return exr.ActivityMonitor(exr.ActivitySpec(window=self.p["window"],
                                                    saturated_share=self.p["saturated_share"]))

    @property
    def newest(self):
        return self.layers[-1]

    def sizes(self):
        return [self.net.layer_size(layer) for layer in self.layers]

    # --- Playing and learning ---------------------------------------------------------------
    def choose(self, state):
        """The action with the largest readout; ties go to straight, then left."""
        self.net.set_inputs("state", np.asarray(state, np.float32))
        self.net.step()
        if self.p["grow"]:
            self.monitor.observe(self.net, self.newest)
        self.y = y = self.net.outputs()
        best = snake.STRAIGHT
        for a in (snake.LEFT, snake.RIGHT):
            if y[a] > y[best]:
                best = a
        return best

    def learn(self, action, reward):
        """Error-driven: the chosen action's readout learns when its sign disagrees with the reward;
        the E-R layers learn from the same error through their fixed random feedback."""
        if self.y[action] * reward > 0:
            return
        errors = np.zeros(len(ACTIONS), np.float32)
        errors[action] = reward
        self.net.apply_error(errors, self.p["lr"])

    def develop(self):
        """Width growth and pruning, between steps."""
        if not self.p["grow"]:
            return
        layer = self.newest
        size = self.net.layer_size(layer)
        if self.evaluation >= self.hold_until:
            frozen = set(self.net.frozen_neurons(layer))
            added = self.width.update(self.net, layer, self.monitor, self.ticks)
            if added:
                self.pending = (self.evaluation, added, self.last_score, frozen)
        self.pruning.update(self.net, layer, self.monitor, self.ticks)
        if self.net.layer_size(layer) != size:
            self.monitor = self.new_monitor()

    def evaluated(self, score):
        """After each evaluation window: undo a width growth that did not raise the score (LIFO:
        the newest neurons go, and the neurons the growth froze learn again)."""
        if self.pending is not None and self.p["undo"] > 0:
            at, added, before, frozen = self.pending
            if self.evaluation - at >= self.p["undo"]:
                if before is not None and score <= before:
                    layer = self.newest
                    removed = self.net.prune_newest(layer, added)
                    for i in range(self.net.layer_size(layer)):
                        if i not in frozen:
                            self.net.freeze_neurons(layer, i, 1, False)
                    self.undo_events.append((self.ticks, removed, self.net.layer_size(layer)))
                    self.monitor = self.new_monitor()
                    self.hold_until = self.evaluation + 1 + self.p["undo"]
                self.pending = None
        self.last_score = score
        self.evaluation += 1

    def deepen(self):
        """Depth growth: a new adaptive layer behind the newest. From layer `freeze_from` on, every
        older layer is frozen; before it they keep learning."""
        before = self.sizes()
        freeze = len(self.layers) + 1 >= self.p["freeze_from"]
        size = self.p["deep_size"] or self.net.layer_size(self.newest)
        deep = grow_depth(self.net, self.newest, readers=self.readouts,
                          spec=self.adaptive_spec(size),
                          name=f"er{len(self.layers) + 1}", freeze_previous=freeze)
        if freeze:
            for layer in self.layers[:-1]:
                self.net.freeze_neurons(layer, 0, self.net.layer_size(layer))
        self.net.set_minimum_size(deep, min(self.p["deep_minimum"] or size, size))
        self.tune(deep)
        self.layers.append(deep)
        self.monitor = self.new_monitor()
        self.pending = None
        self.depth_events.append((self.ticks, before, self.sizes()))


def play(agent, p, seed, games, learning):
    """Plays `games` games; returns per-game (apples, steps, died)."""
    policy = snake.Rng(seed ^ 0x5EED)
    results = []
    for i in range(games):
        g = snake.Game(p["width"], p["height"], (seed * 1000003 + i) & snake.MASK)
        last = snake.MOVED
        while not g.over:
            action = agent.choose(g.state())
            if learning and p["explore"] > 0 and policy.uniform() < p["explore"]:
                action = policy.below(3)
            before = g.apple_distance()
            last = g.step(action)
            agent.ticks += 1
            if learning:
                if last == snake.ATE:
                    r = p["eat"]
                elif last == snake.DIED:
                    r = p["die"]
                else:
                    r = p["approach"] if g.apple_distance() < before else -p["approach"]
                agent.learn(action, r)
                agent.develop()
        results.append((g.score, g.steps, last == snake.DIED))
    return np.array(results, dtype=np.float64)


def develop(p, seed, report=None):
    """The whole developmental run. `report(text)` sees each event as it happens."""
    say = report or (lambda text: None)
    exr.reseed(seed)
    agent = Agent(p)
    baseline_games = play(agent, p, 50_000 + seed, p["eval_games"], learning=False)
    baseline = baseline_games[:, 0].mean()
    plateau = Plateau(baseline=baseline, margin=p["margin"], patience=p["patience"])
    say(f"baseline {baseline:.2f} apples/game, network {agent.sizes()}")
    curve, best_game, steps = [], 0, 0
    agent.last_score = baseline
    for k in range(p["games"] // p["eval_games"]):
        widths, prunes = len(agent.width.events), len(agent.pruning.events) + len(agent.undo_events)
        games = play(agent, p, seed * 7919 + k, p["eval_games"], learning=True)
        score = games[:, 0].mean()
        curve.append(score)
        best_game = max(best_game, int(games[:, 0].max()))
        steps += int(games[:, 1].sum())
        for tick, old, new in agent.width.events[widths:]:
            say(f"  tick {tick}: saturated, width {old} -> {new}")
        agent.evaluated(score)
        for tick, removed, size in (agent.pruning.events + agent.undo_events)[prunes:]:
            say(f"  tick {tick}: pruned {removed} -> {size}")
        say(f"evaluation {k}: {score:.2f} apples/game, network {agent.sizes()}")
        if p["grow"] and plateau.update(score) and len(agent.layers) < p["max_depth"]:
            agent.deepen()
            plateau.restart()
            say(f"  plateau at {plateau.baseline:.2f} (baseline {baseline:.2f}): depth {agent.depth_events[-1][1]}"
                f" -> {agent.sizes()}")
    return agent, {
        "baseline": baseline,
        "first": curve[0],
        "final": float(np.mean(curve[-3:])),
        "best_window": max(curve),
        "best_game": best_game,
        "steps": steps,
        "ticks": agent.ticks,
        "size": sum(agent.sizes()),
        "layers": len(agent.layers),
        "width_events": len(agent.width.events),
        "prune_events": len(agent.pruning.events) + len(agent.undo_events),
        "depth_events": len(agent.depth_events),
        "curve": curve,
    }


@nnt.experiment(
    name="snake_growth",
    description="snake learned always-on by a network that grows: E-R saturation -> width, "
                "LIFO undo of unhelpful growth, score plateau -> depth (doc/development.md)",
    tags=["control", "learning", "development"],
    params={key: (value, "") for key, value in DEFAULTS.items()},
    trials=3,
)
def run(t):
    p = dict(DEFAULTS, **t.params)
    _, m = develop(p, t.seed)
    for key, value in m.items():
        if key != "curve":
            t.record(key, float(value))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, value in DEFAULTS.items():
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(value, bool) else type(value)
        parser.add_argument("--" + key.replace("_", "-"), type=kind, default=value)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", help="save the grown network here (Network.save)")
    args = parser.parse_args()
    params = {key: getattr(args, key) for key in DEFAULTS}
    agent, metrics = develop(params, args.seed, report=print)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in metrics.items() if k != "curve"})
    print(agent.net.describe())
    if args.save:
        agent.net.save(args.save)
