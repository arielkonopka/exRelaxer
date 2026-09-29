"""Doom learned from reward, seeing and hearing the game at its smallest
settings. See README.md here.

    screen 160 x 120 gray -> pooled 40 x 30 ("eye", 1200 sensors) ----------\\
    stereo sound 11025 Hz -> "mic" (2 x 420 samples per tick) -> cochlea    +-> hidden (relu | er | gate)
                             (2 ears x 16 bands) --------------------------/          |
                                                                     8 action readouts (learned from reward)

Actions (one per step): forward, backward, turn left, turn right, strafe
left, strafe right, shoot, use (doors, switches). The reward comes only
from the game's state (rewards.py): hurt and death are penalised, kills,
ammo, armor, items, keys, opened doors and leaving the level rewarded,
firing and idling cost a little. No term counts neural activity.

Learning follows the snake experiments (research log conclusion 3): the
hidden layer is a frozen random mix by default and the readouts learn. After
each step the chosen action's readout gets that step's reward
(`apply_reward_to`); with the default trace rule the readout's input and
output traces spread it over the last steps. Exploration: epsilon-greedy.
"""
import math
import os

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

try:
    import vizdoom as vzd
    import rewards
except ImportError:  # the experiment reports it in run()
    vzd = None

ACTIONS = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "USE"]
SAMPLE_RATE = 11025
FRAME_SKIP = 4                              # game tics per action (35 tics a second)
STEP_SAMPLES = SAMPLE_RATE * FRAME_SKIP // 35  # 1260 per ear per step


def make_game(p, seed):
    game = vzd.DoomGame()
    if p["scenario"] == "map01":
        game.load_config(os.path.join(vzd.scenarios_path, "freedoom2.cfg"))
        game.set_doom_map("map01")
    else:
        game.load_config(os.path.join(vzd.scenarios_path, p["scenario"] + ".cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)  # the smallest ViZDoom offers
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_render_messages(False)
    game.set_automap_buffer_enabled(False)
    game.set_depth_buffer_enabled(False)
    game.set_labels_buffer_enabled(False)
    game.set_sound_enabled(p["sound"])
    game.set_audio_buffer_enabled(p["sound"])
    if p["sound"]:
        game.set_audio_sampling_rate(vzd.SamplingRate.SR_11025)
        game.set_audio_buffer_size(FRAME_SKIP)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_available_buttons([getattr(vzd.Button, a) for a in ACTIONS])
    game.set_available_game_variables(rewards.VARIABLES)
    game.set_episode_timeout(p["episode_tics"])
    game.set_living_reward(0)
    game.set_doom_skill(p["skill"])
    game.set_seed(seed)
    game.init()
    return game


class Player:
    def __init__(self, p):
        model, width = p["model"], p["width"]
        if model not in ("relu", "er", "gate", "clamp"):
            raise ValueError("model must be relu, er, gate or clamp")
        self.pool, self.ticks = p["pool"], p["ticks"]
        self.eye_size = (120 // self.pool) * (160 // self.pool)
        self.hop = STEP_SAMPLES // self.ticks
        self.net = net = exr.Network()
        self.sound = p["sound"]
        self.readout = p.get("readout", "last")
        def neurons(size, frozen=True):
            spec = exr.LayerSpec.dense(size, p["habituation"], model == "er", frozen=frozen,
                                       learning_rule=exr.LearningRule.traced(p["trace"]))
            if p["habituation"]:
                # Fade mode as in the rerun's fade2: a repeated input fades by
                # `habituation_decay` per tick from its `habituation_fade_after`th repeat.
                spec.habituation_rule = exr.Habituation(tolerance=p["habituation_tolerance"],
                                                        decay=p["habituation_decay"],
                                                        fade_after=p["habituation_fade_after"])
            spec.rectify = model == "relu"
            if model == "gate":
                spec.gate = p["gate"]
            if model == "er":
                spec.spontaneous = exr.Spontaneous(amplitude=p.get("spontaneous_amplitude", 0.01))
            return normalized(spec)

        # Settings saved before the 2026-09-28 defaults lack these keys: they
        # rebuild as they were evolved (raw sums, spontaneous amplitude 0.01).
        def normalized(spec):
            if p.get("normalize", False):
                if not hasattr(spec, "normalize"):
                    raise RuntimeError("normalize needs exrelaxer with LayerSpec.normalize (network format 16)")
                spec.normalize = True
            elif hasattr(spec, "normalize"):
                spec.normalize = False
            return spec

        # Hidden stack h1 .. h<depth>: h1 reads the eye (and the ears), each
        # further layer the one below. All frozen unless learn_hidden.
        if p["feedback"] not in ("none", "recurrent", "topdown", "both"):
            raise ValueError("feedback must be none, recurrent, topdown or both")
        self.layers = []
        for l in range(p["depth"]):
            h = net.add_layer(f"h{l + 1}", neurons(width, not p["learn_hidden"]))
            if l == 0:
                net.add_inputs(h, self.eye_size, "eye")
            else:
                net.connect(self.layers[-1], h)
            self.layers.append(h)
        self.hidden = self.layers[0]  # metered layer set below
        if self.sound:
            window = max(512, 1 << (self.hop - 1).bit_length())  # a power of two >= hop
            spec = exr.CochleaSpec(sample_rate=SAMPLE_RATE, hop=self.hop, window=window, bands=p["bands"], channels=2)
            self.ear = net.add_layer("ear", exr.LayerSpec.cochlea(spec, False, False))
            net.add_inputs(self.ear, 2 * self.hop, "mic")
            net.connect(self.ear, self.layers[0])
        # Feedback lines. recurrent: every neuron of a hidden layer also reads
        # that layer's previous output. topdown: `feedback_width` extra neurons
        # in each layer read the layer above (the top layer is read by the
        # layer below it). Both are frozen and random, like the stack.
        recurrent = p["feedback"] in ("recurrent", "both")
        if p["feedback"] in ("topdown", "both"):
            for l in range(p["depth"] - 1):
                net.add_feedback(self.layers[l + 1], self.layers[l], p["feedback_width"])
        if recurrent:
            for h in self.layers:
                net.connect(h, h)
        # Reservoir (echo state): `reservoir` input neurons read the top hidden
        # layer, `reservoir_recurrent` more read the whole reservoir.
        self.reservoir = None
        if p["reservoir"] > 0:
            self.reservoir = net.add_layer("reservoir", neurons(p["reservoir"]))
            net.connect(self.layers[-1], self.reservoir)
            if p["reservoir_recurrent"] > 0:
                net.add_feedback(self.reservoir, self.reservoir, p["reservoir_recurrent"])
        rule = exr.LearningRule.sign() if p["rule"] == "sign" else exr.LearningRule.traced(p["trace"])
        self.readouts = []
        for name in ACTIONS:
            out = net.add_layer(name.lower(), normalized(exr.LayerSpec.dense(1, False, False, learning_rule=rule)))
            for h in (self.layers if p.get("readout_from", "top") == "all" else self.layers[-1:]):
                net.connect(h, out)
            if self.reservoir is not None:
                net.connect(self.reservoir, out)
            net.add_output(out)
            self.readouts.append(out)
        # Uniform weights with variance 1 / fan-in. Recurrent weights (a
        # layer reading itself, the reservoir's recurrent neurons) get
        # recurrent_scale / sqrt(layer size) instead: below 1 the echo fades.
        rs = p["recurrent_scale"]
        for layer in self.layers + ([self.reservoir] if self.reservoir is not None else []) + self.readouts:
            size = net.layer_size(layer)
            for i in range(size):
                w = np.asarray(net.weights(layer, i), dtype=np.float32)
                own = size if recurrent and layer in self.layers else 0
                if layer == self.reservoir and i >= p["reservoir"]:
                    w = w * rs * math.sqrt(3.0 / len(w))
                elif own:
                    w[:-own] *= math.sqrt(3.0 / (len(w) - own))
                    w[-own:] *= rs * math.sqrt(3.0 / own)
                else:
                    w = w * math.sqrt(3.0 / len(w))
                net.set_weights(layer, i, w)
        self.metered = self.layers + ([self.reservoir] if self.reservoir is not None else [])
        self.neurons = sum(net.layer_size(l) for l in self.metered)
        self.spikes = self.ticks_seen = 0

    def act(self, state, rng, explore, meter=False):
        s = state.screen_buffer
        f = self.pool
        eye = s.reshape(120 // f, f, 160 // f, f).mean(axis=(1, 3)).ravel().astype(np.float32) / 255.0
        self.net.set_inputs("eye", eye)
        if self.sound:
            audio = np.zeros((STEP_SAMPLES, 2), dtype=np.float32)
            buf = state.audio_buffer
            if buf is not None:
                n = min(len(buf), STEP_SAMPLES)
                audio[:n] = buf[-n:] / 32768.0
        # readout "sum": the action values are the readouts summed over the
        # step's ticks, so a hidden layer that fires early in the step still
        # counts; "last" (settings saved before 2026-09-29): the last tick only.
        total = np.zeros(len(ACTIONS))
        for t in range(self.ticks):
            if self.sound:
                chunk = audio[t * self.hop:(t + 1) * self.hop]
                self.net.set_inputs("mic", np.concatenate([chunk[:, 0], chunk[:, 1]]))
            self.net.step()
            total += np.asarray(self.net.outputs())
            if meter:
                for layer in self.metered:
                    self.spikes += np.count_nonzero(np.abs(self.net.layer_output(layer)) > 1e-6)
                self.ticks_seen += 1
        self.y = y = total if self.readout == "sum" else np.asarray(self.net.outputs())
        if explore > 0 and rng.random() < explore:
            return int(rng.integers(len(ACTIONS)))
        return int(np.argmax(y))

    def learn(self, action, reward, lr, mode):
        # error: learn only while the readout's sign disagrees with the reward's
        # (as in snake), so rewards of one sign cannot push readouts to the clamp.
        if reward != 0.0 and (mode == "always" or self.y[action] * reward <= 0):
            self.net.apply_reward_to(self.readouts[action], reward, lr)


def play(game, player, p, episodes, lr, explore, rng, meter=False):
    shaper = rewards.Shaper({k: p["w_" + k] for k in ("hurt", "death", "kill", "ammo", "fire", "armor", "item",
                                                      "key", "door", "exit", "idle")},
                            p["idle_steps"], p["idle_distance"])
    buttons = np.eye(len(ACTIONS), dtype=int).tolist()
    stats = {k: [] for k in ("reward", "kills", "damage", "deaths", "items", "keys", "doors", "ammo_picked", "exits",
                             "distance", "steps")}
    parts = {}
    baseline = 0.0
    for _ in range(episodes):
        game.new_episode()
        state = game.get_state()
        shaper.reset(state)
        total, steps = 0.0, 0
        while not game.is_episode_finished():
            a = player.act(state, rng, explore, meter)
            game.make_action(buttons[a], FRAME_SKIP)
            state = None if game.is_episode_finished() else game.get_state()
            r = shaper.step(state) if state is not None else shaper.end(game)
            if lr > 0:
                # Advantage: the reward against its running mean, so a steady
                # stream of penalties (being hurt) does not teach every action alike.
                player.learn(a, r - baseline, lr, p["reward_mode"])
                baseline += p["baseline"] * (r - baseline)
            total += r
            steps += 1
        stats["reward"].append(total)
        for k in ("kills", "damage", "deaths", "items", "keys", "doors", "ammo_picked", "exits"):
            stats[k].append(shaper.counts[k])
        stats["distance"].append(shaper.distance)
        stats["steps"].append(steps)
        for k, v in shaper.totals.items():
            parts[k] = parts.get(k, 0.0) + v / episodes
    return {k: float(np.mean(v)) for k, v in stats.items()}, parts


PARAMS = {
    "scenario": ("map01", "map01 (Freedoom 2, MAP01: a whole level) or a ViZDoom scenario: defend_the_center, "
                          "health_gathering, deadly_corridor, my_way_home, ..."),
    "skill": (2, "Doom skill 1 (easiest) .. 5"),
    "episode_tics": (2100, "episode timeout in game tics (35 a second)"),
    "model": ("er", "hidden neurons: relu, er, gate or clamp"),
    "width": (256, "hidden neurons"),
    "gate": (0.2, "gate model: the fixed threshold"),
    "habituation": (False, "hidden and reservoir neurons habituate to repeated input (fade mode)"),
    "habituation_decay": (0.9, "habituation: fade factor per habituated tick"),
    "habituation_fade_after": (2, "habituation: repeats before fading starts (2 = from the 2nd tick)"),
    "habituation_tolerance": (0.05, "habituation: relative change still counted as a repeat (0 = exact repeats only, which sound and recurrence never give)"),
    "depth": (1, "hidden layers (each `width` neurons)"),
    "feedback": ("none", "feedback lines in the hidden stack: none, recurrent (each layer reads itself), topdown "
                         "(extra neurons read the layer above), both"),
    "feedback_width": (32, "topdown: extra neurons per layer that read the layer above"),
    "reservoir": (0, "echo-state reservoir after the stack: neurons reading the top layer (0: none)"),
    "reservoir_recurrent": (128, "reservoir neurons that read the whole reservoir"),
    "recurrent_scale": (0.5, "scale of recurrent weights (x sqrt(3 / layer size))"),
    "learn_hidden": (False, "the hidden layer learns from the reward too (default: a frozen random mix)"),
    "sound": (True, "hear the game: stereo audio through a two-ear cochlea"),
    "bands": (16, "cochlea bands per ear"),
    "pool": (4, "screen pooling: 4 gives 40 x 30"),
    "ticks": (3, "network ticks per game step (the sound is split across them)"),
    "normalize": (True, "weighted sums divided by the weights' length (the library default since 2026-09-28)"),
    "spontaneous_amplitude": (0.1, "E-R spontaneous firing amplitude (the library default since 2026-09-28)"),
    "readout": ("sum", "action values: sum (the readouts summed over the step's ticks) or last (the last tick only)"),
    "readout_from": ("top", "which hidden layers the readouts read: top, or all (every layer, bottom first)"),
    "rule": ("sign", "readout learning rule: sign (as in snake) or trace (traced, eligibility over recent ticks)"),
    "reward_mode": ("error", "error: learn only while the chosen readout's sign disagrees with the reward; always"),
    "baseline": (0.01, "rate of the running reward mean subtracted before learning (0: learn from the raw reward)"),
    "trace": (0.9, "trace decay per tick"),
    "lr": (0.01, "learning rate"),
    "explore": (0.3, "probability of a random action while training"),
    "train": (100, "training episodes"),
    "test": (10, "test episodes (no learning, no exploration)"),
    "w_hurt": (0.01, "penalty per health point lost"),
    "w_death": (5.0, "penalty for dying"),
    "w_kill": (1.0, "reward per kill"),
    "w_ammo": (0.02, "reward per round of ammo picked up"),
    "w_fire": (0.001, "penalty per round fired (spent)"),
    "w_armor": (0.01, "reward per armor point gained"),
    "w_item": (0.1, "reward per counted item picked up"),
    "w_key": (2.0, "reward per key"),
    "w_door": (0.5, "reward per door or lift opened (first time each)"),
    "w_exit": (10.0, "reward for leaving the level"),
    "w_idle": (0.005, "penalty per step while idle"),
    "idle_steps": (20, "idle: the window of steps (20 = 2.3 s)"),
    "idle_distance": (32.0, "idle: moved less than this many map units over the window"),
}


@nnt.experiment(
    name="doom_rl",
    description="Doom from reward at the smallest settings: 40 x 30 screen + two-ear cochlea -> relu, E-R or gate "
                "mix -> 8 action readouts; reward from hurt, death, kills, ammo, armor, items, keys, doors, exit, idling",
    tags=["er", "dynamic", "control", "learning", "multimodal"],
    params=PARAMS,
    trials=3,
)
def run(t):
    if vzd is None:
        raise RuntimeError("vizdoom is not installed: python3 -m pip install vizdoom (sound needs libopenal1)")
    exr.set_threads(1)  # one game step is far too small for OpenMP; threads only add waiting
    p = t.params
    rng = np.random.default_rng(t.seed)
    player = Player(p)
    game = make_game(p, 1000 + t.seed)
    train, _ = play(game, player, p, p["train"], p["lr"], p["explore"], rng)
    game.close()
    t.record("train_reward", train["reward"])

    game = make_game(p, 5000 + t.seed)
    test, parts = play(game, player, p, p["test"], 0.0, 0.0, rng, meter=True)
    for k, v in test.items():
        t.record(k, v)
    for k, v in parts.items():
        t.record("reward_" + k, v)
    t.record("spikes_per_step", player.spikes / max(player.ticks_seen, 1) * p["ticks"])
    t.record("active_fraction", player.spikes / max(player.ticks_seen, 1) / player.neurons)
    t.record("hidden_neurons", player.neurons)
    game.close()

    # The same network, never trained, on the same test games.
    exr.reseed(t.seed)
    game = make_game(p, 5000 + t.seed)
    control, _ = play(game, Player(p), p, p["test"], 0.0, 0.0, np.random.default_rng(t.seed))
    game.close()
    for k in ("reward", "kills", "damage", "deaths", "distance"):
        t.record(k + "_control", control[k])
