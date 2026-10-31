"""Actor-critic with local three-factor learning over frozen features.

A TD(lambda) critic predicts the discounted future reward V(s) as a linear
function of a feature vector (a frozen layer's outputs, its thresholds, ...).
Its prediction error

    delta_t = r_t + gamma * V(s_t) - V(s_{t-1})      (V(s_t) = 0 at a terminal step)

is the learning signal, the analogue of phasic dopamine: one global scalar,
not sparse, positive when things turn out better than expected. The critic
and the actor learn from it with eligibility traces, so every update is
local (pre x post x delta):

    critic:  e_i  <- gamma * lambda * e_i + x_i            w_i  += a_c * delta * e_i
    actor:   E_ki <- decay * E_ki + post_k * x_i           W_ki += a_a * delta * E_ki

The actor has one linear readout per action. post is 1 for the action taken
and 0 for the others (post="taken"), or the softmax policy gradient
onehot - pi (post="advantage"), so only the readout of the chosen action is
credited. Actions come from a softmax over the readouts (temperature tau)
or epsilon-greedy. An optional entropy bonus (entropy > 0) keeps the policy
from collapsing before the critic has learned anything.

Both learning rates are divided by 1 + a slowly decaying peak of |x|^2, so
they do not depend on how many features there are or how large they are. The actor
weights stay within +-clamp, as the library's.

Order of one step (Agent.step):
  1. the caller has run the network forward and read the features x_t;
  2. delta_t from the critic (V(s_t) read with the weights of step t-1);
  3. critic and actor learn from delta_t through the traces of step t-1;
  4. the traces take in the activity of step t;
  5. an action is chosen (the actor's trace takes in the choice).
At an episode boundary the traces and V(s_{t-1}) are forgotten (reset());
the network's own state is the caller's business.
"""
import numpy as np


def feature_scale(norm, x, bias, decay=0.99):
    """A slowly decaying peak of |x|^2: a burst of activity (a cue after silence) cannot take a
    step many times too large, as a running mean would let it."""
    sq = float(x @ x) + (1.0 if bias else 0.0)
    return sq if norm is None else max(sq, decay * norm)


class Critic:
    """Linear TD(lambda) value predictor V(x) = w . x + b."""

    def __init__(self, features, gamma=0.97, lam=0.9, rate=0.05, bias=True, normalize=True, clip=0.0):
        self.gamma, self.lam, self.rate = float(gamma), float(lam), float(rate)
        self.bias, self.normalize, self.clip = bool(bias), bool(normalize), float(clip)
        self.w = np.zeros(features)
        self.b = 0.0
        self.e = np.zeros(features)
        self.eb = 0.0
        self.norm = None   # running mean of |x|^2
        self.previous = None  # V(s_{t-1}), None after reset()

    def value(self, x):
        return float(self.w @ x + (self.b if self.bias else 0.0))

    def step_size(self, x):
        self.norm = feature_scale(self.norm, x, self.bias)
        return self.rate / (1.0 + self.norm) if self.normalize else self.rate

    def step(self, x, reward, terminal=False, learn=True):
        """One TD step on reaching features x with `reward`; returns (delta, V(x)).
        delta is 0 on the first step after reset()."""
        v = 0.0 if terminal else self.value(x)
        alpha = self.step_size(x)
        delta = 0.0
        if self.previous is not None:
            delta = reward + self.gamma * v - self.previous
            if learn:
                d = float(np.clip(delta, -self.clip, self.clip)) if self.clip > 0 else delta
                self.w += alpha * d * self.e
                if self.bias:
                    self.b += alpha * d * self.eb
        if terminal:
            self.reset()
        else:
            self.e *= self.gamma * self.lam
            self.e += x
            self.eb = self.gamma * self.lam * self.eb + 1.0
            self.previous = v
        return delta, v

    def reset(self):
        self.e[:] = 0.0
        self.eb = 0.0
        self.previous = None


class Actor:
    """One linear readout per action, learned by delta x eligibility (pre x post)."""

    def __init__(self, features, actions, rate=0.005, decay=0.87, post="taken", select="softmax", tau=0.1,
                 epsilon=0.05, bias=True, normalize=True, clamp=10.0, entropy=0.0, rng=None):
        if post not in ("taken", "advantage"):
            raise ValueError("post must be taken or advantage")
        if select not in ("softmax", "greedy"):
            raise ValueError("select must be softmax or greedy (epsilon-greedy)")
        self.rate, self.decay, self.post, self.select = float(rate), float(decay), post, select
        self.tau, self.epsilon, self.bias, self.normalize, self.clamp = float(tau), float(epsilon), bias, normalize, clamp
        self.entropy = float(entropy)
        self.W = np.zeros((actions, features))
        self.b = np.zeros(actions)
        self.E = np.zeros((actions, features))
        self.Eb = np.zeros(actions)
        self.norm = None
        self.rng = rng if rng is not None else np.random.default_rng()
        self.x = None
        self.probs = None
        self.y = None

    def scores(self, x):
        self.x = x
        self.y = self.W @ x + (self.b if self.bias else 0.0)
        return self.y

    def policy(self, y):
        if self.select == "softmax":
            z = (y - y.max()) / max(self.tau, 1e-9)
            p = np.exp(z)
            return p / p.sum()
        p = np.full(len(y), self.epsilon / len(y))
        p[int(np.argmax(y))] += 1.0 - self.epsilon
        return p

    def choose(self, x, explore=True):
        """Scores x, picks an action (the argmax when not exploring); returns (action, probs, greedy)."""
        y = self.scores(x)
        greedy = int(np.argmax(y))
        self.probs = p = self.policy(y)
        a = int(self.rng.choice(len(y), p=p)) if explore else greedy
        return a, p, greedy

    def learn(self, delta, x):
        """Moves the weights by delta along the traces of the previous step(s)."""
        self.norm = feature_scale(self.norm, x, self.bias)
        alpha = self.rate / (1.0 + self.norm) if self.normalize else self.rate
        if self.entropy > 0 and self.probs is not None and self.x is not None:
            # Entropy bonus: each readout moves along d H / d y_k = -pi_k (log pi_k + H) / tau
            # times its own input (local: pre x a post factor from the policy).
            p = np.clip(self.probs, 1e-12, 1.0)
            h = float(-(p * np.log(p)).sum())
            g = -p * (np.log(p) + h)
            self.W += alpha * self.entropy * np.outer(g, self.x)
            if self.bias:
                self.b += alpha * self.entropy * g
        if delta == 0.0:
            np.clip(self.W, -self.clamp, self.clamp, out=self.W)
            return
        self.W += alpha * delta * self.E
        np.clip(self.W, -self.clamp, self.clamp, out=self.W)
        if self.bias:
            self.b += alpha * delta * self.Eb
            np.clip(self.b, -self.clamp, self.clamp, out=self.b)

    def trace(self, x, action):
        """The traces take in this step: post (taken action, or onehot - pi) x pre."""
        post = -self.probs.copy() if self.post == "advantage" else np.zeros(len(self.b))
        post[action] += 1.0
        self.E *= self.decay
        self.E += np.outer(post, x)
        self.Eb = self.decay * self.Eb + post

    def reset(self):
        self.E[:] = 0.0
        self.Eb[:] = 0.0

    def clamped(self):
        return float(np.mean(np.abs(self.W) >= self.clamp - 1e-6))


class Agent:
    """A critic and an actor over one feature vector, with the step order of the module docstring.

    modulator: "td" (delta, the default) or "reward" (the raw reward, as a control: the critic
    still learns and is logged, but the actor follows r_t).
    """

    def __init__(self, features, actions, gamma=0.97, lam=0.9, critic_rate=0.05, actor_rate=0.005,
                 actor_decay=None, clip=1.0, modulator="td", post="taken", select="softmax", tau=0.1,
                 epsilon=0.05, normalize=True, entropy=0.0, center=0.0, rng=None):
        if modulator not in ("td", "reward"):
            raise ValueError("modulator must be td or reward")
        self.critic = Critic(features, gamma, lam, critic_rate, normalize=normalize)
        self.actor = Actor(features, actions, actor_rate, gamma * lam if actor_decay is None else actor_decay,
                           post, select, tau, epsilon, normalize=normalize, entropy=entropy, rng=rng)
        self.clip, self.modulator = float(clip), modulator
        # center > 0: each feature minus its running mean (that rate per step), so constant
        # offsets (E-R thresholds at rest) neither inflate |x|^2 nor slow learning; the biases
        # carry the mean.
        self.center, self.mean = float(center), None
        self.log = EpisodeLog()

    def step(self, x, reward, terminal=False, learn=True, explore=True):
        """Features x of the state reached, the reward the last action earned. Returns the next
        action (None at a terminal step)."""
        x = np.asarray(x, dtype=np.float64)
        if self.center > 0:
            if self.mean is None:
                self.mean = x.copy()
            elif learn:
                self.mean += self.center * (x - self.mean)
            x = x - self.mean
        delta, v = self.critic.step(x, reward, terminal, learn)
        signal = delta if self.modulator == "td" else reward
        if self.clip > 0:
            signal = float(np.clip(signal, -self.clip, self.clip))
        if learn:
            self.actor.learn(signal, x)
        self.log.add(delta, v, reward)
        if terminal:
            self.actor.reset()
            return None
        a, p, greedy = self.actor.choose(x, explore)
        if learn:
            self.actor.trace(x, a)
        self.log.choice(a, p, greedy)
        return a

    def end_episode(self):
        """An episode cut short without a terminal step (timeout): forget the traces."""
        self.critic.reset()
        self.actor.reset()


class EpisodeLog:
    """Per-episode diagnostics: return, length, mean |delta|, V trace, action entropy, off-greedy choices."""

    def __init__(self):
        self.new()

    def new(self):
        self.deltas, self.values, self.rewards, self.entropy, self.off_greedy = [], [], [], [], 0
        self.actions = []

    def add(self, delta, v, reward):
        self.deltas.append(delta)
        self.values.append(v)
        self.rewards.append(reward)

    def choice(self, a, p, greedy):
        q = p[p > 0]
        self.entropy.append(float(-(q * np.log(q)).sum()))
        self.off_greedy += int(a != greedy)
        self.actions.append(a)

    def summary(self):
        d = np.asarray(self.deltas[1:]) if len(self.deltas) > 1 else np.zeros(1)
        n = max(1, len(self.actions))
        return {
            "return": float(np.sum(self.rewards)),
            "length": len(self.actions),
            "mean_abs_delta": float(np.mean(np.abs(d))),
            "mean_delta": float(np.mean(d)),
            "mean_value": float(np.mean(self.values)) if self.values else 0.0,
            "entropy": float(np.mean(self.entropy)) if self.entropy else 0.0,
            "off_greedy": self.off_greedy / n,
        }
