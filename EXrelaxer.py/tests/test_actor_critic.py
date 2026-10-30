"""exrelaxer.actor_critic: the TD(lambda) critic, the delta-modulated actor and their step order."""
import numpy as np
import pytest

from exrelaxer.actor_critic import Actor, Agent, Critic


def chain_trial(critic, delay, size=16):
    """A one-hot tapped delay line: feature k is on k steps after the cue; reward 1 ends the trial."""
    critic.reset()
    deltas = []
    for k in range(delay + 1):
        x = np.zeros(size)
        if k < size:
            x[k] = 1.0
        terminal = k == delay
        deltas.append(critic.step(x, 1.0 if terminal else 0.0, terminal)[0])
    return deltas


def test_critic_learns_discounted_value_and_the_error_moves_to_the_cue():
    critic = Critic(16, gamma=0.9, lam=0.8, rate=0.5, bias=False)
    first = chain_trial(critic, 5)
    assert first[-1] == pytest.approx(1.0)  # nothing predicted yet
    for _ in range(300):
        last = chain_trial(critic, 5)
    assert abs(last[-1]) < 0.01
    # V right after the cue is gamma^(delay - 1): the reward is delay - 1 transitions away
    x = np.zeros(16)
    x[0] = 1.0
    assert critic.value(x) == pytest.approx(0.9 ** 4, abs=0.01)


def test_first_step_after_reset_has_no_error_and_terminal_resets():
    critic = Critic(3)
    assert critic.step(np.ones(3), 5.0)[0] == 0.0
    critic.step(np.ones(3), 1.0, terminal=True)
    assert critic.previous is None and not critic.e.any()


def test_advantage_trace_credits_only_through_the_taken_action():
    actor = Actor(4, 3, rate=0.1, decay=0.0, post="advantage", tau=1.0, rng=np.random.default_rng(1))
    x = np.array([1.0, 0.0, 2.0, 0.0])
    a, p, _ = actor.choose(x)
    actor.trace(x, a)
    post = -p
    post[a] += 1.0
    np.testing.assert_allclose(actor.E, np.outer(post, x))
    np.testing.assert_allclose(actor.E.sum(axis=0), 0.0, atol=1e-12)  # onehot - pi sums to 0
    actor.learn(1.0, x)
    assert actor.W[a] @ x > 0 > actor.W[(a + 1) % 3] @ x


def test_agent_learns_a_contextual_bandit():
    """Two states, the rewarded action depends on the state: delta drives the actor there."""
    rng = np.random.default_rng(0)
    agent = Agent(2, 2, gamma=0.0, lam=0.0, critic_rate=0.3, actor_rate=0.5, post="advantage", tau=0.2,
                  rng=np.random.default_rng(3))
    for _ in range(600):
        s = int(rng.integers(2))
        x = np.eye(2)[s]
        a = agent.step(x, 0.0)
        agent.step(np.zeros(2), 1.0 if a == s else 0.0, terminal=True)
    for s in range(2):
        assert agent.actor.choose(np.eye(2)[s], explore=False)[0] == s
