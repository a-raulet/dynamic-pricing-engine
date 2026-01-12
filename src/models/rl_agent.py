"""Reinforcement Learning agents for dynamic pricing."""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class State:
    """MDP state for surge pricing."""

    hour: int  # 0-23
    demand_level: int  # Discretized: 0=low, 1=medium, 2=high
    recent_surge: int  # Discretized recent surge: 0=none, 1=moderate, 2=heavy

    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to hashable tuple for Q-table."""
        return (self.hour, self.demand_level, self.recent_surge)

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int]) -> "State":
        return cls(hour=t[0], demand_level=t[1], recent_surge=t[2])

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()


class QLearningAgent:
    """
    Tabular Q-learning agent for surge pricing.

    Parameters
    ----------
    n_actions : int
        Number of possible actions (surge levels).
    learning_rate : float
        Learning rate (alpha) for Q-value updates.
    discount_factor : float
        Discount factor (gamma) for future rewards.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Minimum exploration rate.
    epsilon_decay : float
        Decay rate for epsilon per episode.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        random_seed: int = 42,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.RandomState(random_seed)

        # Q-table: state -> action values
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )

        # Statistics
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def reset(self):
        """Reset agent to initial state."""
        self.epsilon = self.epsilon_start
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state: State, greedy: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Parameters
        ----------
        state : State
            Current state.
        greedy : bool
            If True, always select greedy action (no exploration).

        Returns
        -------
        int
            Selected action index.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randint(self.n_actions)
        else:
            q_values = self.q_table[state.to_tuple()]
            max_q = q_values.max()
            best_actions = np.where(q_values == max_q)[0]
            return self.rng.choice(best_actions)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: Optional[State],
        done: bool,
    ):
        """
        Update Q-values using the Q-learning update rule.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        Parameters
        ----------
        state : State
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : State or None
            Next state (None if terminal).
        done : bool
            Whether episode is done.
        """
        state_key = state.to_tuple()

        if done:
            target = reward
        else:
            next_state_key = next_state.to_tuple()
            target = reward + self.gamma * self.q_table[next_state_key].max()

        # Q-learning update
        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> Dict[Tuple, int]:
        """
        Extract greedy policy from Q-table.

        Returns
        -------
        dict
            Mapping from state tuples to optimal actions.
        """
        policy = {}
        for state_key, q_values in self.q_table.items():
            policy[state_key] = int(np.argmax(q_values))
        return policy

    def get_q_values(self, state: State) -> np.ndarray:
        """Get Q-values for a state."""
        return self.q_table[state.to_tuple()].copy()


class SARSAAgent(QLearningAgent):
    """
    SARSA agent (on-policy TD control).

    Unlike Q-learning, SARSA uses the actual next action for updates,
    making it on-policy.
    """

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: Optional[State],
        next_action: Optional[int],
        done: bool,
    ):
        """
        Update Q-values using SARSA update rule.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
        """
        state_key = state.to_tuple()

        if done:
            target = reward
        else:
            next_state_key = next_state.to_tuple()
            target = reward + self.gamma * self.q_table[next_state_key][next_action]

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error


def train_q_learning(
    env,
    agent: QLearningAgent,
    n_episodes: int = 1000,
    verbose: bool = True,
    log_interval: int = 100,
) -> list[float]:
    """
    Train Q-learning agent on environment.

    Parameters
    ----------
    env : RideSharingMDP
        Environment to train on.
    agent : QLearningAgent
        Agent to train.
    n_episodes : int
        Number of training episodes.
    verbose : bool
        Whether to print progress.
    log_interval : int
        How often to print progress.

    Returns
    -------
    list[float]
        Episode rewards.
    """
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(
                f"Episode {episode + 1}: "
                f"Avg Reward = ${avg_reward:,.0f}, "
                f"Epsilon = {agent.epsilon:.3f}"
            )

    return episode_rewards


def evaluate_policy(
    env,
    policy_fn,
    n_episodes: int = 100,
    seed: int = 42,
) -> list[float]:
    """
    Evaluate a policy over multiple episodes.

    Parameters
    ----------
    env : RideSharingMDP
        Environment to evaluate on.
    policy_fn : callable
        Function mapping (state, env) to action.
    n_episodes : int
        Number of evaluation episodes.
    seed : int
        Random seed.

    Returns
    -------
    list[float]
        Episode rewards.
    """
    env.rng = np.random.RandomState(seed)
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = policy_fn(state, env)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return rewards
