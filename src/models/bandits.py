"""Multi-armed bandit algorithms for dynamic pricing."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol
from abc import ABC, abstractmethod


@dataclass
class Context:
    """Represents the state of the world when making a pricing decision."""

    hour: int  # 0-23
    is_weekend: bool
    is_raining: bool
    base_demand: float  # Underlying demand level (0-1)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for contextual bandit."""
        is_morning_rush = 7 <= self.hour <= 9
        is_evening_rush = 17 <= self.hour <= 19
        is_night = self.hour >= 22 or self.hour <= 5

        return np.array(
            [
                float(is_morning_rush),
                float(is_evening_rush),
                float(is_night),
                float(self.is_weekend),
                float(self.is_raining),
                self.base_demand,
            ]
        )

    @property
    def description(self) -> str:
        time_of_day = (
            "night"
            if (self.hour >= 22 or self.hour <= 5)
            else "morning rush"
            if 7 <= self.hour <= 9
            else "evening rush"
            if 17 <= self.hour <= 19
            else "normal"
        )
        weather = "rainy" if self.is_raining else "clear"
        day = "weekend" if self.is_weekend else "weekday"
        return f"{time_of_day}, {day}, {weather}"


class BanditAlgorithm(ABC):
    """Base class for bandit algorithms."""

    def __init__(self, n_arms: int, random_seed: int = 42):
        self.n_arms = n_arms
        self.rng = np.random.RandomState(random_seed)
        self.reset()

    def reset(self):
        """Reset the algorithm state."""
        self.counts = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.t = 0

    @abstractmethod
    def select_action(self, context: Optional[Context] = None) -> int:
        """Select an action."""
        pass

    def update(self, action: int, reward: float, context: Optional[Context] = None):
        """Update algorithm state after observing a reward."""
        self.counts[action] += 1
        self.rewards[action] += reward
        self.t += 1

    @property
    def mean_rewards(self) -> np.ndarray:
        """Get mean reward for each arm."""
        with np.errstate(divide="ignore", invalid="ignore"):
            means = self.rewards / self.counts
            means[~np.isfinite(means)] = 0
        return means


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy algorithm.

    With probability epsilon, explore (random action).
    With probability 1-epsilon, exploit (best known action).
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        random_seed: int = 42,
    ):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = epsilon
        super().__init__(n_arms, random_seed)

    def reset(self):
        super().reset()
        self.epsilon = self.initial_epsilon

    def select_action(self, context: Optional[Context] = None) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.n_arms)
        else:
            means = self.mean_rewards
            max_mean = means.max()
            best_arms = np.where(means == max_mean)[0]
            return self.rng.choice(best_arms)

    def update(self, action: int, reward: float, context: Optional[Context] = None):
        super().update(action, reward, context)
        self.epsilon *= self.epsilon_decay


class UCB(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB1) algorithm.

    Selects the arm with highest upper confidence bound:
    UCB = mean_reward + c * sqrt(log(t) / n_i)
    """

    def __init__(self, n_arms: int, c: float = 2.0, random_seed: int = 42):
        self.c = c
        super().__init__(n_arms, random_seed)

    def select_action(self, context: Optional[Context] = None) -> int:
        if self.t < self.n_arms:
            return self.t

        means = self.mean_rewards
        confidence = self.c * np.sqrt(np.log(self.t + 1) / (self.counts + 1e-8))
        ucb_values = means + confidence

        return np.argmax(ucb_values)


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling with Gaussian rewards.

    Maintains a posterior distribution for each arm's mean reward.
    Samples from posteriors and selects the arm with highest sample.
    """

    def __init__(
        self,
        n_arms: int,
        prior_mean: float = 10.0,
        prior_std: float = 5.0,
        reward_std: float = 10.0,
        random_seed: int = 42,
    ):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.reward_std = reward_std
        super().__init__(n_arms, random_seed)

    def reset(self):
        super().reset()
        self.posterior_means = np.full(self.n_arms, self.prior_mean)
        self.posterior_vars = np.full(self.n_arms, self.prior_std**2)

    def select_action(self, context: Optional[Context] = None) -> int:
        samples = self.rng.normal(self.posterior_means, np.sqrt(self.posterior_vars))
        return np.argmax(samples)

    def update(self, action: int, reward: float, context: Optional[Context] = None):
        super().update(action, reward, context)

        prior_precision = 1 / (self.prior_std**2)
        reward_precision = 1 / (self.reward_std**2)

        n = self.counts[action]
        posterior_precision = prior_precision + n * reward_precision
        self.posterior_vars[action] = 1 / posterior_precision

        mean_reward = self.rewards[action] / n if n > 0 else 0
        self.posterior_means[action] = self.posterior_vars[action] * (
            prior_precision * self.prior_mean + n * reward_precision * mean_reward
        )


class ContextualThompsonSampling(BanditAlgorithm):
    """
    Contextual Thompson Sampling using linear reward model.

    Assumes reward = context.T @ theta + noise, where theta is arm-specific.
    Uses Bayesian linear regression for each arm.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        prior_std: float = 1.0,
        reward_std: float = 10.0,
        random_seed: int = 42,
    ):
        self.context_dim = context_dim
        self.prior_std = prior_std
        self.reward_std = reward_std
        super().__init__(n_arms, random_seed)

    def reset(self):
        super().reset()
        self.prior_cov = (self.prior_std**2) * np.eye(self.context_dim)
        self.prior_precision = np.linalg.inv(self.prior_cov)

        self.posterior_means = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
        self.posterior_covs = [self.prior_cov.copy() for _ in range(self.n_arms)]

        self.XtX = [
            np.zeros((self.context_dim, self.context_dim)) for _ in range(self.n_arms)
        ]
        self.Xty = [np.zeros(self.context_dim) for _ in range(self.n_arms)]

    def select_action(self, context: Context) -> int:
        x = context.to_vector()

        sampled_rewards = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta_sample = self.rng.multivariate_normal(
                self.posterior_means[a], self.posterior_covs[a]
            )
            sampled_rewards[a] = x @ theta_sample

        return np.argmax(sampled_rewards)

    def update(self, action: int, reward: float, context: Context):
        super().update(action, reward, context)
        x = context.to_vector()

        self.XtX[action] += np.outer(x, x)
        self.Xty[action] += reward * x

        reward_precision = 1 / (self.reward_std**2)
        posterior_precision = self.prior_precision + reward_precision * self.XtX[action]
        self.posterior_covs[action] = np.linalg.inv(posterior_precision)
        self.posterior_means[action] = self.posterior_covs[action] @ (
            reward_precision * self.Xty[action]
        )


def run_bandit_experiment(
    env,
    algorithms: dict[str, BanditAlgorithm],
    n_rounds: int = 5000,
    random_seed: int = 42,
) -> dict:
    """
    Run bandit experiment and track metrics.

    Parameters
    ----------
    env : SurgePricingEnvironment
        Environment to run the experiment in.
    algorithms : dict
        Dictionary mapping algorithm names to BanditAlgorithm instances.
    n_rounds : int
        Number of rounds to run.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results with cumulative rewards, regrets, and actions for each algorithm.
    """
    np.random.seed(random_seed)

    results = {
        name: {
            "rewards": [],
            "regrets": [],
            "actions": [],
        }
        for name in algorithms
    }

    contexts = [env.generate_context() for _ in range(n_rounds)]

    for name, algo in algorithms.items():
        algo.reset()
        cumulative_reward = 0
        cumulative_regret = 0

        for t, ctx in enumerate(contexts):
            if hasattr(algo, "context_dim"):
                action = algo.select_action(ctx)
            else:
                action = algo.select_action()

            reward, info = env.get_reward(action, ctx)

            opt_action, opt_expected_reward = env.get_optimal_action(ctx)
            instant_regret = opt_expected_reward - (
                env.base_price * env.surge_levels[action] * info["actual_demand"]
            )

            if hasattr(algo, "context_dim"):
                algo.update(action, reward, ctx)
            else:
                algo.update(action, reward)

            cumulative_reward += reward
            cumulative_regret += max(0, instant_regret)

            results[name]["rewards"].append(cumulative_reward)
            results[name]["regrets"].append(cumulative_regret)
            results[name]["actions"].append(action)

    return results
