"""MDP environment for surge pricing simulation."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, List


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


class RideSharingMDP:
    """
    MDP environment for surge pricing optimization.

    Models a full day of ride-sharing operations with:
    - Hourly demand patterns (rush hours, etc.)
    - Customer memory effects (surge fatigue)
    - Stochastic demand and conversions

    Parameters
    ----------
    surge_levels : list[float]
        Available surge multiplier levels.
    base_elasticity : float
        Base price elasticity of demand.
    base_price : float
        Base price before surge.
    steps_per_hour : int
        Number of decision steps per hour.
    memory_decay : float
        How fast customers forget recent surge (0-1).
    surge_fatigue : float
        Demand reduction per unit of cumulative surge.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        surge_levels: List[float] = [1.0, 1.25, 1.5, 2.0, 2.5],
        base_elasticity: float = -0.8,
        base_price: float = 15.0,
        steps_per_hour: int = 4,
        memory_decay: float = 0.8,
        surge_fatigue: float = 0.1,
        random_seed: int = 42,
    ):
        self.surge_levels = surge_levels
        self.n_actions = len(surge_levels)
        self.base_elasticity = base_elasticity
        self.base_price = base_price
        self.steps_per_hour = steps_per_hour
        self.memory_decay = memory_decay
        self.surge_fatigue = surge_fatigue
        self.rng = np.random.RandomState(random_seed)

        # Episode length: one day
        self.max_steps = 24 * steps_per_hour

        # Demand curve by hour (normalized 0-1)
        self.hourly_demand = self._create_demand_curve()

        # Initialize state
        self.current_step = 0
        self.cumulative_surge = 0.0
        self.total_reward = 0.0
        self.history: List[dict] = []

    def _create_demand_curve(self) -> np.ndarray:
        """Create realistic hourly demand pattern."""
        demand = np.zeros(24)

        # Night (0-6): low
        demand[0:6] = 0.2

        # Morning ramp (6-7)
        demand[6:7] = 0.4

        # Morning rush (7-9)
        demand[7:10] = 0.9

        # Mid-day (10-16)
        demand[10:17] = 0.5

        # Evening rush (17-19)
        demand[17:20] = 0.95

        # Evening (20-23)
        demand[20:24] = 0.6

        # Late night spike (23-1) - bars closing
        demand[23] = 0.7
        demand[0] = 0.5

        return demand

    def reset(self) -> State:
        """Reset environment to start of day."""
        self.current_step = 0
        self.cumulative_surge = 0.0
        self.total_reward = 0.0
        self.history = []

        return self._get_state()

    def _get_state(self) -> State:
        """Get current discretized state."""
        hour = (self.current_step // self.steps_per_hour) % 24

        # Current demand level
        base_demand = self.hourly_demand[hour]
        # Apply surge fatigue
        adjusted_demand = base_demand * (1 - self.surge_fatigue * self.cumulative_surge)
        adjusted_demand = max(0.1, adjusted_demand)

        # Discretize demand
        if adjusted_demand < 0.4:
            demand_level = 0  # Low
        elif adjusted_demand < 0.7:
            demand_level = 1  # Medium
        else:
            demand_level = 2  # High

        # Discretize recent surge
        if self.cumulative_surge < 0.5:
            recent_surge = 0  # None/light
        elif self.cumulative_surge < 1.5:
            recent_surge = 1  # Moderate
        else:
            recent_surge = 2  # Heavy

        return State(hour=hour, demand_level=demand_level, recent_surge=recent_surge)

    def step(self, action: int) -> Tuple[Optional[State], float, bool, dict]:
        """
        Take an action and observe the result.

        Parameters
        ----------
        action : int
            Index of surge level to apply.

        Returns
        -------
        Tuple[State, float, bool, dict]
            (next_state, reward, done, info)
        """
        surge = self.surge_levels[action]
        hour = (self.current_step // self.steps_per_hour) % 24

        # Base demand for this hour
        base_demand = self.hourly_demand[hour]

        # Apply surge fatigue from recent history
        demand_with_fatigue = base_demand * (
            1 - self.surge_fatigue * self.cumulative_surge
        )
        demand_with_fatigue = max(0.1, demand_with_fatigue)

        # Apply elasticity effect of current surge
        elasticity = self.base_elasticity
        # Rush hours have less elastic demand
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            elasticity += 0.2  # Less negative = less elastic

        demand_multiplier = surge**elasticity
        final_demand = demand_with_fatigue * demand_multiplier

        # Expected number of rides (Poisson-like)
        expected_rides = final_demand * 10
        actual_rides = self.rng.poisson(max(0.1, expected_rides))

        # Revenue
        price = self.base_price * surge
        reward = actual_rides * price

        # Update surge memory (exponential decay)
        surge_impact = surge - 1.0
        self.cumulative_surge = self.memory_decay * self.cumulative_surge + surge_impact
        self.cumulative_surge = max(0, self.cumulative_surge)

        # Record history
        self.history.append(
            {
                "step": self.current_step,
                "hour": hour,
                "action": action,
                "surge": surge,
                "base_demand": base_demand,
                "final_demand": final_demand,
                "rides": actual_rides,
                "reward": reward,
            }
        )

        self.total_reward += reward
        self.current_step += 1

        done = self.current_step >= self.max_steps
        next_state = self._get_state() if not done else None

        info = {
            "rides": actual_rides,
            "price": price,
            "demand": final_demand,
            "cumulative_surge": self.cumulative_surge,
        }

        return next_state, reward, done, info

    @property
    def state_space_size(self) -> int:
        """Total number of possible states."""
        return 24 * 3 * 3  # hours x demand_levels x surge_levels

    def get_episode_summary(self) -> dict:
        """Get summary statistics for the episode."""
        df = pd.DataFrame(self.history)
        return {
            "total_reward": self.total_reward,
            "total_rides": int(df["rides"].sum()),
            "avg_surge": float(df["surge"].mean()),
            "avg_price": float((df["surge"] * self.base_price).mean()),
        }

    def get_history_dataframe(self) -> pd.DataFrame:
        """Get episode history as DataFrame."""
        return pd.DataFrame(self.history)


class SurgePricingEnvironment:
    """
    Simpler environment for bandit-style learning (single-step).

    Unlike RideSharingMDP, this doesn't have state transitions,
    making it suitable for bandit algorithms.

    Parameters
    ----------
    surge_levels : list[float]
        Available surge multiplier levels.
    base_elasticity : float
        Base price elasticity of demand.
    elasticity_std : float
        Standard deviation of elasticity noise.
    base_price : float
        Base price before surge.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        surge_levels: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
        base_elasticity: float = -0.8,
        elasticity_std: float = 0.2,
        base_price: float = 15.0,
        random_seed: int = 42,
    ):
        self.surge_levels = surge_levels
        self.n_arms = len(surge_levels)
        self.base_elasticity = base_elasticity
        self.elasticity_std = elasticity_std
        self.base_price = base_price
        self.rng = np.random.RandomState(random_seed)

        # Context-specific elasticity modifiers
        self.elasticity_modifiers = {
            "morning_rush": 0.2,
            "evening_rush": 0.2,
            "night": 0.1,
            "rain": 0.15,
            "weekend": -0.1,
        }

    def get_reward(self, action: int, context: dict) -> Tuple[float, dict]:
        """
        Get reward for an action in a context.

        Parameters
        ----------
        action : int
            Surge level index.
        context : dict
            Context with 'hour', 'is_weekend', 'is_raining', 'base_demand'.

        Returns
        -------
        Tuple[float, dict]
            (reward, info)
        """
        surge = self.surge_levels[action]
        elasticity = self._get_context_elasticity(context)
        base_demand = self._get_base_demand(context)

        # Demand response
        demand_multiplier = surge**elasticity
        actual_demand = base_demand * demand_multiplier

        # Conversion
        conversion = self.rng.random() < actual_demand

        # Revenue
        price = self.base_price * surge
        reward = price if conversion else 0

        info = {
            "surge": surge,
            "elasticity": elasticity,
            "base_demand": base_demand,
            "demand_multiplier": demand_multiplier,
            "actual_demand": actual_demand,
            "conversion": conversion,
            "price": price,
        }

        return reward, info

    def _get_context_elasticity(self, context: dict) -> float:
        """Get elasticity adjusted for context."""
        elasticity = self.base_elasticity
        hour = context.get("hour", 12)

        if 7 <= hour <= 9:
            elasticity += self.elasticity_modifiers["morning_rush"]
        elif 17 <= hour <= 19:
            elasticity += self.elasticity_modifiers["evening_rush"]
        elif hour >= 22 or hour <= 5:
            elasticity += self.elasticity_modifiers["night"]

        if context.get("is_raining", False):
            elasticity += self.elasticity_modifiers["rain"]

        if context.get("is_weekend", False):
            elasticity += self.elasticity_modifiers["weekend"]

        # Add noise
        elasticity += self.rng.normal(0, self.elasticity_std * 0.5)

        return min(elasticity, -0.1)

    def _get_base_demand(self, context: dict) -> float:
        """Get base demand level for context."""
        demand = context.get("base_demand", 0.5)
        hour = context.get("hour", 12)

        if 7 <= hour <= 9 or 17 <= hour <= 19:
            demand *= 1.5

        if context.get("is_raining", False):
            demand *= 1.3

        return min(demand, 1.0)

    def generate_context(self) -> dict:
        """Generate a random context."""
        return {
            "hour": self.rng.randint(0, 24),
            "is_weekend": self.rng.random() < 2 / 7,
            "is_raining": self.rng.random() < 0.2,
            "base_demand": self.rng.uniform(0.3, 0.8),
        }

    def get_optimal_action(self, context: dict) -> Tuple[int, float]:
        """Get the true optimal action for computing regret."""
        best_action = 0
        best_expected_reward = 0

        elasticity = self._get_context_elasticity(context)
        base_demand = self._get_base_demand(context)

        for action, surge in enumerate(self.surge_levels):
            demand_multiplier = surge**elasticity
            expected_demand = base_demand * demand_multiplier
            expected_reward = self.base_price * surge * expected_demand

            if expected_reward > best_expected_reward:
                best_expected_reward = expected_reward
                best_action = action

        return best_action, best_expected_reward
