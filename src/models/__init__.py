"""Pricing models: elasticity, bandits, and reinforcement learning."""

from .elasticity import (
    prepare_elasticity_data,
    build_simple_elasticity_model,
    build_segmented_elasticity_model,
    sample_model,
    extract_elasticity,
    summarize_elasticity,
    compute_demand_curve,
    compute_revenue_curve,
)

from .bandits import (
    Context,
    BanditAlgorithm,
    EpsilonGreedy,
    UCB,
    ThompsonSampling,
    ContextualThompsonSampling,
    run_bandit_experiment,
)

from .rl_agent import (
    State,
    QLearningAgent,
    SARSAAgent,
    train_q_learning,
    evaluate_policy,
)

__all__ = [
    # Elasticity
    "prepare_elasticity_data",
    "build_simple_elasticity_model",
    "build_segmented_elasticity_model",
    "sample_model",
    "extract_elasticity",
    "summarize_elasticity",
    "compute_demand_curve",
    "compute_revenue_curve",
    # Bandits
    "Context",
    "BanditAlgorithm",
    "EpsilonGreedy",
    "UCB",
    "ThompsonSampling",
    "ContextualThompsonSampling",
    "run_bandit_experiment",
    # RL
    "State",
    "QLearningAgent",
    "SARSAAgent",
    "train_q_learning",
    "evaluate_policy",
]
