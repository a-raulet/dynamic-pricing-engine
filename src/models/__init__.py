"""Pricing models: elasticity, bandits, and reinforcement learning."""

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
    QLearningAgent,
    SARSAAgent,
    train_q_learning,
    evaluate_policy,
)

# Lazy import for elasticity (requires PyMC)
def __getattr__(name):
    """Lazy import for PyMC-dependent modules."""
    elasticity_exports = {
        "prepare_elasticity_data",
        "build_simple_elasticity_model",
        "build_segmented_elasticity_model",
        "sample_model",
        "extract_elasticity",
        "summarize_elasticity",
        "compute_demand_curve",
        "compute_revenue_curve",
    }
    if name in elasticity_exports:
        from . import elasticity
        return getattr(elasticity, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Bandits
    "Context",
    "BanditAlgorithm",
    "EpsilonGreedy",
    "UCB",
    "ThompsonSampling",
    "ContextualThompsonSampling",
    "run_bandit_experiment",
    # RL
    "QLearningAgent",
    "SARSAAgent",
    "train_q_learning",
    "evaluate_policy",
    # Elasticity (lazy loaded)
    "prepare_elasticity_data",
    "build_simple_elasticity_model",
    "build_segmented_elasticity_model",
    "sample_model",
    "extract_elasticity",
    "summarize_elasticity",
    "compute_demand_curve",
    "compute_revenue_curve",
]
