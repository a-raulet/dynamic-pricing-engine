"""Bayesian price elasticity models using PyMC."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Optional


def prepare_elasticity_data(
    df: pd.DataFrame,
    price_col: str = "surge_multiplier",
    quantity_col: str = "ride_count",
    group_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Prepare data for elasticity estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ride data.
    price_col : str
        Column representing price (or surge multiplier).
    quantity_col : str
        Column representing quantity (or ride count).
    group_cols : list, optional
        Columns to group by for aggregation.

    Returns
    -------
    pd.DataFrame
        Prepared data with log-transformed variables.
    """
    data = df.copy()

    # Log transformations
    data["log_price"] = np.log(data[price_col])
    data["log_quantity"] = np.log(data[quantity_col])

    # Handle infinities
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["log_price", "log_quantity"])

    return data


def standardize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Standardize array to zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    tuple
        Standardized array, mean, and standard deviation.
    """
    mean = x.mean()
    std = x.std()
    return (x - mean) / std, mean, std


def build_simple_elasticity_model(
    log_price: np.ndarray,
    log_quantity: np.ndarray,
    prior_elasticity_mean: float = -1.0,
    prior_elasticity_std: float = 0.5,
) -> pm.Model:
    """
    Build a simple Bayesian elasticity model.

    The model estimates: log(Q) = alpha + beta * log(P) + epsilon

    Parameters
    ----------
    log_price : np.ndarray
        Log-transformed price/surge values.
    log_quantity : np.ndarray
        Log-transformed quantity/demand values.
    prior_elasticity_mean : float
        Prior mean for elasticity (default -1 = unit elastic).
    prior_elasticity_std : float
        Prior standard deviation for elasticity.

    Returns
    -------
    pm.Model
        PyMC model ready for sampling.
    """
    # Standardize for better sampling
    log_price_std, _, price_scale = standardize(log_price)

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta_surge = pm.Normal(
            "beta_surge", mu=prior_elasticity_mean, sigma=prior_elasticity_std
        )
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value
        mu = alpha + beta_surge * log_price_std

        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=log_quantity)

        # Store scale for later conversion
        model.price_scale = price_scale

    return model


def build_segmented_elasticity_model(
    log_price: np.ndarray,
    log_quantity: np.ndarray,
    segment_indicators: dict[str, np.ndarray],
    prior_elasticity_mean: float = -1.0,
    prior_elasticity_std: float = 0.5,
) -> pm.Model:
    """
    Build a Bayesian elasticity model with segment-specific effects.

    Parameters
    ----------
    log_price : np.ndarray
        Log-transformed price/surge values.
    log_quantity : np.ndarray
        Log-transformed quantity/demand values.
    segment_indicators : dict
        Dictionary mapping segment names to binary indicator arrays.
    prior_elasticity_mean : float
        Prior mean for base elasticity.
    prior_elasticity_std : float
        Prior standard deviation for base elasticity.

    Returns
    -------
    pm.Model
        PyMC model with segment interactions.
    """
    log_price_std, _, price_scale = standardize(log_price)

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta_surge_base = pm.Normal(
            "beta_surge_base", mu=prior_elasticity_mean, sigma=prior_elasticity_std
        )
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Segment interaction effects
        effective_elasticity = beta_surge_base
        for name, indicator in segment_indicators.items():
            beta_interaction = pm.Normal(
                f"beta_surge_x_{name}", mu=0, sigma=prior_elasticity_std / 2
            )
            beta_main = pm.Normal(f"beta_{name}", mu=0, sigma=1)
            effective_elasticity = effective_elasticity + beta_interaction * indicator
            alpha = alpha + beta_main * indicator

        # Expected value
        mu = alpha + effective_elasticity * log_price_std

        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=log_quantity)

        model.price_scale = price_scale

    return model


def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    Sample from a PyMC model.

    Parameters
    ----------
    model : pm.Model
        PyMC model to sample from.
    draws : int
        Number of draws per chain.
    tune : int
        Number of tuning steps.
    chains : int
        Number of chains.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object with posterior samples.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            cores=chains,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
    return trace


def extract_elasticity(
    trace: az.InferenceData,
    price_scale: float,
    var_name: str = "beta_surge",
) -> np.ndarray:
    """
    Extract elasticity samples and convert to original scale.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples from PyMC.
    price_scale : float
        Standard deviation used to standardize log prices.
    var_name : str
        Name of the elasticity variable in the trace.

    Returns
    -------
    np.ndarray
        Elasticity samples on original scale.
    """
    samples = trace.posterior[var_name].values.flatten()
    return samples / price_scale


def summarize_elasticity(elasticity_samples: np.ndarray, hdi_prob: float = 0.94) -> dict:
    """
    Summarize elasticity posterior distribution.

    Parameters
    ----------
    elasticity_samples : np.ndarray
        Posterior samples of elasticity.
    hdi_prob : float
        Probability mass for HDI calculation.

    Returns
    -------
    dict
        Summary statistics including mean, std, and HDI.
    """
    hdi_low = (1 - hdi_prob) / 2 * 100
    hdi_high = (1 + hdi_prob) / 2 * 100

    return {
        "mean": float(elasticity_samples.mean()),
        "std": float(elasticity_samples.std()),
        "median": float(np.median(elasticity_samples)),
        "hdi_low": float(np.percentile(elasticity_samples, hdi_low)),
        "hdi_high": float(np.percentile(elasticity_samples, hdi_high)),
    }


def compute_demand_curve(
    surge_range: np.ndarray,
    elasticity_samples: np.ndarray,
    base_quantity: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute demand curve with uncertainty from elasticity posterior.

    Parameters
    ----------
    surge_range : np.ndarray
        Range of surge multipliers to evaluate.
    elasticity_samples : np.ndarray
        Posterior samples of elasticity.
    base_quantity : float
        Base quantity at surge = 1.0.

    Returns
    -------
    tuple
        Mean demand curve, lower bound (2.5%), upper bound (97.5%).
    """
    # Compute demand for each elasticity sample
    demand_curves = np.outer(surge_range, np.zeros(len(elasticity_samples)))
    for i, e in enumerate(elasticity_samples):
        demand_curves[:, i] = base_quantity * (surge_range ** e)

    mean_demand = demand_curves.mean(axis=1)
    lower_demand = np.percentile(demand_curves, 2.5, axis=1)
    upper_demand = np.percentile(demand_curves, 97.5, axis=1)

    return mean_demand, lower_demand, upper_demand


def compute_revenue_curve(
    surge_range: np.ndarray,
    elasticity_samples: np.ndarray,
    base_quantity: float = 100.0,
) -> tuple[np.ndarray, float]:
    """
    Compute revenue curve and find optimal surge.

    Parameters
    ----------
    surge_range : np.ndarray
        Range of surge multipliers to evaluate.
    elasticity_samples : np.ndarray
        Posterior samples of elasticity.
    base_quantity : float
        Base quantity at surge = 1.0.

    Returns
    -------
    tuple
        Mean revenue curve, optimal surge multiplier.
    """
    mean_elasticity = elasticity_samples.mean()
    demand = base_quantity * (surge_range ** mean_elasticity)
    revenue = surge_range * demand

    optimal_idx = np.argmax(revenue)
    optimal_surge = surge_range[optimal_idx]

    return revenue, optimal_surge
