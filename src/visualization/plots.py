"""Publication-quality visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional

# Color palette inspired by Uber/Lyft branding
UBER_BLACK = "#000000"
UBER_GREEN = "#276EF1"  # Uber's blue actually
LYFT_PINK = "#FF00BF"
SURGE_COLORS = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#9B59B6"]

# Custom color palette for this project
PALETTE = {
    "uber": "#1F1F1F",
    "lyft": "#EA0B8C",
    "primary": "#2C3E50",
    "secondary": "#7F8C8D",
    "accent": "#3498DB",
    "warning": "#E74C3C",
    "success": "#27AE60",
}


def set_publication_style():
    """
    Set matplotlib style for publication-quality figures.

    Call this at the start of each notebook to ensure consistent styling.
    """
    # Use a clean style as base
    plt.style.use("seaborn-v0_8-whitegrid")

    # Custom parameters
    plt.rcParams.update({
        # Figure
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",

        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })

    # Set seaborn defaults
    sns.set_palette([PALETTE["primary"], PALETTE["accent"], PALETTE["warning"]])


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Optional[Path] = None,
    formats: list[str] = ["png", "pdf"],
):
    """
    Save figure to multiple formats.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    name : str
        Base name for the file (without extension).
    output_dir : Path, optional
        Directory to save to. Defaults to outputs/figures/.
    formats : list of str
        File formats to save (png, pdf, svg).
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches="tight")
        print(f"Saved: {filepath}")


def plot_surge_distribution(
    df,
    surge_col: str = "surge_multiplier",
    hue_col: Optional[str] = None,
    title: str = "Distribution of Surge Multipliers",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the distribution of surge multipliers.

    Parameters
    ----------
    df : pd.DataFrame
        Data with surge multiplier column.
    surge_col : str
        Name of the surge multiplier column.
    hue_col : str, optional
        Column to use for color grouping (e.g., 'cab_type').
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if hue_col:
        for i, (name, group) in enumerate(df.groupby(hue_col)):
            color = PALETTE["uber"] if "uber" in name.lower() else PALETTE["lyft"]
            ax.hist(
                group[surge_col],
                bins=20,
                alpha=0.6,
                label=name,
                color=color,
                edgecolor="white",
            )
    else:
        ax.hist(
            df[surge_col],
            bins=20,
            alpha=0.7,
            color=PALETTE["primary"],
            edgecolor="white",
        )

    ax.set_xlabel("Surge Multiplier")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.axvline(x=1.0, color=PALETTE["success"], linestyle="--", label="No surge")

    if hue_col:
        ax.legend()

    return ax


def plot_hourly_pattern(
    df,
    value_col: str,
    hour_col: str = "hour",
    hue_col: Optional[str] = None,
    agg: str = "mean",
    title: str = "Hourly Pattern",
    ylabel: str = "Value",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot hourly patterns with optional grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Data with hour column.
    value_col : str
        Column to aggregate.
    hour_col : str
        Name of the hour column.
    hue_col : str, optional
        Column for color grouping.
    agg : str
        Aggregation function ('mean', 'median', 'sum', 'count').
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    if hue_col:
        grouped = df.groupby([hour_col, hue_col])[value_col].agg(agg).unstack()
        for col in grouped.columns:
            color = PALETTE["uber"] if "uber" in col.lower() else PALETTE["lyft"]
            ax.plot(grouped.index, grouped[col], marker="o", label=col, color=color)
    else:
        hourly = df.groupby(hour_col)[value_col].agg(agg)
        ax.plot(hourly.index, hourly.values, marker="o", color=PALETTE["primary"])

    # Highlight rush hours
    ax.axvspan(7, 9, alpha=0.1, color=PALETTE["warning"], label="Morning rush")
    ax.axvspan(17, 19, alpha=0.1, color=PALETTE["warning"], label="Evening rush")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 2))

    if hue_col:
        ax.legend(loc="upper right")

    return ax


def plot_price_vs_surge(
    df,
    price_col: str = "price",
    surge_col: str = "surge_multiplier",
    hue_col: Optional[str] = None,
    sample_size: int = 5000,
    title: str = "Price vs Surge Multiplier",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Scatter plot of price vs surge multiplier.

    Parameters
    ----------
    df : pd.DataFrame
        Data with price and surge columns.
    price_col : str
        Name of price column.
    surge_col : str
        Name of surge multiplier column.
    hue_col : str, optional
        Column for color grouping.
    sample_size : int
        Number of points to sample for visibility.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Sample for visibility
    if len(df) > sample_size:
        plot_df = df.sample(sample_size, random_state=42)
    else:
        plot_df = df

    if hue_col:
        for name, group in plot_df.groupby(hue_col):
            color = PALETTE["uber"] if "uber" in name.lower() else PALETTE["lyft"]
            ax.scatter(
                group[surge_col],
                group[price_col],
                alpha=0.3,
                s=20,
                label=name,
                color=color,
            )
        ax.legend()
    else:
        ax.scatter(
            plot_df[surge_col],
            plot_df[price_col],
            alpha=0.3,
            s=20,
            color=PALETTE["primary"],
        )

    ax.set_xlabel("Surge Multiplier")
    ax.set_ylabel("Price ($)")
    ax.set_title(title)

    return ax


def plot_weather_impact(
    df,
    weather_col: str,
    value_col: str = "surge_multiplier",
    bins: int = 10,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the impact of a weather variable on surge/price.

    Parameters
    ----------
    df : pd.DataFrame
        Data with weather and value columns.
    weather_col : str
        Weather column to analyze (temp, rain, humidity, etc.).
    value_col : str
        Value column to show impact on.
    bins : int
        Number of bins for the weather variable.
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Create bins
    df = df.copy()
    df["weather_bin"] = pd.cut(df[weather_col], bins=bins)

    # Calculate mean and std for each bin
    stats = df.groupby("weather_bin")[value_col].agg(["mean", "std", "count"])
    stats = stats[stats["count"] >= 10]  # Require minimum samples

    # Plot
    x = range(len(stats))
    ax.bar(x, stats["mean"], yerr=stats["std"], capsize=3, color=PALETTE["accent"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in stats.index], rotation=45, ha="right")

    ax.set_xlabel(weather_col.replace("_", " ").title())
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_title(title or f"Impact of {weather_col} on {value_col}")

    return ax


# Import pandas here to avoid circular imports in type hints above
import pandas as pd
