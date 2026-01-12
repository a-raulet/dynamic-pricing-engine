"""Data loading utilities for Uber/Lyft cab prices dataset."""

from pathlib import Path

import pandas as pd

# Default paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_cab_rides(path: Path | None = None) -> pd.DataFrame:
    """
    Load the cab rides dataset.

    Parameters
    ----------
    path : Path, optional
        Path to the cab_rides.csv file. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Raw cab rides data with columns: distance, cab_type, time_stamp,
        destination, source, price, surge_multiplier, id, product_id, name.
    """
    if path is None:
        path = RAW_DIR / "cab_rides.csv"

    df = pd.read_csv(path)

    # Convert timestamp to datetime
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="ms")

    return df


def load_weather(path: Path | None = None) -> pd.DataFrame:
    """
    Load the weather dataset.

    Parameters
    ----------
    path : Path, optional
        Path to the weather.csv file. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Weather data with columns: temp, location, clouds, pressure,
        rain, time_stamp, humidity, wind.
    """
    if path is None:
        path = RAW_DIR / "weather.csv"

    df = pd.read_csv(path)

    # Convert timestamp to datetime
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="s")

    return df


def load_merged_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the preprocessed merged dataset.

    Parameters
    ----------
    path : Path, optional
        Path to the merged dataset. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Merged and preprocessed data ready for analysis.
    """
    if path is None:
        path = PROCESSED_DIR / "rides_with_weather.parquet"

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)
