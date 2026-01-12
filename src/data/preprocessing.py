"""Data preprocessing and feature engineering for surge pricing analysis."""

import pandas as pd
import numpy as np


def preprocess_rides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw cab rides data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw cab rides dataframe from load_cab_rides().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with:
        - Missing prices removed
        - Surge multiplier filled (1.0 for Lyft if missing)
        - Datetime properly parsed
    """
    df = df.copy()

    # Remove rows with missing prices (these are failed ride requests)
    df = df.dropna(subset=["price"])

    # Fill missing surge multiplier with 1.0 (no surge)
    if "surge_multiplier" in df.columns:
        df["surge_multiplier"] = df["surge_multiplier"].fillna(1.0)

    # Ensure time_stamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time_stamp"]):
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="ms")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for surge pricing analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed rides dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional features:
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - is_weekend: Boolean for Saturday/Sunday
        - is_rush_hour: Boolean for morning (7-9) or evening (17-19) rush
        - time_of_day: Categorical (night, morning, afternoon, evening)
        - price_per_mile: Price divided by distance
    """
    df = df.copy()

    # Time-based features
    df["hour"] = df["time_stamp"].dt.hour
    df["day_of_week"] = df["time_stamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # Rush hour: 7-9 AM and 5-7 PM on weekdays
    morning_rush = (df["hour"] >= 7) & (df["hour"] <= 9)
    evening_rush = (df["hour"] >= 17) & (df["hour"] <= 19)
    df["is_rush_hour"] = (morning_rush | evening_rush) & ~df["is_weekend"]

    # Time of day categories
    def get_time_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    df["time_of_day"] = df["hour"].apply(get_time_of_day)

    # Price efficiency
    df["price_per_mile"] = df["price"] / df["distance"].replace(0, np.nan)

    return df


def merge_with_weather(
    rides_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    time_tolerance: str = "1h",
) -> pd.DataFrame:
    """
    Merge rides data with weather data based on location and nearest time.

    Uses merge_asof to match each ride to the closest weather reading
    for that location within the time tolerance.

    Parameters
    ----------
    rides_df : pd.DataFrame
        Preprocessed rides dataframe with 'source' and 'time_stamp' columns.
    weather_df : pd.DataFrame
        Weather dataframe with 'location' and 'time_stamp' columns.
    time_tolerance : str, default "1h"
        Maximum time difference for matching (pandas offset string).

    Returns
    -------
    pd.DataFrame
        Merged dataframe with weather features added.
    """
    rides_df = rides_df.copy()
    weather_df = weather_df.copy()

    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(rides_df["time_stamp"]):
        rides_df["time_stamp"] = pd.to_datetime(rides_df["time_stamp"])
    if not pd.api.types.is_datetime64_any_dtype(weather_df["time_stamp"]):
        weather_df["time_stamp"] = pd.to_datetime(weather_df["time_stamp"])

    # Rename weather timestamp to avoid collision
    weather_df = weather_df.rename(columns={"time_stamp": "weather_time"})

    # Sort both dataframes by time (required for merge_asof)
    rides_df = rides_df.sort_values("time_stamp").reset_index(drop=True)
    weather_df = weather_df.sort_values("weather_time").reset_index(drop=True)

    # Merge each location separately using merge_asof
    # (merge_asof doesn't support multiple 'by' columns efficiently)
    locations = rides_df["source"].unique()
    merged_parts = []

    for location in locations:
        rides_loc = rides_df[rides_df["source"] == location]
        weather_loc = weather_df[weather_df["location"] == location]

        if len(weather_loc) == 0:
            # No weather data for this location
            merged_parts.append(rides_loc)
            continue

        merged_loc = pd.merge_asof(
            rides_loc,
            weather_loc.drop(columns=["location"]),
            left_on="time_stamp",
            right_on="weather_time",
            tolerance=pd.Timedelta(time_tolerance),
            direction="nearest",
        )
        merged_parts.append(merged_loc)

    merged = pd.concat(merged_parts, ignore_index=True)

    # Clean up and restore original order
    merged = merged.drop(columns=["weather_time"], errors="ignore")
    merged = merged.sort_values("time_stamp").reset_index(drop=True)

    return merged
