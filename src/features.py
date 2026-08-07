"""
features.py — all feature engineering in one place.

Raw data doesn't have things like "driver form over last 5 races" —
we have to compute them ourselves before passing anything to a model.
"""

import pandas as pd
import numpy as np


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_podium: 1 if finished P1/P2/P3, else 0."""
    df = df.copy()
    df["is_podium"] = (df["positionOrder"] <= 3).astype(int)
    return df


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up grid position. grid == 0 means pit lane start (penalty),
    which we recode to 20 so the model treats it as a back-of-grid start.
    """
    df = df.copy()
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["grid_position"] = df["grid"].replace(0, 20)
    return df


def add_driver_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Average finish position over the last N races per driver.

    shift(1) is important — it stops the current race result from leaking
    into its own feature. We can only use what we knew *before* the race.
    """
    df = df.copy()
    df = df.sort_values(["year", "round"])
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")

    df["driver_avg_finish_last5"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    return df


def add_constructor_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Same as driver rolling form but per team.

    A team has 2 drivers per race, so we take the best (min) finish
    from either driver, then roll that over the last N races.
    """
    df = df.copy()
    df = df.sort_values(["year", "round"])

    best_per_race = (
        df.groupby(["raceId", "constructorId"])["positionOrder"]
        .min()
        .reset_index()
        .rename(columns={"positionOrder": "_best_finish"})
    )

    df = df.merge(best_per_race, on=["raceId", "constructorId"], how="left")

    df["constructor_avg_finish_last5"] = (
        df.groupby("constructorId")["_best_finish"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df = df.drop(columns=["_best_finish"])
    return df


def add_career_podium_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Career podium % for each driver up to (but not including) the current race.

    Uses expanding() so the window grows with every race — captures
    long-term quality rather than just recent form.
    """
    df = df.copy()
    df = df.sort_values(["year", "round"])

    df["driver_career_podium_rate"] = (
        df.groupby("driverId")["is_podium"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    return df


def add_qualifying_gap_to_pole(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gap in seconds between a driver's Q3 time and pole position.

    Raw lap times aren't comparable across circuits — 1:18 is fast at some
    tracks and slow at others. The gap to pole normalises for this.
    """
    df = df.copy()
    pole_times = df.groupby("raceId")["q3_seconds"].transform("min")
    df["q3_gap_to_pole"] = df["q3_seconds"] - pole_times
    return df


def add_championship_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Points gap to the championship leader at time of this race."""
    df = df.copy()
    leader_pts = df.groupby("raceId")["points_driver_standing"].transform("max")
    df["points_gap_to_leader"] = leader_pts - df["points_driver_standing"]
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in the correct order."""
    df = add_target_variable(df)
    df = add_grid_features(df)
    df = add_driver_rolling_form(df)
    df = add_constructor_rolling_form(df)
    df = add_career_podium_rate(df)       # needs is_podium from above
    df = add_qualifying_gap_to_pole(df)
    df = add_championship_pressure(df)
    return df


FEATURE_COLUMNS = [
    "grid_position",
    "q3_seconds",
    "q3_gap_to_pole",
    "q1_seconds",
    "driver_avg_finish_last5",
    "constructor_avg_finish_last5",
    "driver_career_podium_rate",
    "points_gap_to_leader",
]

TARGET_COLUMN = "is_podium"
