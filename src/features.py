"""
features.py

All feature engineering lives here.

Feature engineering = creating new columns from raw data that give the
model signals it couldn't extract on its own. For example, the model
can't compute "this driver's average finish over the last 5 races" by
itself — we have to build that column explicitly.
"""

import pandas as pd
import numpy as np


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the column we're trying to predict: is_podium.

    positionOrder is the final race classification (1st = 1, 2nd = 2...).
    We label any finish of 1, 2, or 3 as a podium (1), everything else 0.

    This is called a binary label — the model outputs a probability of
    being in class 1 (podium) for each row.
    """
    df = df.copy()
    df["is_podium"] = (df["positionOrder"] <= 3).astype(int)
    return df


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grid position is the single strongest predictor — starting from
    pole (P1) is a massive advantage.

    grid == 0 means the driver started from the pit lane after a grid
    penalty, which is effectively the back of the grid. We recode it
    to 20 so the model understands it's a disadvantage.
    """
    df = df.copy()
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["grid_position"] = df["grid"].replace(0, 20)
    return df


def add_driver_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculates each driver's average finishing position over their last
    N races (default 5). A lower value = better recent form.

    The tricky part: we must sort by (year, round) first, then group
    by driver. Inside each driver group we compute a rolling mean with
    shift(1) — the shift ensures we only use PAST races, not the current
    one. Using the current race result as a feature would be data leakage
    (cheating, because we wouldn't know it at prediction time).

    min_periods=1 means we still compute a mean even if the driver has
    fewer than `window` past races (e.g., their first season).
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
    Same idea as driver rolling form but aggregated by constructor.

    For each race, a team has TWO drivers. We want a single number for
    team form, so we first take the best (minimum) finish between the
    two drivers per race, then roll that.

    This is computed in two steps:
      1. Get the best driver finish per (race, constructor)
      2. Merge that back and roll it over time
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
    For each row, computes the driver's podium rate from ALL races
    before this one (career-to-date).

    Uses .expanding() instead of .rolling() — expanding means the
    window grows from race 1 all the way to the previous race, rather
    than being capped at N races. This captures long-term driver quality.

    shift(1) again prevents leakage from the current race.
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
    A driver's raw Q3 time doesn't mean much alone — 1:18.000 could be
    fast or slow depending on the circuit. What matters is how far they
    are from the fastest Q3 time (pole position) in that race.

    We compute: driver's Q3 time − fastest Q3 time in the same race.
    This gives a delta in seconds. 0.0 = pole, 0.3 = three tenths off
    pole, etc.

    groupby("raceId")["q3_seconds"].transform("min") computes the minimum
    Q3 time within each raceId group and broadcasts it back to all rows
    of that race — so every driver in that race gets the pole time value.
    """
    df = df.copy()
    pole_times = df.groupby("raceId")["q3_seconds"].transform("min")
    df["q3_gap_to_pole"] = df["q3_seconds"] - pole_times
    return df


def add_championship_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many points behind the championship leader is this driver?

    A large gap = out of contention (may drive differently).
    A small gap = championship battle (maximum pressure, maximum effort).

    points_driver_standing comes from the driver_standings table joined
    earlier. We compute the leader's points in the same race and subtract.
    """
    df = df.copy()
    leader_pts = df.groupby("raceId")["points_driver_standing"].transform("max")
    df["points_gap_to_leader"] = leader_pts - df["points_driver_standing"]
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all feature engineering steps in order and returns the final
    DataFrame. Call this from your notebooks after loading data.

    Note the order matters: add_target_variable must come before
    add_career_podium_rate (which uses is_podium).
    """
    df = add_target_variable(df)
    df = add_grid_features(df)
    df = add_driver_rolling_form(df)
    df = add_constructor_rolling_form(df)
    df = add_career_podium_rate(df)
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
