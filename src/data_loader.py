"""
data_loader.py

Responsible for reading the raw Kaggle CSVs and producing a single
merged DataFrame that's ready for feature engineering.

The raw data lives across 14 separate CSV files — this module joins
the ones we care about into one flat table, one row per (race, driver).
"""

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """
    Reads every CSV we need from data/raw/ and returns them as a
    dictionary keyed by a short name, e.g. {'races': df, 'results': df}.

    Using pathlib.Path instead of hardcoded strings means this works
    on Windows, Mac, and Linux without any changes.
    """
    files = {
        "races":                   "races.csv",
        "results":                 "results.csv",
        "qualifying":              "qualifying.csv",
        "drivers":                 "drivers.csv",
        "constructors":            "constructors.csv",
        "driver_standings":        "driver_standings.csv",
        "constructor_standings":   "constructor_standings.csv",
        "circuits":                "circuits.csv",
        "status":                  "status.csv",
    }

    tables = {}
    for key, filename in files.items():
        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing: {filepath}\n"
                f"Download the dataset from Kaggle and place CSVs in data/raw/"
            )
        tables[key] = pd.read_csv(filepath, na_values=["\\N", ""])
    return tables


def build_master_df(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Joins all the individual tables into one flat DataFrame.
    Each row = one driver's result in one race.

    Join order:
      results
        → races          (to get year, round, circuitId)
        → drivers        (to get driverRef, nationality)
        → constructors   (to get constructorRef)
        → qualifying     (to get Q1/Q2/Q3 lap times)
        → driver_standings   (to get championship points at that point)
        → constructor_standings

    'how="left"' means we keep ALL rows from results even if there's
    no matching row in the right table (e.g., no qualifying data).
    """
    results   = tables["results"]
    races     = tables["races"]
    drivers   = tables["drivers"]
    constructors = tables["constructors"]
    qualifying = tables["qualifying"]
    driver_standings = tables["driver_standings"]
    constructor_standings = tables["constructor_standings"]

    df = results.merge(
        races[["raceId", "year", "round", "circuitId", "name"]],
        on="raceId",
        how="left"
    )

    df = df.merge(
        drivers[["driverId", "driverRef", "nationality", "dob"]],
        on="driverId",
        how="left"
    )

    df = df.merge(
        constructors[["constructorId", "constructorRef", "nationality"]],
        on="constructorId",
        how="left",
        suffixes=("_driver", "_constructor")
    )

    df = df.merge(
        qualifying[["raceId", "driverId", "q1", "q2", "q3"]],
        on=["raceId", "driverId"],
        how="left"
    )

    df = df.merge(
        driver_standings[["raceId", "driverId", "points", "position", "wins"]],
        on=["raceId", "driverId"],
        how="left",
        suffixes=("", "_driver_standing")
    )

    df = df.merge(
        constructor_standings[["raceId", "constructorId", "points", "position"]],
        on=["raceId", "constructorId"],
        how="left",
        suffixes=("", "_constructor_standing")
    )

    return df


def parse_lap_time(time_str: str) -> float | None:
    """
    Converts a qualifying lap time string like "1:23.456" into total
    seconds as a float (83.456).

    Returns None (NaN) if the value is missing or malformed — this
    is common when a driver doesn't set a time in Q2 or Q3.
    """
    if pd.isna(time_str):
        return None
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(parts[0])
    except (ValueError, AttributeError):
        return None


def add_parsed_lap_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies parse_lap_time to Q1, Q2, Q3 columns.

    .apply() runs the function row-by-row on a column — it's slower
    than vectorized operations but necessary here because the parsing
    logic is complex (string manipulation).
    """
    df = df.copy()
    for col in ["q1", "q2", "q3"]:
        df[f"{col}_seconds"] = df[col].apply(parse_lap_time)
    return df


def get_master_df() -> pd.DataFrame:
    """
    Single entry point: load → merge → parse lap times.
    Call this from your notebooks with one line:

        from src.data_loader import get_master_df
        df = get_master_df()
    """
    tables = load_raw_tables()
    df = build_master_df(tables)
    df = add_parsed_lap_times(df)
    return df
