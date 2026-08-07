"""
data_loader.py — reads raw CSVs and merges them into one flat DataFrame.
One row = one driver's result in one race.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Load the CSVs we need from data/raw/ into a dict of DataFrames."""
    files = {
        "races":                 "races.csv",
        "results":               "results.csv",
        "qualifying":            "qualifying.csv",
        "drivers":               "drivers.csv",
        "constructors":          "constructors.csv",
        "driver_standings":      "driver_standings.csv",
        "constructor_standings": "constructor_standings.csv",
        "circuits":              "circuits.csv",
        "status":                "status.csv",
    }

    tables = {}
    for key, filename in files.items():
        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing: {filepath}\n"
                f"Download from Kaggle and put CSVs in data/raw/"
            )
        tables[key] = pd.read_csv(filepath, na_values=["\\N", ""])
    return tables


def build_master_df(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join all tables into one flat DataFrame.

    Join order: results → races → drivers → constructors
                → qualifying → driver_standings → constructor_standings

    Using left joins so we keep all rows from results even when
    qualifying data doesn't exist (pre-2006 races).
    """
    results               = tables["results"]
    races                 = tables["races"]
    drivers               = tables["drivers"]
    constructors          = tables["constructors"]
    qualifying            = tables["qualifying"]
    driver_standings      = tables["driver_standings"]
    constructor_standings = tables["constructor_standings"]

    df = results.merge(
        races[["raceId", "year", "round", "circuitId", "name"]],
        on="raceId", how="left"
    )
    df = df.merge(
        drivers[["driverId", "driverRef", "nationality", "dob"]],
        on="driverId", how="left"
    )
    df = df.merge(
        constructors[["constructorId", "constructorRef", "nationality"]],
        on="constructorId", how="left",
        suffixes=("_driver", "_constructor")
    )
    df = df.merge(
        qualifying[["raceId", "driverId", "q1", "q2", "q3"]],
        on=["raceId", "driverId"], how="left"
    )
    df = df.merge(
        driver_standings[["raceId", "driverId", "points", "position", "wins"]],
        on=["raceId", "driverId"], how="left",
        suffixes=("", "_driver_standing")
    )
    df = df.merge(
        constructor_standings[["raceId", "constructorId", "points", "position"]],
        on=["raceId", "constructorId"], how="left",
        suffixes=("", "_constructor_standing")
    )

    return df


def parse_lap_time(time_str: str) -> float | None:
    """Convert '1:23.456' → 83.456 seconds. Returns None if missing/malformed."""
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
    """Add q1_seconds, q2_seconds, q3_seconds columns parsed from raw strings."""
    df = df.copy()
    for col in ["q1", "q2", "q3"]:
        df[f"{col}_seconds"] = df[col].apply(parse_lap_time)
    return df


def get_master_df() -> pd.DataFrame:
    """Shortcut: load + merge + parse lap times in one call."""
    tables = load_raw_tables()
    df = build_master_df(tables)
    df = add_parsed_lap_times(df)
    return df
