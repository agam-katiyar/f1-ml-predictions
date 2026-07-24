"""
Run this to convert exploration.py to a notebook:
    pip install jupytext
    jupytext --to notebook notebooks/01_data_exploration.py

Or just open Jupyter and copy sections cell by cell.
Each # %% marks a new notebook cell.
"""

# %% [markdown]
# # 01 — Data Exploration
#
# Before building any model, we need to understand the data:
# - How many rows and columns?
# - What does each table actually contain?
# - Are there missing values?
# - What patterns exist that a model could learn from?
#
# This is called EDA — Exploratory Data Analysis.

# %%
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works outside Jupyter too
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from pathlib import Path

# Anchor all paths relative to THIS file, not wherever Python was launched from
NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.data_loader import load_raw_tables, build_master_df, add_parsed_lap_times

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", 50)

# %% [markdown]
# ## Load the raw tables

# %%
tables = load_raw_tables()

for name, df in tables.items():
    print(f"{name:30s}  shape: {df.shape}")

# %% [markdown]
# ## Inspect results.csv — our most important table
#
# Every row is one driver's result in one race.
# `positionOrder` = final classified position. This is what we're predicting.

# %%
results = tables["results"]
print(results.dtypes)
print()
results.head(10)

# %%
print("Missing values in results:")
print(results.isnull().sum()[results.isnull().sum() > 0])

# %% [markdown]
# ## Key question 1: Does grid position predict podium?
#
# If you start from pole (P1), you're already ahead of everyone.
# Let's see how strong this signal actually is.

# %%
df = build_master_df(tables)
df = add_parsed_lap_times(df)

df["is_podium"] = (df["positionOrder"] <= 3).astype(int)
df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
df["grid_position"] = df["grid"].replace(0, 20)

podium_by_grid = (
    df[df["grid_position"].between(1, 20)]
    .groupby("grid_position")["is_podium"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(podium_by_grid["grid_position"], podium_by_grid["is_podium"], color="steelblue")
ax.set_xlabel("Starting Grid Position")
ax.set_ylabel("Podium Rate")
ax.set_title("Podium Rate by Starting Grid Position (1950–2024)")
ax.set_xticks(range(1, 21))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "podium_rate_by_grid.png", dpi=150)
plt.close()
print("  Saved: podium_rate_by_grid.png")

# %% [markdown]
# ## Key question 2: Class balance
#
# Only 3 of 20 drivers get a podium per race. That's ~15% of rows.
# This is called class imbalance — the model will see far more "no podium"
# examples and could learn to just predict "no podium" always and get 85%
# accuracy. We need to handle this explicitly.

# %%
balance = df["is_podium"].value_counts(normalize=True)
print("Class distribution:")
print(balance.rename({0: "No Podium", 1: "Podium"}))

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["No Podium", "Podium"], balance.values, color=["#e74c3c", "#2ecc71"])
ax.set_ylabel("Proportion of rows")
ax.set_title("Target Variable Class Balance")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
plt.close()
print("  Saved: class_balance.png")

# %% [markdown]
# ## Key question 3: How much qualifying data is available?
#
# Not all drivers set Q2/Q3 times (only top-15/10 do).
# We need to know the extent of missing data before using qualifying features.

# %%
for col in ["q1_seconds", "q2_seconds", "q3_seconds"]:
    if col in df.columns:
        missing_pct = df[col].isna().mean()
        print(f"{col}: {missing_pct:.1%} missing")

# %% [markdown]
# ## Key question 4: Which constructors dominate?
#
# Constructor identity (team) is a massive factor in F1.
# Let's see which teams won the most races.

# %%
constructor_wins = (
    df[df["positionOrder"] == 1]
    .groupby("constructorRef")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(10, 5))
constructor_wins.plot(kind="bar", ax=ax, color="tomato")
ax.set_xlabel("Constructor")
ax.set_ylabel("Race Wins")
ax.set_title("Top 15 Constructors by Race Wins (1950–2024)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "constructor_wins.png", dpi=150)
plt.close()
print("  Saved: constructor_wins.png")

# %% [markdown]
# ## Key question 5: Races per year
#
# The number of races per season has grown a lot — from 7 in 1950
# to 24 in recent years. Important context for rolling calculations.

# %%
races_per_year = df.groupby("year")["raceId"].nunique()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(races_per_year.index, races_per_year.values, marker="o", markersize=3, color="steelblue")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Races")
ax.set_title("F1 Races Per Season (1950–2024)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "races_per_year.png", dpi=150)
plt.close()
print("  Saved: races_per_year.png")

print("\nDone! Review the figures in outputs/figures/")
