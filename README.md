# f1-ml-predictions

A machine learning project to predict Formula 1 podium finishers using historical race data from 1950–2024.

## Overview

This project uses gradient boosting (XGBoost/LightGBM) and logistic regression trained on F1 race history to classify whether a driver finishes on the podium (top 3). Features include grid position, qualifying pace, recent driver/constructor form, and circuit context.

## Dataset

[Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — Kaggle

Download the dataset and place the CSV files inside `data/raw/`.

## Project Structure

```
f1-ml-predictions/
├── data/
│   ├── raw/          # Original Kaggle CSVs (unmodified)
│   └── processed/    # Cleaned + feature-engineered data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   └── 05_gradient_boosting.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   └── predict.py
├── models/           # Saved trained model files
├── outputs/
│   └── figures/      # Charts and evaluation plots
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/f1-ml-predictions.git
cd f1-ml-predictions

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration` | Understand the raw tables, distributions, key patterns |
| `02_data_cleaning` | Handle missing values, parse lap times, merge tables |
| `03_feature_engineering` | Build rolling form, qualifying features, team strength |
| `04_baseline_model` | Logistic Regression baseline with evaluation metrics |
| `05_gradient_boosting` | XGBoost/LightGBM with hyperparameter tuning |

## Key Features Engineered

- **Grid position** — starting position on the grid
- **Qualifying pace** — Q3 lap time in seconds
- **Driver recent form** — rolling average finish position (last 5 races)
- **Constructor strength** — team's average points in current season
- **Podium rate** — career podium % for the driver

## Results

*Will be updated as models are trained.*

## Tech Stack

Python · Pandas · scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn
