"""
model.py — training, evaluation, and saving models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int = 2020,
) -> tuple:
    """
    Split by year, not randomly.

    Random splits leak future form data into past predictions because of
    the rolling features. Training on pre-2020, testing on 2020+ is closer
    to how the model would actually be used.
    """
    train_mask = df["year"] < test_year
    test_mask  = df["year"] >= test_year

    X_train = df.loc[train_mask, feature_cols]
    X_test  = df.loc[test_mask,  feature_cols]
    y_train = df.loc[train_mask, target_col]
    y_test  = df.loc[test_mask,  target_col]

    print(f"Train: {train_mask.sum()} rows  ({int(df.loc[train_mask, 'year'].min())}–{test_year-1})")
    print(f"Test:  {test_mask.sum()} rows  ({test_year}–{int(df.loc[test_mask, 'year'].max())})")
    print(f"Podium rate — train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline() -> Pipeline:
    """
    Logistic regression wrapped in a Pipeline with a scaler.

    Scaling is required — grid position (1–20) and qualifying gap (seconds)
    are on very different scales. Pipeline also prevents the scaler from
    accidentally fitting on test data.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def build_xgboost_model() -> XGBClassifier:
    """
    XGBoost with settings tuned for the class imbalance (~13% podiums).

    scale_pos_weight=5 roughly matches the 5:1 ratio of non-podiums to
    podiums, so the model pays more attention to the minority class.
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    """Print classification report + save confusion matrix and ROC curve."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Podium", "Podium"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(model_name, fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Podium", "Podium"],
        yticklabels=["No Podium", "Podium"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_eval.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def save_model(model, filename: str):
    """Save trained model to models/ using joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"Saved: {path}")


def load_model(filename: str):
    return joblib.load(MODELS_DIR / filename)
