"""
model.py

Handles training, evaluation, and saving of ML models.

We keep the model code separate from notebooks so you can import it
cleanly and avoid copy-pasting the same boilerplate across notebooks.
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

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int = 2020,
) -> tuple:
    """
    Splits data by year rather than randomly.

    Why temporal split instead of random? Because F1 results have
    time-based dependencies — a driver's rolling form leaks information
    forward in time. A random split would let future data "inform" past
    predictions, giving artificially high accuracy. Splitting at a fixed
    year is closer to real-world deployment: train on old data, predict new.

    Returns: X_train, X_test, y_train, y_test
    """
    train_mask = df["year"] < test_year
    test_mask  = df["year"] >= test_year

    X_train = df.loc[train_mask, feature_cols]
    X_test  = df.loc[test_mask,  feature_cols]
    y_train = df.loc[train_mask, target_col]
    y_test  = df.loc[test_mask,  target_col]

    print(f"Train: {train_mask.sum()} rows ({df.loc[train_mask, 'year'].min()}–{test_year-1})")
    print(f"Test:  {test_mask.sum()} rows ({test_year}–{df.loc[test_mask, 'year'].max()})")
    print(f"Podium rate — train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline() -> Pipeline:
    """
    Returns a sklearn Pipeline that chains two steps:
      1. StandardScaler — normalises all features to have mean=0, std=1
      2. LogisticRegression — the classifier

    Why scale? Logistic regression is sensitive to feature magnitudes.
    Grid position (1–20) and qualifying gap (0–5 seconds) are on
    completely different scales. Scaling puts them on equal footing.

    Pipeline ensures the scaler is fitted only on training data and
    applied consistently to test data — prevents a subtle bug called
    data leakage from scaling.
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
    XGBoost classifier with sensible defaults.

    scale_pos_weight handles class imbalance: ~83% non-podium vs ~17%
    podium. Setting it to (non-podium count / podium count ≈ 5) tells
    XGBoost to penalise misclassifying the minority class (podiums) 5×
    more than the majority class.

    n_estimators = number of trees to build (boosting rounds).
    learning_rate = how much each new tree corrects the previous error.
    max_depth = how deep each tree grows (controls overfitting).
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    """
    Prints a full evaluation report and shows two plots:
      1. Confusion matrix — raw count of TP/TN/FP/FN
      2. ROC curve — AUC score, overall ranking quality

    predict_proba returns a 2D array — [:, 1] takes the second column
    which is the probability of class 1 (podium). This is used for
    AUC-ROC and the ROC curve.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Podium", "Podium"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(model_name, fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
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
    print(f"Plot saved to {save_path}")


def save_model(model, filename: str):
    """
    Serialises a trained model to disk using joblib.

    joblib is preferred over pickle for sklearn objects because it
    handles large numpy arrays more efficiently.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(filename: str):
    path = MODELS_DIR / filename
    return joblib.load(path)
