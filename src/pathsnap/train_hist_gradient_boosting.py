
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


FEATURE_COLUMNS = [
    "distance_to_road_m",
    "heading_difference_deg",
    "speed_mps",
    "time_gap_s",
    "incoming_degree",
    "outgoing_degree",
]

TARGET = "label"
GROUP = "trip_id"


def load_dataset(csv_path):
    """Load and validate the candidate feature dataset."""
    df = pd.read_csv(csv_path)

    required = FEATURE_COLUMNS + [TARGET, GROUP]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET, GROUP]).copy()

    # Convert feature columns to numeric values.
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # HistGradientBoosting supports NaN, but not infinity.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The label column must contain only 0 and 1.")

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "The dataset must contain both label classes 0 and 1.")

    return df


def train_and_evaluate(df, output_dir):
    """Train using trip-wise splitting and evaluate on held-out trips."""

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    groups = df[GROUP]

    if groups.nunique() < 2:
        raise ValueError(
            "At least two different trips are needed for a "
            "trip-wise train/validation split."
        )

    # Keep all candidate rows from a trip in the same split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_indices, valid_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_valid = X.iloc[valid_indices]
    y_train = y.iloc[train_indices]
    y_valid = y.iloc[valid_indices]

    if y_train.nunique() != 2:
        raise ValueError(
            "Training split contains only one class. "
            "More labelled trips may be needed."
        )

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Training trips: {groups.iloc[train_indices].nunique()}")
    print(f"Validation trips: {groups.iloc[valid_indices].nunique()}")

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    print("\nTraining HistGradientBoosting...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(
            precision_score(y_valid, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_valid, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_valid, predictions, average="macro", zero_division=0)
        ),
    }

    if y_valid.nunique() == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_valid, probabilities)
        )

    print("\nValidation results")
    print("------------------")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_valid, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_valid,
            predictions,
            labels=[0, 1],
            target_names=["Incorrect candidate", "Correct candidate"],
            zero_division=0,
        )
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "hist_gradient_boosting.joblib"
    metrics_path = output_dir / "hgb_metrics.json"

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the PathSnap HistGradientBoosting model."
    )

    parser.add_argument(
        "--csv",
        default="results/train_features.csv",
        help="Path to the prepared training feature CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/models",
        help="Directory for the saved model and metrics.",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.csv}")
    df = load_dataset(args.csv)

    print(f"Loaded {len(df):,} candidate rows.")
    print(f"Number of trips: {df[GROUP].nunique()}")
    print(f"Label distribution:\n{df[TARGET].value_counts()}")

    train_and_evaluate(df, args.output_dir)


if __name__ == "__main__":
    main()
