
"""Baseline machine-learning training and comparison for PathSnap."""

from pathlib import Path
from typing import Any, Dict, List

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import prepare_feature_matrix
from .labels import extract_labels


def build_baseline_models() -> Dict[str, Pipeline]:
    """Create the baseline classifiers."""

    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        "hist_gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                max_leaf_nodes=31,
                random_state=42,
            )),
        ]),
    }


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """Calculate classification performance metrics."""

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(
            precision_score(actual, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(actual, predicted, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
        "positive_f1": float(
            f1_score(actual, predicted, pos_label=1, zero_division=0)
        ),
    }


def train_and_compare(
    candidates: pd.DataFrame,
    output_dir: Path = Path("artifacts/baselines"),
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Perform trip-grouped cross-validation and train final baseline models.

    Each trip remains entirely within one CV fold.
    Final models are fitted on the supplied training candidates only.

    The caller must not pass student_test or evaluator_test data here.
    """

    required = {"trip_id", "label"}

    missing = required.difference(candidates.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if candidates.empty:
        raise ValueError("Training dataset is empty")

    if candidates["trip_id"].isnull().any():
        raise ValueError("Training data contains missing trip IDs")

    X, feature_columns = prepare_feature_matrix(candidates)
    y = extract_labels(candidates)
    groups = candidates["trip_id"].astype(str).to_numpy()

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        raise ValueError("At least two distinct trips are required")

    if len(np.unique(y)) != 2:
        raise ValueError("Training requires both positive and negative labels")

    splits = min(n_splits, len(unique_groups))
    cv = GroupKFold(n_splits=splits)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    models = build_baseline_models()

    for model_name, model in models.items():
        fold_metrics = []

        print(f"\nTraining baseline: {model_name}")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups),
            start=1,
        ):
            if len(np.unique(y[train_idx])) < 2:
                raise ValueError(
                    f"{model_name}, fold {fold}: training fold "
                    "contains only one class"
                )

            fold_model = clone(model)

            fold_model.fit(X[train_idx], y[train_idx])

            predictions = fold_model.predict(X[val_idx])

            metrics = calculate_metrics(y[val_idx], predictions)
            fold_metrics.append(metrics)

            print(
                f"  Fold {fold}/{splits}: "
                f"Macro-F1={metrics['macro_f1']:.4f}"
            )

        mean_metrics = {
            key: float(np.mean([fold[key] for fold in fold_metrics]))
            for key in fold_metrics[0]
        }

        std_f1 = float(
            np.std([fold["macro_f1"] for fold in fold_metrics])
        )

        # Refit the selected baseline candidates on all supplied
        # training trips after cross-validation.
        final_model = clone(model)
        final_model.fit(X, y)

        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(final_model, model_path)

        result = {
            "model": model_name,
            "cv_folds": splits,
            "num_trips": int(len(unique_groups)),
            "num_candidates": int(len(y)),
            **mean_metrics,
            "macro_f1_std": std_f1,
            "artifact": str(model_path),
        }

        results.append(result)

        print(
            f"  Mean Macro-F1: {mean_metrics['macro_f1']:.4f} "
            f"(+/- {std_f1:.4f})"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(
        output_dir / "baseline_comparison.csv",
        index=False,
    )

    metadata = {
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "num_training_trips": int(len(unique_groups)),
        "num_candidates": int(len(y)),
        "cv_strategy": "GroupKFold",
        "cv_folds": splits,
        "random_seed": 42,
        "models": results,
    }

    with (output_dir / "baseline_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nBASELINE COMPARISON COMPLETE")
    print(results_df.to_string(index=False))

    return results_df
