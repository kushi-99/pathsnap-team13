
"""Evaluation utilities for PathSnap candidate-scoring models."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Evaluate binary candidate classification."""

    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Labels and predictions must be 1D")

    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Empty or mismatched arrays")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Actual labels must be binary")

    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Predicted labels must be binary")

    metrics = {
        "num_candidates": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(
            precision_score(actual, predicted, average="macro",
                            zero_division=0)
        ),
        "macro_recall": float(
            recall_score(actual, predicted, average="macro",
                         zero_division=0)
        ),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro",
                     zero_division=0)
        ),
        "positive_precision": float(
            precision_score(actual, predicted, pos_label=1,
                            zero_division=0)
        ),
        "positive_recall": float(
            recall_score(actual, predicted, pos_label=1,
                         zero_division=0)
        ),
        "positive_f1": float(
            f1_score(actual, predicted, pos_label=1,
                     zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
    }

    if y_prob is not None:
        probabilities = np.asarray(y_prob, dtype=np.float64)

        if probabilities.ndim != 1 or len(probabilities) != len(actual):
            raise ValueError("Probability array has invalid shape")

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities contain NaN or Inf")

        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Probabilities must be between 0 and 1")

        metrics["log_loss"] = float(
            log_loss(actual, probabilities, labels=[0, 1])
        )

        metrics["roc_auc"] = (
            float(roc_auc_score(actual, probabilities))
            if len(np.unique(actual)) == 2
            else None
        )

    return metrics


def evaluate_top1(
    candidates: pd.DataFrame,
    score_column: str = "match_prob",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Evaluate observation-level Top-1 candidate selection.

    Select the highest-scoring candidate for each observation.
    This is classifier-only accuracy, not Viterbi route accuracy.

    Uncovered observations cannot be inferred from candidate rows alone.
    """

    required = {
        "trip_id", "obs_idx", "arc_id",
        score_column, label_column,
    }

    missing = required.difference(candidates.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if candidates.empty:
        raise ValueError("Candidate table is empty")

    if candidates[["trip_id", "obs_idx", "arc_id"]].isna().any().any():
        raise ValueError("Candidate identifiers contain missing values")

    if candidates.duplicated(["trip_id", "obs_idx", "arc_id"]).any():
        raise ValueError("Duplicate candidate identifiers")

    scores = pd.to_numeric(
        candidates[score_column], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores contain NaN or Inf")

    labels = pd.to_numeric(
        candidates[label_column], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid binary ground-truth labels")

    frame = candidates.copy()
    frame[label_column] = labels.astype(int)
    frame[score_column] = scores

    group_columns = ["trip_id", "obs_idx"]

    # Stable tie-breaking by arc ID.
    frame = frame.sort_values(
        group_columns + [score_column, "arc_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    top1 = frame.drop_duplicates(
        group_columns, keep="first"
    )

    total = len(top1)
    correct = int(top1[label_column].sum())

    covered = (
        frame.groupby(group_columns)[label_column]
        .max()
        .reset_index(name="has_positive_candidate")
    )

    covered_count = int(covered["has_positive_candidate"].sum())

    return {
        "num_observations_with_candidates": total,
        "num_covered_observations": covered_count,
        "num_correct_top1": correct,
        "top1_accuracy_all_candidate_observations": correct / total,
        "top1_accuracy_covered": (
            correct / covered_count if covered_count else None
        ),
        "candidate_miss_count_within_candidate_groups":
            total - covered_count,
        "note": (
            "Observations with zero generated candidates are not "
            "included. Provide the full observation count separately "
            "for true all-observation accuracy."
        ),
    }


def evaluate_per_trip(
    candidates: pd.DataFrame,
    score_column: str = "match_prob",
    label_column: str = "label",
) -> pd.DataFrame:
    """Calculate classifier-only Top-1 accuracy for each trip."""

    if "trip_id" not in candidates.columns:
        raise ValueError("Missing trip_id")

    rows = []

    for trip_id, group in candidates.groupby("trip_id", sort=True):
        result = evaluate_top1(
            group,
            score_column=score_column,
            label_column=label_column,
        )

        rows.append({
            "trip_id": str(trip_id),
            "num_observations": result[
                "num_observations_with_candidates"
            ],
            "covered_observations": result[
                "num_covered_observations"
            ],
            "top1_accuracy": result[
                "top1_accuracy_all_candidate_observations"
            ],
            "top1_accuracy_covered": result[
                "top1_accuracy_covered"
            ],
        })

    return pd.DataFrame(rows)


def save_evaluation(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Save evaluation metrics as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, allow_nan=False)

    return output_path
