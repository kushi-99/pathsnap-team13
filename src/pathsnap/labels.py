
"""Ground-truth label handling and validation for PathSnap."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LABEL_COLUMN = "label"
VALID_LABELS = {0, 1}


def extract_labels(
    candidates: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
) -> np.ndarray:
    """Extract binary ground-truth labels from candidate records."""

    if label_column not in candidates.columns:
        raise ValueError(
            f"Required label column '{label_column}' is missing"
        )

    labels = pd.to_numeric(
        candidates[label_column],
        errors="coerce",
    )

    if labels.isnull().any():
        raise ValueError("Labels contain missing or non-numeric values")

    values = labels.to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError("Labels contain NaN or Inf")

    unique_values = set(np.unique(values).tolist())

    if not unique_values.issubset(VALID_LABELS):
        raise ValueError(
            f"Expected binary labels 0 and 1, found {sorted(unique_values)}"
        )

    return values.astype(np.int64)


def validate_labels(
    labels: Sequence[int],
    require_both_classes: bool = False,
) -> np.ndarray:
    """Validate a binary label vector."""

    values = np.asarray(labels)

    if values.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array")

    if len(values) == 0:
        raise ValueError("Label array is empty")

    if not np.isfinite(values).all():
        raise ValueError("Labels contain NaN or Inf")

    if not np.isin(values, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")

    values = values.astype(np.int64)

    if require_both_classes and len(np.unique(values)) != 2:
        raise ValueError("Both positive and negative classes are required")

    return values


def label_distribution(
    labels: Sequence[int],
) -> Dict[str, Any]:
    """Calculate positive and negative label counts and percentages."""

    values = validate_labels(labels)

    total = len(values)
    positive = int(np.sum(values == 1))
    negative = int(np.sum(values == 0))

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": positive / total,
        "negative_rate": negative / total,
    }


def validate_candidate_labels(
    candidates: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate candidate records and return their label statistics."""

    if candidates.empty:
        raise ValueError("Candidate DataFrame is empty")

    labels = extract_labels(
        candidates,
        label_column=label_column,
    )

    summary = label_distribution(labels)

    if summary["positive"] == 0:
        raise ValueError("No positive candidate labels found")

    return candidates.copy(), summary


def attach_labels(
    candidates: pd.DataFrame,
    labels: Sequence[int],
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Attach externally supplied labels to candidate records."""

    values = validate_labels(labels)

    if len(values) != len(candidates):
        raise ValueError(
            f"Label count ({len(values)}) does not match "
            f"candidate count ({len(candidates)})"
        )

    result = candidates.copy()
    result[label_column] = values

    return result
