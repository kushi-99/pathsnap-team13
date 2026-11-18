
"""Feature schema and numerical preparation utilities for PathSnap."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd

from ml.mlp.config import DEFAULT_FEATURE_COLUMNS


def get_feature_columns(
    schema_path: Optional[Path] = None,
) -> List[str]:
    """Return the canonical 28-feature schema."""

    if schema_path is None:
        return list(DEFAULT_FEATURE_COLUMNS)

    schema_path = Path(schema_path)

    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")

    with schema_path.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    columns = schema.get("feature_columns")

    if not isinstance(columns, list) or not columns:
        raise ValueError("Invalid feature schema")

    if len(columns) != len(set(columns)):
        raise ValueError("Feature schema contains duplicate columns")

    return columns


def validate_feature_frame(
    candidates: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Validate that candidate data contains all required features."""

    if candidates.empty:
        raise ValueError("Candidate DataFrame is empty")

    columns = list(
        feature_columns
        if feature_columns is not None
        else get_feature_columns()
    )

    missing = [column for column in columns if column not in candidates.columns]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required features: {missing}"
        )

    features = candidates.loc[:, columns].copy()

    features = features.apply(pd.to_numeric, errors="coerce")

    values = features.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        bad_columns = [
            column
            for column in columns
            if not np.isfinite(
                features[column].to_numpy(dtype=np.float64)
            ).all()
        ]

        raise ValueError(
            f"Non-finite values detected in features: {bad_columns}"
        )

    return features


def prepare_feature_matrix(
    candidates: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert candidate features into a model-ready float32 matrix.

    Scaling is deliberately excluded. Fit StandardScaler only on the
    training split to prevent data leakage.
    """

    features = validate_feature_frame(
        candidates,
        feature_columns=feature_columns,
    )

    matrix = features.to_numpy(dtype=np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or Inf")

    return matrix, list(features.columns)


def add_feature_schema(
    candidates: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a copy of candidates with validated feature columns."""

    validate_feature_frame(candidates, feature_columns)

    return candidates.copy()


def save_feature_schema(
    output_path: Path,
    feature_columns: Optional[Sequence[str]] = None,
) -> Path:
    """Save the canonical feature-column schema as JSON."""

    columns = list(
        feature_columns
        if feature_columns is not None
        else get_feature_columns()
    )

    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Invalid feature column list")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"feature_columns": columns},
            file,
            indent=4,
        )

    return output_path
