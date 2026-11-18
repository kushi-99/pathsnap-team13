
"""Candidate-generation interface for the PathSnap project.

Reuses the validated candidate-generation implementation from the MLP
pipeline so that all models receive a consistent candidate table.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.mlp.config import CandidateConfig
from ml.mlp.data_adapter import extract_trip_candidates

from .data_loader import RawTrip, load_partition, load_raw_trip


def generate_candidates(
    trip: RawTrip,
    config: Optional[CandidateConfig] = None,
    include_labels: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate road-segment candidates for every GPS observation in a trip.

    Args:
        trip: Loaded PathSnap trip.
        config: Candidate-generation configuration.
        include_labels: Whether to attach ground-truth labels.

    Returns:
        A candidate-level DataFrame and a generation summary.
    """

    if config is None:
        config = CandidateConfig()

    candidates, summary = extract_trip_candidates(
        trip,
        cfg=config,
        load_labels=include_labels,
    )

    required_columns = {"trip_id", "obs_idx", "arc_id"}

    missing = required_columns.difference(candidates.columns)

    if missing:
        raise ValueError(
            f"Candidate generator returned missing columns: {sorted(missing)}"
        )

    if candidates.empty:
        raise ValueError(
            f"No candidates generated for trip {trip.trip_id}"
        )

    if candidates[["trip_id", "obs_idx", "arc_id"]].isnull().any().any():
        raise ValueError("Candidate output contains missing identifiers")

    if candidates.duplicated(["trip_id", "obs_idx", "arc_id"]).any():
        raise ValueError("Duplicate candidate rows detected")

    return candidates, summary


def generate_trip_candidates(
    trip_dir: Path,
    config: Optional[CandidateConfig] = None,
    include_labels: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load one trip and generate its candidate table."""

    trip = load_raw_trip(
        Path(trip_dir),
        load_route=include_labels,
    )

    return generate_candidates(
        trip,
        config=config,
        include_labels=include_labels,
    )


def generate_partition_candidates(
    partition_dir: Path,
    config: Optional[CandidateConfig] = None,
    include_labels: bool = True,
    strict: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate and combine candidate tables for a complete partition.

    Returns:
        combined_candidates:
            One row per observation-road-segment candidate.

        trip_summaries:
            One summary row per successfully processed trip.
    """

    trips = load_partition(
        Path(partition_dir),
        load_route=include_labels,
        strict=strict,
    )

    candidate_tables: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []

    for index, trip in enumerate(trips, start=1):
        candidates, summary = generate_candidates(
            trip,
            config=config,
            include_labels=include_labels,
        )

        candidate_tables.append(candidates)
        summaries.append(dict(summary))

        print(
            f"[CANDIDATES] Processed {index}/{len(trips)} trips "
            f"({trip.trip_id}): {len(candidates):,} candidate rows"
        )

    combined = pd.concat(
        candidate_tables,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summaries)

    if combined.empty:
        raise RuntimeError("Partition candidate generation produced no rows")

    return combined, summary_df


def save_candidates(
    candidates: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save candidate data as CSV or Parquet."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        candidates.to_csv(output_path, index=False)

    elif suffix in (".parquet", ".pq"):
        candidates.to_parquet(output_path, index=False)

    else:
        raise ValueError(
            f"Unsupported output format: {suffix}. "
            "Use CSV or Parquet."
        )

    print(
        f"[CANDIDATES] Saved {len(candidates):,} rows to {output_path}"
    )

    return output_path
