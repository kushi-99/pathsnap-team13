"""Validate raw trips and candidate-level feature tables."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import list_trip_ids, load_trip
from .features import FEATURE_COLUMNS


REQUIRED_COLUMNS = [
    "trip_id", "observation_index", "candidate_arc_id", "reference_arc_id",
    "label", *FEATURE_COLUMNS,
]


def validate_raw_trips(data_root):
    rows = []
    for split in ("train", "student_test"):
        for trip_id in list_trip_ids(data_root, split):
            trip = load_trip(data_root, trip_id, split)
            track, nodes, arcs, route = (trip[key] for key in ("track", "nodes", "arcs", "route"))
            endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
            valid_endpoints = ((endpoints >= 0) & (endpoints < len(nodes))).all(axis=1)
            route_ids = route.to_numpy(dtype=int)
            gaps = np.diff(track.timestamp_s.to_numpy(dtype=float))
            coords = track[["longitude", "latitude"]].to_numpy(dtype=float)
            node_coords = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
            rows.append({
                "split": split, "trip_id": str(trip_id),
                "gps_observations": len(track), "road_nodes": len(nodes),
                "road_arcs": len(arcs), "route_arcs": len(route),
                "missing_track_cells": int(track.isna().sum().sum()),
                "duplicate_track_rows": int(track.duplicated().sum()),
                "invalid_gps_coordinates": int((~track.longitude.between(-180, 180) | ~track.latitude.between(-90, 90)).sum()),
                "nonfinite_gps_coordinates": int((~np.isfinite(coords)).any(axis=1).sum()),
                "invalid_node_coordinates": int((~nodes.longitude.between(-180, 180) | ~nodes.latitude.between(-90, 90)).sum()),
                "nonfinite_node_coordinates": int((~np.isfinite(node_coords)).any(axis=1).sum()),
                "nonincreasing_timestamp_steps": int((gaps <= 0).sum()),
                "max_time_gap_s": float(gaps.max()) if len(gaps) else np.nan,
                "invalid_arc_endpoints": int((~valid_endpoints).sum()),
                "route_arc_ids_out_of_range": int(((route_ids < 0) | (route_ids >= len(arcs))).sum()),
                "route_arc_ids_repeated": int(pd.Series(route_ids).duplicated().sum()),
            })
    return pd.DataFrame(rows)


def validate_feature_table(path, split, chunksize=100_000):
    path = Path(path)
    header = list(pd.read_csv(path, nrows=0).columns)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    if len(header) != len(set(header)):
        raise ValueError(f"{path} contains duplicate column names")

    stats = {name: {"missing": 0, "negative": 0, "min": np.inf, "max": -np.inf} for name in FEATURE_COLUMNS}
    label_counts = {0: 0, 1: 0}
    row_count = 0
    invalid_labels = 0
    label_alignment_mismatches = 0
    missing_identifier_rows = 0
    trip_observations = {}
    carry = None

    def add_observation_groups(frame):
        if frame is None or frame.empty:
            return
        grouped = frame.groupby(["trip_id", "observation_index"], sort=False).agg(
            positives=("label", "max"), candidate_count=("candidate_arc_id", "size"))
        for trip_id, trip_groups in grouped.groupby(level=0, sort=False):
            values = trip_observations.setdefault(str(trip_id), [0, 0, []])
            values[0] += len(trip_groups)
            values[1] += int(trip_groups.positives.sum())
            values[2].extend(trip_groups.candidate_count.astype(int).tolist())

    columns = ["trip_id", "observation_index", "candidate_arc_id", "reference_arc_id", "label", *FEATURE_COLUMNS]
    for chunk in pd.read_csv(path, dtype={"trip_id": str}, usecols=columns, chunksize=chunksize):
        row_count += len(chunk)
        missing_identifier_rows += int(chunk[["trip_id", "observation_index", "candidate_arc_id", "reference_arc_id", "label"]].isna().any(axis=1).sum())
        numeric_label = pd.to_numeric(chunk.label, errors="coerce")
        invalid_labels += int((~numeric_label.isin([0, 1])).sum())
        for label, count in numeric_label.value_counts(dropna=True).items():
            if int(label) in label_counts:
                label_counts[int(label)] += int(count)
        label_alignment_mismatches += int((numeric_label != (chunk.candidate_arc_id == chunk.reference_arc_id).astype(int)).sum())
        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(chunk[feature], errors="coerce")
            stats[feature]["missing"] += int(values.isna().sum())
            finite = values[np.isfinite(values)]
            if len(finite):
                stats[feature]["min"] = min(stats[feature]["min"], float(finite.min()))
                stats[feature]["max"] = max(stats[feature]["max"], float(finite.max()))
                stats[feature]["negative"] += int((finite < 0).sum())

        work = pd.concat([carry, chunk], ignore_index=True) if carry is not None else chunk
        last = work.iloc[-1]
        tail_mask = (work.trip_id.astype(str) == str(last.trip_id)) & (work.observation_index.astype(int) == int(last.observation_index))
        add_observation_groups(work.loc[~tail_mask])
        carry = work.loc[tail_mask].copy()
    add_observation_groups(carry)

    feature_report = pd.DataFrame([
        {"split": split, "feature": name, "rows": row_count,
         "missing_count": values["missing"], "missing_percent": 100 * values["missing"] / max(row_count, 1),
         "minimum": values["min"] if np.isfinite(values["min"]) else np.nan,
         "maximum": values["max"] if np.isfinite(values["max"]) else np.nan,
         "negative_count": values["negative"]}
        for name, values in stats.items()
    ])
    trip_report = pd.DataFrame([
        {"split": split, "trip_id": trip_id, "observations": v[0],
         "observations_with_reference_candidate": v[1],
         "candidate_miss_rate": 1 - v[1] / v[0] if v[0] else np.nan,
         "mean_candidates_per_observation": float(np.mean(v[2])) if v[2] else np.nan,
         "p95_candidates_per_observation": float(np.quantile(v[2], .95)) if v[2] else np.nan}
        for trip_id, v in sorted(trip_observations.items())
    ])
    summary = {
        "split": split, "rows": row_count, "trip_count": len(trip_observations),
        "label_0_rows": label_counts[0], "label_1_rows": label_counts[1],
        "invalid_label_rows": invalid_labels, "label_arc_alignment_mismatches": label_alignment_mismatches,
        "missing_identifier_rows": missing_identifier_rows, "schema": "|".join(header),
    }
    return feature_report, trip_report, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-features", default="results/train_features_aligned.csv")
    parser.add_argument("--student-features", default="results/student_test_features_aligned.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = validate_raw_trips(args.data_root)
    raw.to_csv(out_dir / "dataset_validation_report.csv", index=False)
    feature_reports, trip_reports, summaries = [], [], []
    for split, path in (("train", args.train_features), ("student_test", args.student_features)):
        print(f"Validating {split} feature table: {path}")
        feature, trips, summary = validate_feature_table(path, split)
        feature_reports.append(feature)
        trip_reports.append(trips)
        summaries.append(summary)
    pd.concat(feature_reports, ignore_index=True).to_csv(out_dir / "feature_validation_report.csv", index=False)
    trips = pd.concat(trip_reports, ignore_index=True)
    for split, filename in (("train", "candidate_trip_validation_train.csv"), ("student_test", "candidate_trip_validation_student_test.csv")):
        trips[trips.split == split].to_csv(out_dir / filename, index=False)
    schemas_match = summaries[0]["schema"] == summaries[1]["schema"]
    pd.DataFrame(summaries).drop(columns="schema").assign(feature_schemas_match=schemas_match).to_csv(out_dir / "feature_table_validation_summary.csv", index=False)
    print(f"Raw trips validated: {len(raw)}; training feature rows: {summaries[0]['rows']:,}; student-test feature rows: {summaries[1]['rows']:,}")
    print(f"Feature schemas match: {schemas_match}; reports saved under {out_dir}")


if __name__ == "__main__":
    main()
