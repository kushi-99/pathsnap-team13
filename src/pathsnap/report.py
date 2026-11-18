"""Render the data-foundation report from measured project outputs."""
import argparse
import json
from pathlib import Path

import pandas as pd


FEATURE_DETAILS = {
    "distance_to_road_m": ("Minimum projected distance from the GPS point to the candidate road segment.", "meters", "Candidate geometric fit; candidate search may omit the true road."),
    "heading_difference_deg": ("Smallest circular difference between previous-step GPS bearing and directed arc bearing.", "degrees", "Missing for the first or near-stationary GPS step; noisy at low speeds."),
    "speed_mps": ("Projected distance from the previous GPS point divided by a positive time gap.", "meters/second", "GPS noise or long time gaps can produce large values."),
    "time_gap_s": ("Current GPS timestamp minus the previous timestamp.", "seconds", "First observation is missing; timestamp gaps are retained."),
    "incoming_degree": ("Number of directed arcs ending at the candidate arc's start node.", "count", "Directed connectivity only; it does not encode road class."),
    "outgoing_degree": ("Number of directed arcs starting at the candidate arc's end node.", "count", "Directed connectivity only; it does not encode road class."),
}


def _metric_table(metrics):
    rows = ["| Model | Selected parameters | Candidate accuracy | Macro F1 | Log loss | Top-1 covered | Top-1 all | Fit (s) | Inference (s) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples():
        parameters = json.loads(getattr(row, "parameters", "{}"))
        if row.model == "logistic_regression":
            selected = f"C={parameters.get('C', 'unknown')}"
        else:
            selected = (
                f"k={parameters.get('n_neighbors', 'unknown')}, "
                f"weights={parameters.get('weights', 'unknown')}"
            )
        rows.append(
            f"| {row.model} | {selected} | "
            f"{row.candidate_accuracy_on_covered_observations:.4f} | "
            f"{row.candidate_macro_f1_on_covered_observations:.4f} | "
            f"{row.candidate_log_loss_on_covered_observations:.4f} | "
            f"{row.observation_top1_accuracy_when_reference_candidate_present:.4f} | "
            f"{row.observation_top1_accuracy_all:.4f} | {row.fit_seconds:.2f} | "
            f"{row.validation_inference_seconds:.2f} |"
        )
    return "\n".join(rows)


def build_report(results_dir):
    results_dir = Path(results_dir)
    raw = pd.read_csv(results_dir / "dataset_validation_report.csv")
    feature = pd.read_csv(results_dir / "feature_validation_report.csv")
    table_summary = pd.read_csv(results_dir / "feature_table_validation_summary.csv")
    baseline = pd.read_csv(results_dir / "baseline_metrics.csv")
    logistic_metrics = baseline.loc[baseline.model == "logistic_regression"].iloc[0]
    knn_metrics = baseline.loc[baseline.model == "knn"].iloc[0]
    logistic_parameters = json.loads(logistic_metrics.parameters)
    knn_parameters = json.loads(knn_metrics.parameters)
    knn_row_cap = int(knn_metrics.fit_rows)
    candidate_trips = pd.read_csv(results_dir / "candidate_trip_validation_train.csv", dtype={"trip_id": str})
    alignment_path = results_dir / "label_alignment_quality.csv"
    feature_lines = ["| Feature | Meaning | Unit | Train range | Train missing | Student-test missing | Negative values | Limitation |", "|---|---|---|---:|---:|---:|---:|---|"]
    for name, (meaning, unit, limitation) in FEATURE_DETAILS.items():
        train = feature[(feature.split == "train") & (feature.feature == name)].iloc[0]
        test = feature[(feature.split == "student_test") & (feature.feature == name)].iloc[0]
        feature_lines.append(
            f"| `{name}` | {meaning} | {unit} | {train.minimum:.6g} to {train.maximum:.6g} | "
            f"{int(train.missing_count):,} | {int(test.missing_count):,} | {int(train.negative_count):,} / {int(test.negative_count):,} | {limitation} |"
        )
    split_stats = raw.groupby("split").agg(
        trips=("trip_id", "nunique"), gps=("gps_observations", "sum"),
        nodes=("road_nodes", "sum"), arcs=("road_arcs", "sum"), routes=("route_arcs", "sum"),
        missing=("missing_track_cells", "sum"), bad_gps=("invalid_gps_coordinates", "sum"),
        bad_nodes=("invalid_node_coordinates", "sum"), bad_endpoints=("invalid_arc_endpoints", "sum"),
        nonincreasing=("nonincreasing_timestamp_steps", "sum"), max_gap=("max_time_gap_s", "max"),
    )
    train = split_stats.loc["train"]
    test = split_stats.loc["student_test"]
    train_table = table_summary[table_summary.split == "train"].iloc[0]
    test_table = table_summary[table_summary.split == "student_test"].iloc[0]
    outliers = candidate_trips[candidate_trips.trip_id.isin(["00000054", "00000033"])]
    if alignment_path.exists():
        alignment = pd.read_csv(alignment_path, dtype={"trip_id": str})
        outliers = outliers.merge(
            alignment.loc[alignment.split == "train", ["trip_id", "route_dist_median_m", "route_dist_p90_m"]],
            on="trip_id",
            how="left",
            validate="one_to_one",
        )
    outlier_lines = ["| Trip | Candidate miss rate | Median GPS-to-route distance (m) | P90 distance (m) |", "|---|---:|---:|---:|"]
    for row in outliers.sort_values("candidate_miss_rate", ascending=False).itertuples():
        distances = [getattr(row, name, None) for name in ("route_dist_median_m", "route_dist_p90_m")]
        median, p90 = (f"{value:.2f}" if pd.notna(value) else "—" for value in distances)
        outlier_lines.append(f"| {row.trip_id} | {row.candidate_miss_rate:.1%} | {median} | {p90} |")
    schema_match = bool(table_summary.feature_schemas_match.all())
    missing_features = feature[(feature.split == "train") & (feature.missing_count > 0)]
    missing_summary = ", ".join(f"`{r.feature}` {int(r.missing_count):,}" for r in missing_features.itertuples()) or "None"
    maps = {
        "gps_trajectory.html": "GPS trajectory",
        "road_network.html": "Nearby road network",
        "gps_network_overlay.html": "GPS and road-network overlay",
        "candidate_roads.html": "Candidate roads",
        "problematic_trip_00000054.html": "Trip 00000054",
        "problematic_trip_00000033.html": "Trip 00000033",
    }
    map_links = "\n".join(f"- [{title}](maps/{name})" for name, title in maps.items()
                         if (results_dir / "maps" / name).is_file())
    report = f'''# PathSnap Team 13 — Data Foundation and Feature Validation

## Executive summary

The data foundation loads 80 training trips and 10 student-test trips, builds candidate-road feature tables, validates source records and labels, and records Logistic Regression and KNN baselines. Training data contain {int(train.gps):,} GPS observations and {int(train_table.rows):,} candidate rows; student-test data contain {int(test.gps):,} observations and {int(test_table.rows):,} candidate rows. Candidate labels are inferred by aligning GPS observations to an ordered reference route. They are not independently verified pointwise ground truth.

## Objective and dataset structure

PathSnap scores likely road segments for noisy GPS observations before route reconstruction. Each trip has whitespace-separated `.track` (longitude, latitude, timestamp seconds), `.nodes` (longitude, latitude), `.arcs` (directed from/to node row indices), and `.route` (ordered arc IDs). Each arc ID is its row index in `.arcs`. The `train` and `student_test` folders are under `data/Student folder/`.

| Split | Trips | GPS observations | Road nodes | Directed arcs | Route arcs | Missing track cells | Invalid GPS / nodes / endpoints | Non-increasing timestamp steps | Largest gap (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | {int(train.trips)} | {int(train.gps):,} | {int(train.nodes):,} | {int(train.arcs):,} | {int(train.routes):,} | {int(train.missing)} | {int(train.bad_gps)} / {int(train.bad_nodes)} / {int(train.bad_endpoints)} | {int(train.nonincreasing)} | {train.max_gap:,.0f} |
| Student test | {int(test.trips)} | {int(test.gps):,} | {int(test.nodes):,} | {int(test.arcs):,} | {int(test.routes):,} | {int(test.missing)} | {int(test.bad_gps)} / {int(test.bad_nodes)} / {int(test.bad_endpoints)} | {int(test.nonincreasing)} | {test.max_gap:,.0f} |

Raw checks are recorded per trip in `dataset_validation_report.csv`. Feature-table schemas match between splits: {schema_match}. The train table has {int(train_table.label_1_rows):,} positive and {int(train_table.label_0_rows):,} negative candidate rows; the student-test table has {int(test_table.label_1_rows):,} positive and {int(test_table.label_0_rows):,} negative rows. Student-test rows were not used to fit or tune models.

## Candidate generation and preprocessing

Coordinates are projected to a local UTM zone. A KD-tree over road-segment midpoints retrieves up to 40 nearby arcs per GPS point. The pipeline retains candidates within 100 m or keeps the closest retrieved segment when all are outside that radius. The feature table has one row per `(trip_id, observation_index, candidate_arc_id)` and includes reference metadata for supervision and evaluation. The models use only the six features below. The fixed midpoint query approximates nearest segment search and can miss the reference road.

The six model inputs are:

{chr(10).join(feature_lines)}

Missing values are preserved in the feature tables and handled by median imputation inside each sklearn model pipeline. Imputers are fitted on the training partition only. Training missing counts are {missing_summary}. Negative feature counts are zero in both feature tables. The largest recorded time gap is {train.max_gap:,.0f} seconds in training and {test.max_gap:,.0f} seconds in student test; resulting speeds and time gaps are retained rather than silently clipped or dropped. Full minima, maxima, missing counts, and negative checks are in `feature_validation_report.csv`.

## Label alignment and data-quality caveats

The `.route` file provides an ordered road-arc sequence rather than pointwise labels. A forward-only dynamic program aligns each GPS observation to a route position using GPS-to-segment distance, monotonic route progression, and timestamp-aware advancement. Candidate label 1 means its arc equals the inferred reference arc for that observation; all other candidate arcs receive 0. Validation checks that each label agrees with the candidate/reference arc IDs in the row. This alignment is an inference and should not be described as verified ground truth.

Two training trips have unusually high candidate miss rates. The route-distance columns use the existing alignment audit where available.

{chr(10).join(outlier_lines)}

They remain in the data and are documented as source/label-quality limitations. Per-trip candidate coverage is in `candidate_trip_validation_train.csv` and `candidate_trip_validation_student_test.csv`.

## Baseline models and evaluation

Both models use the same six features, median imputation, and standard scaling. Logistic Regression uses class balancing with selected regularization C={logistic_parameters["C"]}. KNN uses {knn_parameters["n_neighbors"]} neighbors with {knn_parameters["weights"]} weighting and a reproducible balanced training sample capped at {knn_row_cap:,} candidate rows, so its fit data differ in size from Logistic Regression. The recorded evaluation uses a fixed 80/20 GroupShuffleSplit by trip (random state 42; 64 training trips and 16 validation trips). No trip appears in both partitions. Candidate classification metrics are computed on observations containing at least one inferred reference candidate. Covered-observation Top-1 is reported separately from all-observation Top-1, where missed reference candidates count as incorrect.

{_metric_table(baseline)}

Metrics and exact validation trip IDs are in [baseline_metrics.csv](baseline_metrics.csv). The tuning search is recorded in [hyperparameter_tuning.csv](hyperparameter_tuning.csv). Logistic Regression has higher all-observation Top-1 and lower log loss in this run. KNN has higher candidate Macro F1 and takes longer to score the holdout.

## EDA and maps

The feature distribution panels display speed and time gaps through the 99th percentile of each split's sample. The correlation chart uses a bounded random sample of training candidate rows.

- [GPS observations per trip](figures/gps_observations_per_trip.png)
- [Candidate roads per GPS observation](figures/candidates_per_observation.png)
- [Distributions of all six features](figures/feature_distributions.png)
- [Missing feature values](figures/missing_values.png)
- [Feature correlations](figures/feature_correlation.png)
- [Candidate miss rates by trip](figures/candidate_miss_rate_by_trip.png)
- [Timestamp gaps and trip size](figures/trip_data_quality.png)
- [Logistic Regression and KNN Top-1 accuracy](figures/top1_accuracy.png)

The network display is deterministically thinned to at most 1,500 arcs. All coordinates and road geometries come from the dataset. Green candidate segments indicate inferred positive labels and dashed orange segments indicate negative labels. Base tiles require internet access.

{map_links or "Generate the interactive maps with the command in the README."}

## Limitations and conclusion

The route-to-GPS labels are inferred; candidate generation may omit a reference arc; GPS gaps can create extreme speed values; only 80 labeled training trips are available; and the baseline comparison is one trip-grouped holdout. KNN's balanced {knn_row_cap:,}-row cap also means its training sample differs from Logistic Regression's. These limits should accompany performance claims. The validated feature schema, preserved missing values, trip-level checks, measured baselines, figures, and maps provide a reproducible foundation for the team's route-reconstruction and deployment work.
'''
    (results_dir / "data_foundation_report.md").write_text(report, encoding="utf-8")
    summary = f"""# Review 2 speaking summary — Kushi Niranjana

I validated {int(train.trips)} training trips and {int(test.trips)} student-test trips, with {int(train_table.rows):,} training and {int(test_table.rows):,} student-test candidate rows. The preprocessing pipeline generates nearby roads for each GPS fix and computes six distance, motion, and road-connectivity features. The model pipelines fit median imputation on their training rows to handle missing motion features.

Candidate labels come from inferred alignment to an ordered reference route. Trips 00000054 and 00000033 have unusually high candidate miss rates. I kept both trips in the dataset and documented them in the report and maps.

On the same trip-grouped 80/20 holdout, the models gave these results.

{_metric_table(baseline)}

[Technical report](data_foundation_report.md) · [Full metrics](baseline_metrics.csv) · [Validation summary](feature_table_validation_summary.csv)
"""
    (results_dir / "kushi_review2_executive_summary.md").write_text(summary, encoding="utf-8")
    print(f"Reports saved under {results_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    build_report(args.results_dir)


if __name__ == "__main__":
    main()
