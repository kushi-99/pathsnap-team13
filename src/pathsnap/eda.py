import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import list_trip_ids, load_trip
from .features import FEATURE_COLUMNS


def _read_feature_sample(path, max_rows=100_000, chunksize=100_000):
    sample = pd.DataFrame()
    labels = {0: 0, 1: 0}
    feature_missing = {name: 0 for name in FEATURE_COLUMNS}
    row_count = 0
    rng = np.random.default_rng(42)
    for chunk in pd.read_csv(path, dtype={"trip_id": str}, chunksize=chunksize):
        row_count += len(chunk)
        for name in FEATURE_COLUMNS:
            feature_missing[name] += int(chunk[name].isna().sum())
        for label, count in chunk.label.value_counts().items():
            if int(label) in labels:
                labels[int(label)] += int(count)
        # Keeping the smallest independent random keys gives each row the same
        # chance of selection while bounding the sample size across chunks.
        keyed = chunk[FEATURE_COLUMNS].assign(_sample_key=rng.random(len(chunk)))
        sample = pd.concat([sample, keyed], ignore_index=True).nsmallest(max_rows, "_sample_key")
    return sample.reindex(columns=FEATURE_COLUMNS), labels, feature_missing, row_count


def generate_figures(results_dir, train_features, student_features, baseline_path, figures_dir):
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    train_inv = pd.read_csv(results_dir / "eda_inventory.csv", dtype={"trip_id": str})
    test_inv = pd.read_csv(results_dir / "eda_student_test_inventory.csv", dtype={"trip_id": str})
    tr = _read_feature_sample(train_features)
    te = _read_feature_sample(student_features)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.size": 9})

    def save(name):
        plt.tight_layout()
        plt.savefig(figures_dir / name)
        plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train_inv.gps_observations, bins=18, alpha=.7, label="Training"); ax.hist(test_inv.gps_observations, bins=10, alpha=.7, label="Student test")
    ax.set_xlabel("GPS observations per trip"); ax.set_ylabel("Trips"); ax.set_title("GPS observations per trip"); ax.legend(); save("gps_observations_per_trip.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    train_trip_features = pd.read_csv(results_dir / "candidate_trip_validation_train.csv")
    test_trip_features = pd.read_csv(results_dir / "candidate_trip_validation_student_test.csv")
    ax.hist(train_trip_features.mean_candidates_per_observation, bins=24, alpha=.7, label="Training"); ax.hist(test_trip_features.mean_candidates_per_observation, bins=10, alpha=.7, label="Student test")
    ax.set_xlabel("Mean candidate roads per observation"); ax.set_ylabel("Trips"); ax.set_title("Candidate roads per GPS observation"); ax.legend(); save("candidates_per_observation.png")
    feature_axes = (
        ("Candidate road distance", "Distance (m)"),
        ("GPS-to-road heading difference", "Difference (degrees)"),
        ("GPS-derived speed", "Speed (m/s)"),
        ("Time between GPS fixes", "Time gap (seconds)"),
        ("Incoming degree", "Directed roads"),
        ("Outgoing degree", "Directed roads"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for ax, name, (title, xlabel) in zip(axes.flat, FEATURE_COLUMNS, feature_axes):
        values_by_split = []
        for sample in (tr[0], te[0]):
            values = sample[name].dropna().to_numpy()
            if name in ("speed_mps", "time_gap_s") and len(values):
                values = values[values <= np.quantile(values, .99)]
            values_by_split.append(values)

        combined = np.concatenate(values_by_split)
        if name.endswith("_degree"):
            bins = np.arange(-.5, int(combined.max(initial=0)) + 1.5)
        else:
            bins = np.histogram_bin_edges(combined, bins=35)
        for label, values in zip(("Training", "Student test"), values_by_split):
            if len(values):
                ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, label=label)
        ax.set(title=title, xlabel=xlabel, ylabel="Density")
    axes[0, 0].legend()
    save("feature_distributions.png")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(FEATURE_COLUMNS, [tr[2][x] for x in FEATURE_COLUMNS], color="#c76b52"); ax.tick_params(axis="x", rotation=35); ax.set_ylabel("Missing feature values"); ax.set_title("Training feature missing-value counts"); save("missing_values.png")
    corr = tr[0][FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(7, 6)); image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm"); ax.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right"); ax.set_yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)): ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson correlation"); ax.set_title("Feature correlation (bounded random sample)"); save("feature_correlation.png")
    trip_report = pd.read_csv(Path(results_dir) / "candidate_trip_validation_train.csv", dtype={"trip_id": str})
    trip_report = trip_report.sort_values("candidate_miss_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(trip_report.trip_id, trip_report.candidate_miss_rate * 100, color=["#b85450" if x in ("00000054", "00000033") else "#5788b5" for x in trip_report.trip_id]); ax.tick_params(axis="x", rotation=90); ax.set_ylabel("Observations without a positive candidate (%)"); ax.set_title("Training candidate miss rate by trip"); save("candidate_miss_rate_by_trip.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, inventory in (("Training", train_inv), ("Student test", test_inv)):
        ax.scatter(inventory.max_time_gap_s, inventory.gps_observations, alpha=.7, label=label)
    ax.set_xscale("log"); ax.set_xlabel("Maximum time gap (s, log scale)"); ax.set_ylabel("GPS observations in trip"); ax.set_title("Trip size and maximum timestamp gap"); ax.legend(); save("trip_data_quality.png")

    baseline = pd.read_csv(baseline_path)
    positions = np.arange(len(baseline))
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, column, label in (
        (-.18, "observation_top1_accuracy_all", "All observations"),
        (.18, "observation_top1_accuracy_when_reference_candidate_present", "Covered observations"),
    ):
        values = baseline[column] * 100
        bars = ax.bar(positions + offset, values, width=.36, label=label)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    model_names = {"logistic_regression": "Logistic Regression", "knn": "KNN"}
    ax.set_xticks(positions, [model_names.get(name, name) for name in baseline.model])
    ax.set(ylabel="Top-1 accuracy (%)", title="Baseline road selection on held-out trips", ylim=(0, 100))
    ax.legend()
    save("top1_accuracy.png")
    print(f"EDA figures saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--split",
        choices=["train", "student_test"],
        default="train",
    )
    parser.add_argument("--out", default="results/eda_inventory.csv")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--train-features", default="results/train_features_aligned.csv")
    parser.add_argument("--student-features", default="results/student_test_features_aligned.csv")
    parser.add_argument("--baseline-metrics", default="results/baseline_metrics.csv")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    rows = []

    for trip_id in list_trip_ids(args.data_root, args.split):
        trip = load_trip(args.data_root, trip_id, args.split)

        track = trip["track"]
        nodes = trip["nodes"]
        arcs = trip["arcs"]
        route = trip["route"]

        timestamps = track["timestamp_s"].to_numpy(dtype=float)
        time_gaps = np.diff(timestamps)

        valid_arc_endpoints = (
            (arcs["from_node_id"] >= 0)
            & (arcs["to_node_id"] >= 0)
            & (arcs["from_node_id"] < len(nodes))
            & (arcs["to_node_id"] < len(nodes))
        )

        invalid_coordinates = (
            ~track["longitude"].between(-180, 180)
            | ~track["latitude"].between(-90, 90)
        )

        rows.append({
            "split": args.split,
            "trip_id": trip_id,
            "gps_observations": len(track),
            "road_nodes": len(nodes),
            "road_arcs": len(arcs),
            "route_arcs": len(route),
            "missing_track_cells": int(track.isna().sum().sum()),
            "invalid_coordinates": int(invalid_coordinates.sum()),
            "timestamps_strictly_increasing": bool((time_gaps > 0).all()),
            "median_time_gap_s": (
                float(np.median(time_gaps)) if len(time_gaps) else np.nan
            ),
            "max_time_gap_s": (
                float(np.max(time_gaps)) if len(time_gaps) else np.nan
            ),
            "invalid_arc_endpoints": int((~valid_arc_endpoints).sum()),
            "min_longitude": float(track["longitude"].min()),
            "max_longitude": float(track["longitude"].max()),
            "min_latitude": float(track["latitude"].min()),
            "max_latitude": float(track["latitude"].max()),
        })

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Saved EDA inventory to {output_path}")
    if args.make_figures:
        generate_figures(
            output_path.parent, args.train_features, args.student_features,
            args.baseline_metrics, args.figures_dir,
        )


if __name__ == "__main__":
    main()
