import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def balanced_training_sample(frame, max_rows, random_state=42):
    """Cap KNN training size with equal samples from the two label classes."""
    if len(frame) <= max_rows:
        return frame

    rng = np.random.default_rng(random_state)
    chosen = []
    per_class = max_rows // 2
    labels = frame["label"].to_numpy()

    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        count = min(len(positions), per_class)
        if count:
            chosen.extend(rng.choice(positions, size=count, replace=False).tolist())

    return frame.iloc[np.sort(np.asarray(chosen, dtype=int))]


def evaluate_model(name, model, eligible_frame, all_frame, output_dir,
                   max_knn_rows=None):
    """Use one trip-grouped split; score candidates and all route observations."""
    groups = eligible_frame["trip_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, validation_idx = next(
        splitter.split(eligible_frame[FEATURE_COLUMNS], eligible_frame["label"], groups)
    )

    train_frame = eligible_frame.iloc[train_idx]
    validation_frame = eligible_frame.iloc[validation_idx]
    if max_knn_rows is not None:
        train_frame = balanced_training_sample(train_frame, max_knn_rows)

    fit_start = time.perf_counter()
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["label"].astype(int))
    fit_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    covered_prob = model.predict_proba(validation_frame[FEATURE_COLUMNS])[:, 1]
    covered_pred = (covered_prob >= 0.5).astype(int)
    y_true = validation_frame["label"].astype(int).to_numpy()

    validation_trip_ids = sorted(
        validation_frame["trip_id"].astype(str).unique().tolist()
    )
    all_validation = all_frame[
        all_frame["trip_id"].astype(str).isin(validation_trip_ids)
    ].copy()
    all_prob = model.predict_proba(all_validation[FEATURE_COLUMNS])[:, 1]
    inference_seconds = time.perf_counter() - infer_start

    scored = all_validation[[
        "trip_id", "observation_index", "candidate_arc_id", "reference_arc_id", "label"
    ]].copy()
    scored["score"] = all_prob
    top = scored.loc[
        scored.groupby(["trip_id", "observation_index"])["score"].idxmax()
    ]
    all_obs = all_validation.groupby(
        ["trip_id", "observation_index"]
    )["label"].max()
    covered_count = int((all_obs == 1).sum())

    covered_scored = validation_frame[[
        "trip_id", "observation_index", "candidate_arc_id", "reference_arc_id"
    ]].copy()
    covered_scored["score"] = covered_prob
    covered_top = covered_scored.loc[
        covered_scored.groupby(["trip_id", "observation_index"])["score"].idxmax()
    ]

    metrics = {
        "model": name,
        "candidate_accuracy_on_covered_observations": accuracy_score(y_true, covered_pred),
        "candidate_macro_f1_on_covered_observations": f1_score(
            y_true, covered_pred, average="macro", zero_division=0
        ),
        "candidate_log_loss_on_covered_observations": log_loss(
            y_true, covered_prob, labels=[0, 1]
        ),
        "observation_top1_accuracy_all": float(
            (top["candidate_arc_id"].to_numpy()
             == top["reference_arc_id"].to_numpy()).mean()
        ),
        "observation_top1_accuracy_when_reference_candidate_present": float(
            (covered_top["candidate_arc_id"].to_numpy()
             == covered_top["reference_arc_id"].to_numpy()).mean()
        ),
        "validation_candidate_miss_rate": float((all_obs == 0).mean()),
        "validation_observations": len(all_obs),
        "validation_observations_with_reference_candidate": covered_count,
        "fit_rows": len(train_frame),
        "fit_trips": int(train_frame["trip_id"].nunique()),
        "validation_trips": len(validation_trip_ids),
        "validation_trip_ids": ",".join(validation_trip_ids),
        "fit_seconds": fit_seconds,
        "validation_inference_seconds": inference_seconds,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"{name}.joblib")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/train_features_aligned.csv")
    parser.add_argument("--out-dir", default="results/baseline_models")
    parser.add_argument("--max-knn-rows", type=int, default=100000)
    args = parser.parse_args()

    all_frame = pd.read_csv(args.features, dtype={"trip_id": str})
    has_positive = all_frame.groupby(
        ["trip_id", "observation_index"]
    )["label"].transform("max").eq(1)
    eligible_frame = all_frame.loc[has_positive].copy()
    eligible_frame["label"] = eligible_frame["label"].astype(int)

    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    knn = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    )

    results = [
        evaluate_model("logistic_regression", logistic, eligible_frame, all_frame,
                       args.out_dir),
        evaluate_model("knn", knn, eligible_frame, all_frame, args.out_dir,
                       max_knn_rows=args.max_knn_rows),
    ]

    output_path = Path("results/baseline_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_path, index=False)
    print(result_frame.to_string(index=False))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
