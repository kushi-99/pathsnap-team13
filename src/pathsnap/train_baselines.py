import argparse
import json
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


def grouped_train_validation_split(frame, random_state=42):
    groups = frame["trip_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    return next(splitter.split(frame[FEATURE_COLUMNS], frame["label"], groups))


def top1_accuracy(model, frame):
    scored = frame[["trip_id", "observation_index", "candidate_arc_id", "reference_arc_id"]].copy()
    scored["score"] = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    top = scored.loc[
        scored.groupby(["trip_id", "observation_index"])["score"].idxmax()
    ]
    return float((top["candidate_arc_id"] == top["reference_arc_id"]).mean())


def tune_hyperparameters(eligible_frame, max_knn_rows, max_observations):
    """Tune on a trip-grouped inner split, leaving the final holdout untouched."""
    outer_train_idx, _ = grouped_train_validation_split(eligible_frame)
    outer_train = eligible_frame.iloc[outer_train_idx]
    inner_train_idx, inner_validation_idx = grouped_train_validation_split(
        outer_train, random_state=43
    )
    inner_train = outer_train.iloc[inner_train_idx]
    inner_validation = outer_train.iloc[inner_validation_idx]

    observations = inner_validation[
        ["trip_id", "observation_index"]
    ].drop_duplicates()
    if len(observations) > max_observations:
        observations = observations.sample(
            n=max_observations, random_state=42
        )
    tuning_frame = inner_validation.merge(
        observations, on=["trip_id", "observation_index"], how="inner"
    )

    results = []
    best = {}

    for regularization in (0.1, 1.0, 10.0):
        parameters = {"C": regularization}
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                C=regularization,
            ),
        )
        model.fit(inner_train[FEATURE_COLUMNS], inner_train["label"].astype(int))
        accuracy = top1_accuracy(model, tuning_frame)
        results.append({
            "model": "logistic_regression",
            "parameters": json.dumps(parameters, sort_keys=True),
            "top1_accuracy": accuracy,
            "training_rows": len(inner_train),
            "tuning_observations": len(observations),
        })
        if "logistic_regression" not in best or accuracy > best["logistic_regression"][0]:
            best["logistic_regression"] = (accuracy, parameters)

    knn_train = balanced_training_sample(inner_train, max_knn_rows)
    for neighbors in (3, 5, 9):
        for weights in ("uniform", "distance"):
            parameters = {"n_neighbors": neighbors, "weights": weights}
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                KNeighborsClassifier(
                    n_neighbors=neighbors, weights=weights, n_jobs=-1
                ),
            )
            model.fit(knn_train[FEATURE_COLUMNS], knn_train["label"].astype(int))
            accuracy = top1_accuracy(model, tuning_frame)
            results.append({
                "model": "knn",
                "parameters": json.dumps(parameters, sort_keys=True),
                "top1_accuracy": accuracy,
                "training_rows": len(knn_train),
                "tuning_observations": len(observations),
            })
            if "knn" not in best or accuracy > best["knn"][0]:
                best["knn"] = (accuracy, parameters)

    return best, pd.DataFrame(results)


def evaluate_model(name, model, eligible_frame, all_frame, output_dir,
                   max_knn_rows=None):
    """Use one trip-grouped split; score candidates and all route observations."""
    train_idx, validation_idx = grouped_train_validation_split(eligible_frame)

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
        "parameters": json.dumps(
            model.named_steps[
                "logisticregression" if name == "logistic_regression" else "kneighborsclassifier"
            ].get_params(),
            sort_keys=True,
        ),
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
    parser.add_argument("--max-knn-rows", type=int, default=25000)
    parser.add_argument("--tuning-knn-rows", type=int, default=25000)
    parser.add_argument("--tuning-observations", type=int, default=250)
    args = parser.parse_args()

    if (
        args.max_knn_rows < 10
        or args.tuning_knn_rows < 10
        or args.tuning_observations < 1
    ):
        parser.error(
            "KNN row limits must be at least 10 and tuning observations must be positive."
        )

    all_frame = pd.read_csv(args.features, dtype={"trip_id": str})
    has_positive = all_frame.groupby(
        ["trip_id", "observation_index"]
    )["label"].transform("max").eq(1)
    eligible_frame = all_frame.loc[has_positive].copy()
    eligible_frame["label"] = eligible_frame["label"].astype(int)

    best_parameters, tuning_results = tune_hyperparameters(
        eligible_frame, args.tuning_knn_rows, args.tuning_observations
    )
    tuning_path = Path("results/hyperparameter_tuning.csv")
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(tuning_path, index=False)
    print("Selected parameters (inner trip-grouped holdout):")
    for name, (accuracy, parameters) in best_parameters.items():
        print(f"  {name}: {parameters}, top-1={accuracy:.4f}")
    print(f"Saved tuning results to {tuning_path}")

    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            **best_parameters["logistic_regression"][1],
        ),
    )
    knn = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(
            n_jobs=-1, **best_parameters["knn"][1]
        ),
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
