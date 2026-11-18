"""
train_random_forest.py – PathSnap GPS Map-Matching
Author : Vijayeendra (Team 13)
Purpose: Train and evaluate baseline (untuned) Random Forest Classifier on common 6-feature schema.
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from .features import FEATURE_COLUMNS


def get_train_val_split(features_path="results/train_features_aligned.csv", test_size=0.2, random_state=42):
    """Load aligned features and create trip-grouped train/val split strictly matching baseline."""
    all_frame = pd.read_csv(features_path, dtype={"trip_id": str})
    for col in FEATURE_COLUMNS:
        all_frame[col] = pd.to_numeric(all_frame[col], errors="coerce")

    has_positive = all_frame.groupby(["trip_id", "observation_index"])["label"].transform("max").eq(1)
    eligible_frame = all_frame.loc[has_positive].copy()
    eligible_frame["label"] = eligible_frame["label"].astype(int)

    groups = eligible_frame["trip_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, validation_idx = next(
        splitter.split(eligible_frame[FEATURE_COLUMNS], eligible_frame["label"], groups)
    )

    train_frame = eligible_frame.iloc[train_idx].copy()
    val_frame = eligible_frame.iloc[validation_idx].copy()

    validation_trip_ids = sorted(val_frame["trip_id"].astype(str).unique().tolist())
    all_validation = all_frame[all_frame["trip_id"].astype(str).isin(validation_trip_ids)].copy()

    return train_frame, val_frame, all_validation, validation_trip_ids


def evaluate_predictions(y_true, y_pred, y_prob_covered, all_prob, all_validation, val_frame, fit_seconds, inf_seconds, model_name="random_forest"):
    """Compute comprehensive candidate classification and observation top-1 metrics."""
    scored = all_validation[[
        "trip_id", "observation_index", "candidate_arc_id", "reference_arc_id", "label"
    ]].copy()
    scored["score"] = all_prob
    top = scored.loc[scored.groupby(["trip_id", "observation_index"])["score"].idxmax()]
    all_obs = all_validation.groupby(["trip_id", "observation_index"])["label"].max()
    covered_count = int((all_obs == 1).sum())

    val_scored = val_frame[[
        "trip_id", "observation_index", "candidate_arc_id", "reference_arc_id"
    ]].copy()
    val_scored["score"] = y_prob_covered
    top_covered_idx = val_scored.groupby(["trip_id", "observation_index"])["score"].idxmax()
    covered_top = val_scored.loc[top_covered_idx]

    eps = 1e-15
    y_prob_clipped = np.clip(y_prob_covered, eps, 1.0 - eps)

    metrics = {
        "model": model_name,
        "candidate_accuracy": float(accuracy_score(y_true, y_pred)),
        "candidate_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "candidate_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "candidate_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "candidate_positive_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "candidate_roc_auc": float(roc_auc_score(y_true, y_prob_covered)),
        "candidate_log_loss": float(log_loss(y_true, y_prob_clipped, labels=[0, 1])),
        "observation_top1_accuracy_all": float((top["candidate_arc_id"].to_numpy() == top["reference_arc_id"].to_numpy()).mean()),
        "observation_top1_accuracy_covered": float((covered_top["candidate_arc_id"].to_numpy() == covered_top["reference_arc_id"].to_numpy()).mean()),
        "candidate_miss_rate": float((all_obs == 0).mean()),
        "fit_seconds": round(fit_seconds, 2),
        "inference_seconds": round(inf_seconds, 2),
    }
    return metrics


def train_untuned_random_forest(train_frame, val_frame, all_validation, output_dir="results/models"):
    """Train standard untuned baseline Random Forest."""
    print("=" * 65)
    print("Training Baseline 3: Random Forest (Untuned Baseline) - Vijayeendra")
    print("=" * 65)

    rf_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["label"].astype(int)
    X_val_all = all_validation[FEATURE_COLUMNS]

    print(f"Fitting Random Forest on {len(X_train):,} training candidate rows ...", flush=True)
    t0_fit = time.perf_counter()
    rf_pipe.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0_fit
    print(f"Random Forest fitted in {fit_sec:.2f}s", flush=True)

    print(f"Running inference on {len(X_val_all):,} validation candidate rows ...", flush=True)
    t0_inf = time.perf_counter()
    y_prob_all = rf_pipe.predict_proba(X_val_all)[:, 1]
    inf_sec = time.perf_counter() - t0_inf
    print(f"Inference completed in {inf_sec:.2f}s", flush=True)

    val_indices_in_all = all_validation.index.isin(val_frame.index)
    y_true_covered = val_frame["label"].astype(int).to_numpy()
    y_prob_covered = rf_pipe.predict_proba(val_frame[FEATURE_COLUMNS])[:, 1]
    y_pred_covered = (y_prob_covered >= 0.5).astype(int)

    metrics = evaluate_predictions(
        y_true_covered, y_pred_covered, y_prob_covered, y_prob_all,
        all_validation, val_frame, fit_sec, inf_sec,
        model_name="random_forest_untuned"
    )

    print("\n--- Validation Performance (Untuned Random Forest) ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    # Confusion matrix
    cm = confusion_matrix(y_true_covered, y_pred_covered, labels=[0, 1])
    print(f"\nConfusion Matrix on Covered Validation Candidates:\n{cm}")

    # Feature importances
    rf_model = rf_pipe.named_steps["randomforestclassifier"]
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string())

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_pipe, out_p / "random_forest_untuned.joblib")
    importances.to_csv(Path("results") / "feature_importance_untuned_rf.csv")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(Path("results") / "random_forest_untuned_metrics.csv", index=False)
    print(f"\nModel and metrics saved successfully to {out_p} and results/")

    return rf_pipe, metrics, importances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/train_features_aligned.csv")
    parser.add_argument("--out-dir", default="results/models")
    args = parser.parse_args()

    train_f, val_f, all_val, val_trips = get_train_val_split(args.features)
    print(f"Loaded train trips: {train_f['trip_id'].nunique()}, validation trips: {len(val_trips)}")
    train_untuned_random_forest(train_f, val_f, all_val, output_dir=args.out_dir)


if __name__ == "__main__":
    main()
