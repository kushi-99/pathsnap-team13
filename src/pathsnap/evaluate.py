"""Evaluate saved models on the held-out evaluator_test split only."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .candidates import generate_candidates
from .data_loader import list_trip_ids, load_trip
from .features import FEATURE_COLUMNS, build_features
from .labels import assign_reference_labels


def evaluate_predictions(trip, labeled_features, scores, model_name):
    if "reference_arc_id" not in labeled_features:
        labeled_features = assign_reference_labels(
            trip["track"],
            trip["nodes"],
            trip["arcs"],
            trip["route"],
            labeled_features,
            trip_id=trip["trip_id"],
        )
    scored = labeled_features.copy()
    scored["score"] = scores
    top = scored.loc[
        scored.groupby("observation_index")["score"].idxmax()
    ].copy()
    top["correct"] = top.candidate_arc_id == top.reference_arc_id
    present = scored.groupby("observation_index").label.max().astype(bool)
    covered_top = top[top.observation_index.isin(present[present].index)]
    observations = len(trip["track"])
    covered_count = int(present.sum())
    correct_count = int(top.correct.sum())
    covered_correct = int(covered_top.correct.sum())
    return {
        "model": model_name,
        "trip_id": trip["trip_id"],
        "observations": observations,
        "candidate_count": len(scored),
        "reference_candidate_observations": covered_count,
        "candidate_coverage": covered_count / observations,
        "top1_accuracy_all": correct_count / observations,
        "top1_accuracy_when_candidate_present": (
            covered_correct / covered_count if covered_count else 0.0
        ),
        "mean_selected_score": float(top.score.mean()),
        "correct_top1_count": correct_count,
        "correct_top1_covered_count": covered_correct,
        "score_sum": float(top.score.sum()),
    }


def evaluate_evaluator_split(data_root, model_dir):
    trip_ids = list_trip_ids(data_root, "evaluator_test")
    if not trip_ids:
        raise FileNotFoundError("No trips found under data/evaluator_test.")
    models = {
        name: joblib.load(Path(model_dir) / f"{name}.joblib")
        for name in ("logistic_regression", "knn")
    }
    results = []
    for trip_id in trip_ids:
        trip = load_trip(data_root, trip_id, "evaluator_test")
        candidates = generate_candidates(
            trip["track"], trip["nodes"], trip["arcs"], trip_id=trip_id
        )
        features = build_features(
            trip["track"], trip["nodes"], trip["arcs"], candidates
        )
        labeled = assign_reference_labels(
            trip["track"], trip["nodes"], trip["arcs"], trip["route"],
            features, trip_id=trip_id,
        )
        for name, model in models.items():
            scores = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
            results.append(evaluate_predictions(trip, labeled, scores, name))
        print(f"Evaluated evaluator_test/{trip_id}")

    frame = pd.DataFrame(results)
    summaries = []
    for name, group in frame.groupby("model", sort=False):
        observations = int(group.observations.sum())
        covered = int(group.reference_candidate_observations.sum())
        summaries.append({
            "model": name,
            "trip_id": "ALL",
            "observations": observations,
            "candidate_count": int(group.candidate_count.sum()),
            "reference_candidate_observations": covered,
            "candidate_coverage": covered / observations,
            "top1_accuracy_all": group.correct_top1_count.sum() / observations,
            "top1_accuracy_when_candidate_present": (
                group.correct_top1_covered_count.sum() / covered if covered else 0.0
            ),
            "mean_selected_score": group.score_sum.sum() / observations,
            "correct_top1_count": int(group.correct_top1_count.sum()),
            "correct_top1_covered_count": int(group.correct_top1_covered_count.sum()),
            "score_sum": group.score_sum.sum(),
        })
    return pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model-dir", default="results/baseline_models")
    parser.add_argument("--out", default="results/evaluator_test_metrics.csv")
    args = parser.parse_args()
    results = evaluate_evaluator_split(args.data_root, args.model_dir)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results[results.trip_id == "ALL"].to_string(index=False))
    print(f"Saved evaluator-only metrics to {output}")


if __name__ == "__main__":
    main()
