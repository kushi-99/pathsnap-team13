
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from skopt import Optimizer
from skopt.space import Integer, Real

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score


FEATURES = [
    "distance_to_road_m",
    "heading_difference_deg",
    "speed_mps",
    "time_gap_s",
    "incoming_degree",
    "outgoing_degree",
]

TARGET = "label"
GROUP = "trip_id"


def load_data(csv_path):
    print(f"Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = FEATURES + [TARGET, GROUP]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET, GROUP]).copy()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    groups = df.loc[valid, GROUP]

    if y.nunique() < 2:
        raise ValueError("Dataset must contain both classes (0 and 1).")

    print(f"Rows: {len(X):,}")
    print(f"Trips: {groups.nunique()}")

    return X, y, groups


def objective(params, X, y, groups, folds):
    max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization = params

    model = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_leaf_nodes=int(max_leaf_nodes),
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=float(l2_regularization),
        random_state=42,
    )

    cv = GroupKFold(n_splits=folds)

    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    mean_score = float(np.mean(scores))

    return mean_score, scores


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimisation for HistGradientBoosting"
    )

    parser.add_argument(
        "--csv",
        default="results/train_features.csv",
        help="Training feature CSV path",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=6,
        help="Number of Bayesian optimisation iterations",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of GroupKFold splits",
    )

    args = parser.parse_args()

    X, y, groups = load_data(args.csv)

    if groups.nunique() < args.folds:
        raise ValueError(
            f"Need at least {args.folds} different trips for GroupKFold."
        )

    print(f"Using {args.folds}-fold GroupKFold")
    print(f"Bayesian optimisation iterations: {args.iterations}")

    search_space = [
        Integer(80, 200, name="max_iter"),
        Real(0.03, 0.2, prior="log-uniform", name="learning_rate"),
        Integer(15, 63, name="max_leaf_nodes"),
        Integer(20, 100, name="min_samples_leaf"),
        Real(1e-4, 10.0, prior="log-uniform", name="l2_regularization"),
    ]

    optimizer = Optimizer(
        dimensions=search_space,
        random_state=42,
        n_initial_points=3,
        acq_func="EI",
    )

    best_score = -np.inf
    best_params = None
    search_results = []

    for iteration in range(args.iterations):
        print(f"\nIteration {iteration + 1}/{args.iterations}")

        params = optimizer.ask()

        param_names = [
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ]

        current_params = dict(zip(param_names, params))

        print(f"Parameters: {current_params}")

        mean_score, fold_scores = objective(
            params, X, y, groups, args.folds
        )

        print(f"Fold scores: {fold_scores}")
        print(f"Mean Macro F1: {mean_score:.4f}")

        optimizer.tell(params, -mean_score)

        search_results.append({
            **{
                key: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in current_params.items()
            },
            "mean_macro_f1": mean_score,
            "fold_scores": str(
                [float(score) for score in fold_scores]
            ),
        })

        if mean_score > best_score:
            best_score = mean_score
            best_params = current_params.copy()

    print("\nBest parameters:", best_params)
    print(f"Best CV Macro F1: {best_score:.4f}")

    # Convert NumPy values into standard Python values.
    best_params_json = {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in best_params.items()
    }

    print("\nTraining final tuned model on all training trips...")

    final_model = HistGradientBoostingClassifier(
        **best_params_json,
        random_state=42,
    )

    final_model.fit(X, y)

    output_dir = Path("results/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "hist_gradient_boosting_tuned.joblib"
    results_path = output_dir / "hgb_bayesian_search_results.csv"
    params_path = output_dir / "hgb_best_params.json"

    # Save the trained model.
    joblib.dump(final_model, model_path)

    # Save all optimisation iterations.
    pd.DataFrame(search_results).to_csv(
        results_path,
        index=False,
    )

    # Save best parameters and CV score.
    with open(params_path, "w") as f:
        json.dump(
            {
                "best_params": best_params_json,
                "best_cv_macro_f1": float(best_score),
                "iterations": int(args.iterations),
                "folds": int(args.folds),
            },
            f,
            indent=4,
        )

    print("\nTuning completed successfully!")
    print(f"Tuned model saved: {model_path}")
    print(f"Search results saved: {results_path}")
    print(f"Best parameters saved: {params_path}")


if __name__ == "__main__":
    main()
