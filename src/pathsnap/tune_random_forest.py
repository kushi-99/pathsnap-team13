"""
tune_random_forest.py – PathSnap GPS Map-Matching
Author : Vijayeendra (Team 13)
Purpose: Bayesian Hyperparameter Optimisation for Random Forest using Optuna.
         Maximises validation Macro F1 on the strict trip-grouped split.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .features import FEATURE_COLUMNS
from .train_random_forest import evaluate_predictions, get_train_val_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(train_frame, val_frame, all_validation, n_trials=15, output_dir="results/models"):
    print("=" * 65)
    print("Bayesian Hyperparameter Optimisation for Random Forest - Vijayeendra")
    print(f"Executing {n_trials} Optuna Bayesian optimization trials...")
    print("=" * 65)

    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["label"].astype(int)
    X_val = val_frame[FEATURE_COLUMNS]
    y_val = val_frame["label"].astype(int).to_numpy()

    # Pre-impute to avoid redundant imputation during fast trials
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    trial_records = []

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 120, step=10)
        max_depth = trial.suggest_int("max_depth", 10, 22, step=2)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 16, step=2)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8, step=1)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            max_samples=0.25,  # fast bootstrap sampling for Bayesian trials
            random_state=42,
            n_jobs=-1,
        )

        t0_trial = time.perf_counter()
        clf.fit(X_train_imp, y_train)
        fit_duration = time.perf_counter() - t0_trial

        y_prob = clf.predict_proba(X_val_imp)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
        macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_val, y_pred))
        auc = float(roc_auc_score(y_val, y_prob))

        trial_records.append({
            "trial_number": trial.number,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "macro_f1": macro_f1,
            "accuracy": acc,
            "roc_auc": auc,
            "duration_s": round(fit_duration, 2),
        })

        print(f"Trial {trial.number:2d}: Macro F1 = {macro_f1:.4f} | Acc = {acc:.4f} | AUC = {auc:.4f} | {fit_duration:.1f}s | depth={max_depth}, n_est={n_estimators}")
        return macro_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t0_opt = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.perf_counter() - t0_opt

    best_params = study.best_params
    best_value = study.best_value
    print("\n" + "=" * 65)
    print(f"Bayesian Optimisation Completed in {total_opt_time:.1f}s")
    print(f"Best Validation Macro F1: {best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")
    print("=" * 65)

    # Save trial history and best hyperparameters
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_records).to_csv(results_dir / "bayesian_trials.csv", index=False)
    with open(results_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Train final tuned model using best hyperparameters on the full training split
    print("\nTraining final tuned Random Forest model using best hyperparameters...")
    tuned_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
        )
    )

    t0_final = time.perf_counter()
    tuned_pipe.fit(X_train, y_train)
    fit_time_final = time.perf_counter() - t0_final
    print(f"Final tuned model fitted in {fit_time_final:.2f}s")

    X_val_all = all_validation[FEATURE_COLUMNS]
    t0_inf = time.perf_counter()
    y_prob_all = tuned_pipe.predict_proba(X_val_all)[:, 1]
    y_prob_covered = tuned_pipe.predict_proba(val_frame[FEATURE_COLUMNS])[:, 1]
    inf_time_final = time.perf_counter() - t0_inf
    y_pred_covered = (y_prob_covered >= 0.5).astype(int)

    metrics = evaluate_predictions(
        y_val, y_pred_covered, y_prob_covered, y_prob_all,
        all_validation, val_frame, fit_time_final, inf_time_final,
        model_name="random_forest_tuned"
    )

    print("\n--- Validation Performance (Bayesian-Tuned Random Forest) ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    # Feature importances
    rf_best = tuned_pipe.named_steps["randomforestclassifier"]
    importances = pd.Series(rf_best.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nTuned Feature Importances:")
    print(importances.to_string())

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_pipe, out_p / "random_forest_tuned.joblib")
    importances.to_csv(results_dir / "feature_importance_tuned_rf.csv")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "random_forest_tuned_metrics.csv", index=False)
    print(f"\nTuned model and metrics saved successfully to {out_p} and {results_dir}")

    return tuned_pipe, metrics, importances, trial_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/train_features_aligned.csv")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--out-dir", default="results/models")
    args = parser.parse_args()

    train_f, val_f, all_val, val_trips = get_train_val_split(args.features)
    run_bayesian_optimization(train_f, val_f, all_val, n_trials=args.trials, output_dir=args.out_dir)


if __name__ == "__main__":
    main()
