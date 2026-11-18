import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from .features import FEATURE_COLUMNS


def evaluate(name, model, X, y, groups, out_dir):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    numeric = X[FEATURE_COLUMNS]
    model.fit(numeric.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(numeric.iloc[val_idx])
    prob = model.predict_proba(numeric.iloc[val_idx])[:, 1]
    metrics = {"model": name,
               "candidate_accuracy": accuracy_score(y.iloc[val_idx], pred),
               "candidate_macro_f1": f1_score(y.iloc[val_idx], pred, average="macro", zero_division=0),
               "candidate_log_loss": log_loss(y.iloc[val_idx], prob, labels=[0, 1]),
               "train_trips": int(groups.iloc[train_idx].nunique()),
               "validation_trips": int(groups.iloc[val_idx].nunique())}
    val = pd.DataFrame({"trip_id": groups.iloc[val_idx].to_numpy(),
                        "observation_index": X.iloc[val_idx]["observation_index"].to_numpy(),
                        "candidate_arc_id": X.iloc[val_idx]["candidate_arc_id"].to_numpy(),
                        "y": y.iloc[val_idx].to_numpy(), "score": prob})
    top = val.loc[val.groupby(["trip_id", "observation_index"]).score.idxmax()]
    # Top-1 candidate hit rate uses the positive candidate label, not test-set ground truth.
    metrics["observation_top1_accuracy"] = float(top.y.mean())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(out_dir) / f"{name}.joblib")
    return metrics


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/train_features.csv")
    parser.add_argument("--out-dir", default="artifacts/baselines")
    parser.add_argument("--max-knn-rows", type=int, default=100000)
    args = parser.parse_args()
    df = pd.read_csv(args.features)
    # Only observations with a generated positive candidate can train a candidate ranker.
    positive_obs = df.groupby(["trip_id", "observation_index"]).label.transform("max").eq(1)
    df = df.loc[positive_obs].copy()
    X = df[["trip_id", "observation_index", "candidate_arc_id"] + FEATURE_COLUMNS]
    y = df.label.astype(int)
    groups = df.trip_id.astype(str)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    results = [evaluate("logistic_regression", lr, X, y, groups, args.out_dir)]

    # Bound KNN memory/time on the first run while retaining both label classes.
    rng = np.random.default_rng(42)
    picked = []
    for cls in (0, 1):
        ids = np.flatnonzero(y.to_numpy() == cls)
        take = min(len(ids), args.max_knn_rows // 2)
        picked.extend(rng.choice(ids, size=take, replace=False).tolist())
    picked = np.array(sorted(picked), dtype=int)
    knn_X, knn_y, knn_groups = X.iloc[picked], y.iloc[picked], groups.iloc[picked]
    knn = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))
    results.append(evaluate("knn", knn, knn_X, knn_y, knn_groups, args.out_dir))
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv("results/baseline_metrics.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
    