"""Reproducible training pipeline for the PathSnap MLP baseline classifier.

Implements:
- Trip-aware GroupKFold (5-fold) cross-validation strictly on the 80 training trips.
- Per-fold StandardScaler fitting on training data only (zero leakage).
- Comparison of baseline MLP architectures (e.g., 128-64 vs 128-64-32 with ReLU + Dropout).
- Full epoch-wise training & validation loss/metric histories.
- Final model training on all 80 training trips (using CV-selected architecture & epoch budget)
  and persistence of all inference artifacts.

NEVER accesses `student_test` or `evaluator_test` during training, scaling, or architecture choice.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ml.mlp.config import (
    ARTIFACTS_DIR,
    CandidateConfig,
    DEFAULT_FEATURE_COLUMNS,
    MLPConfig,
    RESULTS_DIR,
    TRAIN_DIR,
    ensure_directories,
)
from ml.mlp.data_adapter import (
    build_or_load_partition_dataset,
    load_external_candidate_dataset,
)
from ml.mlp.metrics_utils import (
    compute_candidate_and_obs_metrics,
    compute_classifier_route_continuity,
)
from ml.mlp.model import PathSnapMLP, count_parameters, set_global_seed


def _train_one_epoch(
    model: PathSnapMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    rng: np.random.Generator,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Run one shuffled mini-batch epoch and return (mean_loss, macro_f1)."""
    model.train()
    n = len(x_train)
    perm = rng.permutation(n)
    total_loss = 0.0
    preds = np.empty(n, dtype=np.int64)
    targets = np.empty(n, dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = perm[start:end]
        xb = torch.from_numpy(x_train[idx]).to(device)
        yb = torch.from_numpy(y_train[idx]).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = end - start
        total_loss += float(loss.item()) * bs
        with torch.no_grad():
            preds[start:end] = (logits >= 0.0).cpu().numpy().astype(np.int64)
            targets[start:end] = y_train[idx].astype(np.int64)

    mean_loss = total_loss / max(n, 1)
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    return mean_loss, macro_f1


def _evaluate_epoch_val(
    model: PathSnapMLP,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_df: pd.DataFrame,
    val_total_obs: int,
    val_misses: int,
    criterion: nn.Module,
    batch_size: int = 8192,
    device: str = "cpu",
) -> Tuple[float, float, float, np.ndarray]:
    """Compute validation loss, macro-F1, observation Top-1 accuracy, and probabilities."""
    model.eval()
    n = len(x_val)
    total_loss = 0.0
    probs = np.empty(n, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x_val[start:end]).to(device)
            yb = torch.from_numpy(y_val[start:end]).to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * (end - start)
            probs[start:end] = torch.sigmoid(logits).cpu().numpy()

    mean_loss = total_loss / max(n, 1)
    metrics = compute_candidate_and_obs_metrics(
        val_df, probs, total_obs_count=val_total_obs, candidate_miss_count=val_misses
    )
    return mean_loss, float(metrics["macro_f1"]), float(metrics["obs_top1_accuracy"]), probs


def run_group_cv_experiment(
    train_df: pd.DataFrame,
    train_summary_df: pd.DataFrame,
    cfg: MLPConfig,
    partition_dir: Path,
    device: str = "cpu",
) -> Dict[str, object]:
    """Run leak-free GroupKFold CV for a given MLPConfig on the 80 training trips."""
    set_global_seed(cfg.random_seed)
    feature_cols = cfg.feature_columns

    X_all = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_all = train_df["label"].to_numpy(dtype=np.float32)
    groups = train_df["trip_id"].to_numpy()

    obs_by_trip = dict(zip(train_summary_df["trip_id"].astype(str), train_summary_df["num_observations"].astype(int)))
    miss_by_trip = dict(zip(train_summary_df["trip_id"].astype(str), train_summary_df["candidate_misses"].astype(int)))

    gkf = GroupKFold(n_splits=cfg.n_cv_folds)
    oof_probs = np.zeros(len(train_df), dtype=np.float64)
    fold_histories: List[Dict[str, object]] = []
    fold_metrics_list: List[Dict[str, object]] = []
    best_epochs: List[int] = []
    fold_assignments: Dict[str, int] = {}

    t_cv_start = time.perf_counter()
    total_infer_s = 0.0

    print(f"\n=== Running {cfg.n_cv_folds}-Fold Trip-Grouped CV: {cfg.experiment_name} {cfg.hidden_dims} ===")

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        tr_trips = set(groups[tr_idx].tolist())
        va_trips = set(groups[va_idx].tolist())
        assert tr_trips.isdisjoint(va_trips), "DATA LEAKAGE DETECTED: trip split across train and val!"
        for tid in va_trips:
            fold_assignments[str(tid)] = fold_idx

        # Fit scaler strictly on training fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_idx]).astype(np.float32)
        X_va = scaler.transform(X_all[va_idx]).astype(np.float32)
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        val_df_fold = train_df.iloc[va_idx].reset_index(drop=True)
        val_total_obs = sum(obs_by_trip[str(t)] for t in va_trips)
        val_misses = sum(miss_by_trip[str(t)] for t in va_trips)

        fold_seed = cfg.random_seed + fold_idx * 100
        set_global_seed(fold_seed)
        rng = np.random.default_rng(fold_seed)

        model = PathSnapMLP(
            input_dim=len(feature_cols),
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
        ).to(device)

        if cfg.pos_weight_mode == "balanced":
            n_pos = max(float(y_tr.sum()), 1.0)
            n_neg = float(len(y_tr) - n_pos)
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        best_val_loss = float("inf")
        best_epoch = 1
        best_state = None
        best_probs = None
        patience_counter = 0

        history = {
            "fold": fold_idx,
            "train_trips": len(tr_trips),
            "val_trips": len(va_trips),
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "val_top1_acc": [],
        }

        t_fold_start = time.perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            tr_loss, tr_f1 = _train_one_epoch(
                model, X_tr, y_tr, optimizer, criterion, cfg.batch_size, rng, device=device
            )
            va_loss, va_f1, va_top1, va_probs = _evaluate_epoch_val(
                model, X_va, y_va, val_df_fold, val_total_obs, val_misses, criterion, device=device
            )

            history["epochs"].append(epoch)
            history["train_loss"].append(round(tr_loss, 6))
            history["val_loss"].append(round(va_loss, 6))
            history["train_macro_f1"].append(round(tr_f1, 6))
            history["val_macro_f1"].append(round(va_f1, 6))
            history["val_top1_acc"].append(round(va_top1, 6))

            if va_loss < best_val_loss - 1e-5:
                best_val_loss = va_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_probs = va_probs.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    break

        fold_train_s = time.perf_counter() - t_fold_start
        model.load_state_dict(best_state)

        # Measure clean inference time on validation fold
        t_inf0 = time.perf_counter()
        _ = model.predict_proba_numpy(X_va, batch_size=8192, device=device)
        fold_infer_s = time.perf_counter() - t_inf0
        total_infer_s += fold_infer_s

        oof_probs[va_idx] = best_probs
        best_epochs.append(best_epoch)

        fold_metrics = compute_candidate_and_obs_metrics(
            val_df_fold, best_probs, total_obs_count=val_total_obs, candidate_miss_count=val_misses
        )
        fold_metrics["fold"] = fold_idx
        fold_metrics["best_epoch"] = best_epoch
        fold_metrics["train_time_s"] = round(fold_train_s, 3)
        fold_metrics["infer_time_s"] = round(fold_infer_s, 4)
        fold_metrics_list.append(fold_metrics)
        fold_histories.append(history)

        print(
            f"  Fold {fold_idx}/{cfg.n_cv_folds} (best_ep={best_epoch:2d}, {fold_train_s:.1f}s) | "
            f"val_loss={best_val_loss:.4f} | Macro-F1={fold_metrics['macro_f1']:.4f} | "
            f"Top-1={fold_metrics['obs_top1_accuracy']:.4f} (cov={fold_metrics['obs_top1_accuracy_covered']:.4f})"
        )

    total_cv_train_s = time.perf_counter() - t_cv_start

    # Aggregate OOF metrics across all 80 training trips
    tot_obs = int(train_summary_df["num_observations"].sum())
    tot_miss = int(train_summary_df["candidate_misses"].sum())
    oof_metrics = compute_candidate_and_obs_metrics(
        train_df, oof_probs, total_obs_count=tot_obs, candidate_miss_count=tot_miss
    )
    continuity_agg, _ = compute_classifier_route_continuity(train_df, oof_probs, partition_dir)

    # Fold-level std for stability reporting
    f1_folds = [m["macro_f1"] for m in fold_metrics_list]
    top1_folds = [m["obs_top1_accuracy"] for m in fold_metrics_list]
    ll_folds = [m["log_loss"] for m in fold_metrics_list]

    n_params = count_parameters(
        PathSnapMLP(
            input_dim=len(feature_cols),
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
        )
    )

    summary = {
        "experiment_name": cfg.experiment_name,
        "hidden_dims": list(cfg.hidden_dims),
        "dropout": cfg.dropout,
        "use_batch_norm": cfg.use_batch_norm,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "num_parameters": n_params,
        "mean_best_epoch": int(round(float(np.mean(best_epochs)))),
        "best_epochs_per_fold": best_epochs,
        "cv_train_time_s": round(total_cv_train_s, 3),
        "cv_infer_time_s": round(total_infer_s, 4),
        "infer_ms_per_1k_candidates": round(total_infer_s / max(len(train_df), 1) * 1e6, 4),
        "oof_metrics": oof_metrics,
        "oof_route_continuity": continuity_agg,
        "fold_macro_f1_mean": float(np.mean(f1_folds)),
        "fold_macro_f1_std": float(np.std(f1_folds, ddof=1)),
        "fold_top1_mean": float(np.mean(top1_folds)),
        "fold_top1_std": float(np.std(top1_folds, ddof=1)),
        "fold_log_loss_mean": float(np.mean(ll_folds)),
        "fold_log_loss_std": float(np.std(ll_folds, ddof=1)),
        "fold_metrics": fold_metrics_list,
        "fold_histories": fold_histories,
        "fold_assignments": fold_assignments,
    }
    return {"summary": summary, "oof_probs": oof_probs}


def train_final_model_on_all_train_trips(
    train_df: pd.DataFrame,
    cfg: MLPConfig,
    num_epochs: int,
    device: str = "cpu",
) -> Tuple[PathSnapMLP, StandardScaler, List[Dict[str, float]], float]:
    """Fit scaler and MLP on all 80 permitted training trips for `num_epochs`."""
    set_global_seed(cfg.random_seed)
    rng = np.random.default_rng(cfg.random_seed)
    feature_cols = cfg.feature_columns

    X_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_all = train_df["label"].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    model = PathSnapMLP(
        input_dim=len(feature_cols),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    full_history: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    print(f"\n=== Training Final Model ({cfg.experiment_name}) on all 80 train trips for {num_epochs} epochs ===")
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_f1 = _train_one_epoch(
            model, X_scaled, y_all, optimizer, criterion, cfg.batch_size, rng, device=device
        )
        full_history.append({"epoch": epoch, "train_loss": round(tr_loss, 6), "train_macro_f1": round(tr_f1, 6)})
        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(f"  Epoch {epoch:2d}/{num_epochs} | train_loss={tr_loss:.4f} | train_macro_f1={tr_f1:.4f}")

    train_time_s = time.perf_counter() - t0
    return model, scaler, full_history, train_time_s


def save_artifacts(
    model: PathSnapMLP,
    scaler: StandardScaler,
    cfg: MLPConfig,
    cand_cfg: CandidateConfig,
    final_epochs: int,
    final_train_time_s: float,
    cv_summary: Dict[str, object],
    full_train_history: List[Dict[str, float]],
) -> None:
    """Persist all trained MLP artifacts cleanly into `ml/mlp/artifacts/`."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "hidden_dims": list(model.hidden_dims),
            "dropout": model.dropout,
            "use_batch_norm": model.use_batch_norm,
        },
        ARTIFACTS_DIR / "mlp_model.pt",
    )
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    (ARTIFACTS_DIR / "feature_columns.json").write_text(
        json.dumps({"feature_columns": cfg.feature_columns}, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "label_mapping.json").write_text(
        json.dumps({"0": "non_match", "1": "match", "positive_class": 1}, indent=2),
        encoding="utf-8",
    )

    model_cfg_payload = {
        "mlp_config": cfg.to_dict(),
        "candidate_config": cand_cfg.__dict__,
        "num_parameters": count_parameters(model),
        "final_trained_epochs": final_epochs,
        "random_seed": cfg.random_seed,
    }
    (ARTIFACTS_DIR / "model_config.json").write_text(
        json.dumps(model_cfg_payload, indent=2), encoding="utf-8"
    )

    training_meta = {
        "selected_experiment": cfg.experiment_name,
        "final_train_time_s": round(final_train_time_s, 3),
        "cv_train_time_s": cv_summary["cv_train_time_s"],
        "cv_infer_time_s": cv_summary["cv_infer_time_s"],
        "infer_ms_per_1k_candidates": cv_summary["infer_ms_per_1k_candidates"],
        "cv_oof_macro_f1": cv_summary["oof_metrics"]["macro_f1"],
        "cv_oof_top1_accuracy": cv_summary["oof_metrics"]["obs_top1_accuracy"],
        "cv_oof_log_loss": cv_summary["oof_metrics"]["log_loss"],
        "full_train_history": full_train_history,
        "fold_assignments": cv_summary["fold_assignments"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (ARTIFACTS_DIR / "training_metadata.json").write_text(
        json.dumps(training_meta, indent=2), encoding="utf-8"
    )
    print(f"\nSaved all model artifacts to: {ARTIFACTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PathSnap MLP baseline with GroupKFold CV.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional external team candidate dataset.")
    parser.add_argument("--epochs", type=int, default=25, help="Max epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    args = parser.parse_args()

    ensure_directories()
    cand_cfg = CandidateConfig()

    if args.dataset_path:
        train_df = load_external_candidate_dataset(Path(args.dataset_path))
        feature_cols = [c for c in train_df.columns if c not in ("trip_id", "obs_idx", "arc_id", "label")]
        # Infer trip observation counts from external dataset
        grp = train_df.groupby(["trip_id", "obs_idx"])["label"].max().reset_index()
        train_summary_df = (
            grp.groupby("trip_id")
            .agg(num_observations=("label", "count"), candidate_hits=("label", "sum"))
            .reset_index()
        )
        train_summary_df["candidate_misses"] = (
            train_summary_df["num_observations"] - train_summary_df["candidate_hits"]
        )
    else:
        train_df, train_summary_df = build_or_load_partition_dataset(
            TRAIN_DIR, "train", cfg=cand_cfg, use_cache=True, load_labels=True
        )
        feature_cols = list(DEFAULT_FEATURE_COLUMNS)

    # Experiment with two well-structured baseline MLP architectures on 80-trip GroupKFold CV
    candidate_configs = [
        MLPConfig(
            experiment_name="mlp_128_64",
            hidden_dims=(128, 64),
            dropout=0.20,
            use_batch_norm=True,
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping_patience=5,
            n_cv_folds=5,
            random_seed=args.seed,
            feature_columns=feature_cols,
        ),
        MLPConfig(
            experiment_name="mlp_128_64_32",
            hidden_dims=(128, 64, 32),
            dropout=0.20,
            use_batch_norm=True,
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping_patience=5,
            n_cv_folds=5,
            random_seed=args.seed,
            feature_columns=feature_cols,
        ),
    ]

    all_experiments: List[Dict[str, object]] = []
    best_exp = None
    best_cfg = None
    best_score = -1.0

    for cfg in candidate_configs:
        res = run_group_cv_experiment(train_df, train_summary_df, cfg, TRAIN_DIR, device="cpu")
        all_experiments.append(res["summary"])
        # Select strictly on CV (Top-1 accuracy primary, Macro-F1 tie-breaker)
        score = float(res["summary"]["oof_metrics"]["obs_top1_accuracy"]) + 0.01 * float(
            res["summary"]["oof_metrics"]["macro_f1"]
        )
        if score > best_score:
            best_score = score
            best_exp = res
            best_cfg = cfg

    # Save architecture comparison table
    comp_rows = []
    for s in all_experiments:
        m = s["oof_metrics"]
        c = s["oof_route_continuity"]
        comp_rows.append(
            {
                "experiment": s["experiment_name"],
                "hidden_dims": "-".join(map(str, s["hidden_dims"])),
                "dropout": s["dropout"],
                "params": s["num_parameters"],
                "cv_macro_f1": round(m["macro_f1"], 4),
                "cv_macro_f1_std": round(s["fold_macro_f1_std"], 4),
                "cv_top1_acc": round(m["obs_top1_accuracy"], 4),
                "cv_top1_acc_covered": round(m["obs_top1_accuracy_covered"], 4),
                "cv_log_loss": round(m["log_loss"], 4),
                "cv_obs_continuity": round(c["obs_step_continuity_overall"], 4),
                "cv_comp_continuity": round(c["compressed_route_continuity_overall"], 4),
                "cv_train_time_s": s["cv_train_time_s"],
                "infer_ms_per_1k": s["infer_ms_per_1k_candidates"],
                "mean_best_epoch": s["mean_best_epoch"],
            }
        )
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(RESULTS_DIR / "cv_architecture_comparison.csv", index=False)
    (RESULTS_DIR / "cv_architecture_comparison.json").write_text(
        json.dumps(all_experiments, indent=2), encoding="utf-8"
    )

    print("\n=== CV ARCHITECTURE COMPARISON (80 Training Trips, 5-Fold GroupKFold) ===")
    print(comp_df.to_string(index=False))
    print(f"\nSelected baseline MLP architecture (strictly from CV): {best_cfg.experiment_name}")

    # Save OOF predictions & fold histories for selected architecture
    (RESULTS_DIR / "cv_selected_summary.json").write_text(
        json.dumps(best_exp["summary"], indent=2), encoding="utf-8"
    )
    oof_out = train_df[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    oof_out["prob"] = best_exp["oof_probs"]
    oof_out.to_pickle(RESULTS_DIR / "cv_oof_predictions.pkl")

    # Refit final model on all 80 training trips using CV mean best epoch
    final_epochs = max(5, int(best_exp["summary"]["mean_best_epoch"]))
    final_model, final_scaler, full_hist, final_train_s = train_final_model_on_all_train_trips(
        train_df, best_cfg, num_epochs=final_epochs, device="cpu"
    )

    save_artifacts(
        model=final_model,
        scaler=final_scaler,
        cfg=best_cfg,
        cand_cfg=cand_cfg,
        final_epochs=final_epochs,
        final_train_time_s=final_train_s,
        cv_summary=best_exp["summary"],
        full_train_history=full_hist,
    )


if __name__ == "__main__":
    main()
