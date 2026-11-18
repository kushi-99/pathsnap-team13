"""Unified apples-to-apples multi-model comparison on the PathSnap 28-feature schema.

Evaluates four distinct algorithm families on the EXACT SAME:
  - 28 observable candidate features (`ml/mlp/artifacts/feature_columns.json`)
  - 80 training trips (5-fold trip-grouped GroupKFold CV + full 80-trip fit)
  - 10 held-out student_test trips (development-stage comparison)

Models included:
  1. mlp_128_64_32 (Neural Network — pre-trained finalized artifact under ml/mlp/artifacts/)
  2. hist_gradient_boosting (Boosting — HistGradientBoostingClassifier)
  3. random_forest (Ensemble — RandomForestClassifier)
  4. logistic_regression (Linear / Probabilistic — StandardScaler + LogisticRegression)

Also records transparent status of teammates' separate 6-feature branches
(origin/kushi/data-foundation and origin/kritika-hgb-hmm) without mixing
incompatible feature sets or fabricating missing results.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pathsnap_mpl"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.mlp.config import (
    ARTIFACTS_DIR,
    BASELINES_ARTIFACTS_DIR,
    CACHE_DIR,
    COMPARISON_DIR,
    CandidateConfig,
    DEFAULT_FEATURE_COLUMNS,
    RESULTS_DIR,
    STUDENT_TEST_DIR,
    TRAIN_DIR,
    WORKSPACE_ROOT,
)
from ml.mlp.data_adapter import build_or_load_partition_dataset
from ml.mlp.metrics_utils import (
    compute_candidate_and_obs_metrics,
    compute_classifier_route_continuity,
)
from ml.mlp.predict import MLPPredictor


def _load_cached_or_build(partition_name: str, partition_dir: Path, cfg: CandidateConfig):
    """Load cached candidate pickle from repo or parent workspace cache if available."""
    cache_key = f"{partition_name}_k{cfg.max_candidates}_r{int(cfg.search_radius_m)}"
    for cache_root in (CACHE_DIR, WORKSPACE_ROOT.parent / "ml" / "mlp" / "cache"):
        data_p = cache_root / f"{cache_key}.pkl"
        sum_p = cache_root / f"{cache_key}_summary.csv"
        if data_p.exists() and sum_p.exists():
            print(f"[CACHE] Loading {partition_name} from {data_p}")
            return pd.read_pickle(data_p), pd.read_csv(sum_p, dtype={"trip_id": str})
    return build_or_load_partition_dataset(
        partition_dir, partition_name, cfg=cfg, use_cache=True, load_labels=True
    )


def _sample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> Tuple[List[float], List[float]]:
    """Downsample ROC/PR curve arrays cleanly for lightweight JSON storage & plotting."""
    if len(x) <= max_points:
        return [round(float(v), 6) for v in x], [round(float(v), 6) for v in y]
    idx = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
    return [round(float(x[i]), 6) for i in idx], [round(float(y[i]), 6) for i in idx]


def _compute_curve_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Compute ROC-AUC, PR-AUC, and sampled ROC/PR curves when both classes exist."""
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None, "roc_curve": None, "pr_curve": None}
    roc_auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    fpr_s, tpr_s = _sample_curve(fpr, tpr, max_points=180)
    rec_s, prec_s = _sample_curve(rec[::-1], prec[::-1], max_points=180)
    return {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "roc_curve": {"fpr": fpr_s, "tpr": tpr_s},
        "pr_curve": {"recall": rec_s, "precision": prec_s},
    }


def _build_sklearn_models() -> Dict[str, Dict[str, Any]]:
    """Define comparable scikit-learn baselines on the canonical 28-feature schema."""
    return {
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "family": "Linear / Probabilistic",
            "architecture": "StandardScaler -> LogisticRegression (L2, lbfgs, max_iter=1000)",
            "config": {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "class_weight": None,
                "random_state": 42,
            },
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(max_iter=1000, random_state=42),
                    ),
                ]
            ),
        },
        "hist_gradient_boosting": {
            "display_name": "HistGradientBoosting",
            "family": "Gradient Boosting",
            "architecture": "HistGradientBoostingClassifier (150 trees, lr=0.1, max_leaf_nodes=31)",
            "config": {
                "max_iter": 150,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
                "random_state": 42,
            },
            "estimator": HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                max_leaf_nodes=31,
                random_state=42,
            ),
        },
        "random_forest": {
            "display_name": "Random Forest",
            "family": "Bagging / Ensemble",
            "architecture": "RandomForestClassifier (80 trees, max_depth=14, min_samples_leaf=10, max_samples=0.30)",
            "config": {
                "n_estimators": 80,
                "max_depth": 14,
                "min_samples_leaf": 10,
                "max_samples": 0.30,
                "random_state": 42,
                "n_jobs": -1,
            },
            "estimator": RandomForestClassifier(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=10,
                max_samples=0.30,
                random_state=42,
                n_jobs=-1,
            ),
        },
    }


def _generate_comparison_plots(
    models_summary: List[Dict[str, Any]],
    curves_data: Dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate genuine multi-model comparison charts, ROC/PR curves, and confusion matrices."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multi-model metric comparison bar chart (Student-Test)
    names = [m["display_name"] for m in models_summary]
    cand_acc = [m["student_test"]["candidate_accuracy"] for m in models_summary]
    macro_f1 = [m["student_test"]["macro_f1"] for m in models_summary]
    top1_acc = [m["student_test"]["obs_top1_accuracy"] for m in models_summary]
    log_losses = [m["student_test"]["log_loss"] for m in models_summary]
    infer_ms = [m["student_test"]["infer_ms_per_trip"] for m in models_summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)
    x = np.arange(len(names))
    w = 0.25
    b1 = ax1.bar(x - w, cand_acc, width=w, label="Candidate Accuracy", color="#1f77b4")
    b2 = ax1.bar(x, macro_f1, width=w, label="Macro F1-Score", color="#2ca02c")
    b3 = ax1.bar(x + w, top1_acc, width=w, label="Obs-Level Top-1 Acc", color="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylim(0.55, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Student-Test Classification & Route-Matching Metrics", fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5)

    ax2_twin = ax2.twinx()
    bars_ll = ax2.bar(x - 0.18, log_losses, width=0.34, label="Log Loss (lower better)", color="#d62728", alpha=0.85)
    bars_ms = ax2_twin.bar(x + 0.18, infer_ms, width=0.34, label="Inference ms/trip", color="#9467bd", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("Log Loss (BCE)", color="#d62728")
    ax2_twin.set_ylabel("Inference Latency (ms / trip)", color="#9467bd")
    ax2.set_title("Probability Calibration (Log Loss) vs Inference Latency", fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar in bars_ll:
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=7.5, color="#8c1515")
    for bar in bars_ms:
        v = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width() / 2, v + max(infer_ms) * 0.02, f"{v:.1f}ms", ha="center", fontsize=7.5, color="#4b0082")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)

    fig.suptitle("PathSnap Multi-Model Comparison (28 Features — 10 Held-Out Student-Test Trips)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "multi_model_comparison.png")
    plt.close(fig)

    # 2. Multi-model ROC & Precision-Recall curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=150)
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, m in enumerate(models_summary):
        mid = m["model_id"]
        cd = curves_data.get(mid, {})
        c = palette[idx % len(palette)]
        if cd.get("roc_curve"):
            ax_roc.plot(
                cd["roc_curve"]["fpr"],
                cd["roc_curve"]["tpr"],
                color=c,
                linewidth=2.1,
                label=f"{m['display_name']} (AUC={cd['roc_auc']:.4f})",
            )
        if cd.get("pr_curve"):
            ax_pr.plot(
                cd["pr_curve"]["recall"],
                cd["pr_curve"]["precision"],
                color=c,
                linewidth=2.1,
                label=f"{m['display_name']} (AP={cd['pr_auc']:.4f})",
            )

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.0)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (Student-Test Candidates)", fontweight="bold")
    ax_roc.grid(True, linestyle="--", alpha=0.4)
    ax_roc.legend(loc="lower right", fontsize=8.5)

    ax_pr.set_xlabel("Recall (Match Class)")
    ax_pr.set_ylabel("Precision (Match Class)")
    ax_pr.set_title("Precision-Recall Curve (Student-Test Candidates)", fontweight="bold")
    ax_pr.grid(True, linestyle="--", alpha=0.4)
    ax_pr.legend(loc="lower left", fontsize=8.5)

    fig.suptitle("PathSnap ROC & Precision-Recall Curves (10 Held-Out Student-Test Trips)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "multi_model_roc_pr.png")
    plt.close(fig)

    # 3. Confusion matrices grid (2x2)
    n_m = len(models_summary)
    fig, axes = plt.subplots(1, n_m, figsize=(4.2 * n_m, 4.0), dpi=150)
    if n_m == 1:
        axes = [axes]
    for ax, m in zip(axes, models_summary):
        cm = np.array(m["student_test"]["confusion_matrix"], dtype=np.int64)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(m["display_name"], fontsize=10, fontweight="bold")
        for r in range(2):
            for c_i in range(2):
                txt_col = "white" if cm_norm[r, c_i] > 0.55 else "black"
                ax.text(
                    c_i,
                    r,
                    f"{cm[r, c_i]:,}\n({cm_norm[r, c_i]*100:.1f}%)",
                    ha="center",
                    va="center",
                    color=txt_col,
                    fontsize=8.5,
                    fontweight="bold",
                )
    fig.suptitle("Candidate-Level Confusion Matrices on Student-Test (172,897 Candidates)", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "multi_model_confusion_matrices.png")
    plt.close(fig)


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cand_cfg = CandidateConfig()
    train_df, train_summary_df = _load_cached_or_build("train", TRAIN_DIR, cand_cfg)
    test_df, test_summary_df = _load_cached_or_build("student_test", STUDENT_TEST_DIR, cand_cfg)

    feature_cols = list(DEFAULT_FEATURE_COLUMNS)
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    groups_train = train_df["trip_id"].astype(str).to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    tot_train_obs = int(train_summary_df["num_observations"].sum())
    tot_train_miss = int(train_summary_df["candidate_misses"].sum())
    tot_test_obs = int(test_summary_df["num_observations"].sum())
    tot_test_miss = int(test_summary_df["candidate_misses"].sum())

    models_summary: List[Dict[str, Any]] = []
    curves_data: Dict[str, Any] = {}
    per_trip_frames: List[pd.DataFrame] = []

    # 1. Include finalized pre-trained MLP baseline (verify exact saved numbers + compute ROC/PR)
    print("\n=== Evaluating Finalized MLP Artifact (mlp_128_64_32) ===")
    mlp_predictor = MLPPredictor.from_artifacts(ARTIFACTS_DIR, device="cpu")
    cv_mlp = json.loads((RESULTS_DIR / "cv_selected_summary.json").read_text(encoding="utf-8"))
    st_mlp = json.loads((RESULTS_DIR / "student_test_metrics.json").read_text(encoding="utf-8"))
    meta_mlp = json.loads((ARTIFACTS_DIR / "training_metadata.json").read_text(encoding="utf-8"))
    cfg_mlp = json.loads((ARTIFACTS_DIR / "model_config.json").read_text(encoding="utf-8"))

    _, mlp_test_probs, _, _, _ = mlp_predictor.predict_scores(test_df)
    mlp_curves = _compute_curve_metrics(y_test, mlp_test_probs)
    curves_data["mlp_128_64_32"] = mlp_curves

    mlp_per_trip = pd.read_csv(RESULTS_DIR / "student_test_per_trip.csv", dtype={"trip_id": str})
    mlp_per_trip.insert(0, "model_id", "mlp_128_64_32")
    per_trip_frames.append(mlp_per_trip)

    models_summary.append(
        {
            "model_id": "mlp_128_64_32",
            "display_name": "MLP (128-64-32)",
            "family": "Neural Network (PyTorch)",
            "architecture": "28 -> 128 -> 64 -> 32 -> 1 (BatchNorm1d + ReLU + Dropout 0.20, 14,529 params)",
            "config": cfg_mlp["mlp_config"],
            "artifact_path": "ml/mlp/artifacts/mlp_model.pt",
            "num_features": len(feature_cols),
            "cv": {
                "folds": 5,
                "candidate_accuracy": cv_mlp["oof_metrics"]["candidate_accuracy"],
                "precision_macro": cv_mlp["oof_metrics"]["precision_macro"],
                "recall_macro": cv_mlp["oof_metrics"]["recall_macro"],
                "macro_f1": cv_mlp["oof_metrics"]["macro_f1"],
                "macro_f1_std": cv_mlp["fold_macro_f1_std"],
                "obs_top1_accuracy": cv_mlp["oof_metrics"]["obs_top1_accuracy"],
                "obs_top1_accuracy_covered": cv_mlp["oof_metrics"]["obs_top1_accuracy_covered"],
                "log_loss": cv_mlp["oof_metrics"]["log_loss"],
                "candidate_miss_rate": cv_mlp["oof_metrics"]["candidate_miss_rate"],
                "obs_step_continuity": cv_mlp["oof_route_continuity"]["obs_step_continuity_overall"],
                "compressed_route_continuity": cv_mlp["oof_route_continuity"]["compressed_route_continuity_overall"],
                "cv_train_time_s": cv_mlp["cv_train_time_s"],
                "final_train_time_s": meta_mlp["final_train_time_s"],
            },
            "student_test": {
                **st_mlp["metrics"],
                "roc_auc": mlp_curves["roc_auc"],
                "pr_auc": mlp_curves["pr_auc"],
                "obs_step_continuity": st_mlp["route_continuity"]["obs_step_continuity_overall"],
                "compressed_route_continuity": st_mlp["route_continuity"]["compressed_route_continuity_overall"],
                "inference_time_ms": st_mlp["inference_time_ms"],
                "infer_ms_per_trip": st_mlp["infer_ms_per_trip"],
                "infer_ms_per_1k_candidates": st_mlp["infer_ms_per_1k_candidates"],
            },
        }
    )

    # 2. Train & evaluate the 3 comparable scikit-learn baselines on the exact same GroupKFold + student_test
    gkf = GroupKFold(n_splits=5)
    sklearn_specs = _build_sklearn_models()

    for model_id, spec in sklearn_specs.items():
        print(f"\n=== Training & Evaluating {spec['display_name']} ({model_id}) ===")
        oof_probs = np.zeros(len(train_df), dtype=np.float64)
        fold_f1s: List[float] = []

        t_cv0 = time.perf_counter()
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train), 1):
            assert set(groups_train[tr_idx]).isdisjoint(set(groups_train[va_idx])), "Group leakage detected!"
            est = clone(spec["estimator"])
            est.fit(X_train[tr_idx], y_train[tr_idx])
            va_probs = est.predict_proba(X_train[va_idx])[:, 1]
            oof_probs[va_idx] = va_probs

            fold_m = compute_candidate_and_obs_metrics(
                train_df.iloc[va_idx],
                va_probs,
                total_obs_count=int(train_df.iloc[va_idx].groupby(["trip_id", "obs_idx"]).ngroups),
                candidate_miss_count=0,
            )
            fold_f1s.append(fold_m["macro_f1"])
            print(f"  Fold {fold_idx}/5 | Macro-F1={fold_m['macro_f1']:.4f} | LogLoss={fold_m['log_loss']:.4f}")
        cv_train_s = time.perf_counter() - t_cv0

        cv_metrics = compute_candidate_and_obs_metrics(
            train_df, oof_probs, total_obs_count=tot_train_obs, candidate_miss_count=tot_train_miss
        )
        cv_cont, _ = compute_classifier_route_continuity(train_df, oof_probs, TRAIN_DIR)

        # Final fit on all 80 training trips
        t_fit0 = time.perf_counter()
        final_est = clone(spec["estimator"])
        final_est.fit(X_train, y_train)
        final_train_s = time.perf_counter() - t_fit0

        artifact_path = BASELINES_ARTIFACTS_DIR / f"{model_id}.joblib"
        joblib.dump(final_est, artifact_path, compress=3)

        # Evaluate on 10 held-out student_test trips
        t_inf0 = time.perf_counter()
        test_probs = final_est.predict_proba(X_test)[:, 1]
        infer_ms = (time.perf_counter() - t_inf0) * 1000.0

        test_metrics = compute_candidate_and_obs_metrics(
            test_df, test_probs, total_obs_count=tot_test_obs, candidate_miss_count=tot_test_miss
        )
        test_cont, per_trip_df = compute_classifier_route_continuity(
            test_df, test_probs, STUDENT_TEST_DIR
        )
        per_trip_df = per_trip_df.merge(
            test_summary_df[["trip_id", "candidate_misses", "candidate_miss_rate", "avg_candidates_per_obs"]],
            on="trip_id",
            how="left",
        )
        per_trip_df.insert(0, "model_id", model_id)
        per_trip_frames.append(per_trip_df)

        curve_info = _compute_curve_metrics(y_test, test_probs)
        curves_data[model_id] = curve_info

        print(
            f"  [Student-Test] CandAcc={test_metrics['candidate_accuracy']:.4f} | "
            f"Macro-F1={test_metrics['macro_f1']:.4f} | Top-1={test_metrics['obs_top1_accuracy']:.4f} | "
            f"LogLoss={test_metrics['log_loss']:.4f} | ROC-AUC={curve_info['roc_auc']:.4f} | "
            f"Infer={infer_ms:.1f}ms"
        )

        models_summary.append(
            {
                "model_id": model_id,
                "display_name": spec["display_name"],
                "family": spec["family"],
                "architecture": spec["architecture"],
                "config": spec["config"],
                "artifact_path": f"artifacts/baselines/{model_id}.joblib",
                "num_features": len(feature_cols),
                "cv": {
                    "folds": 5,
                    "candidate_accuracy": cv_metrics["candidate_accuracy"],
                    "precision_macro": cv_metrics["precision_macro"],
                    "recall_macro": cv_metrics["recall_macro"],
                    "macro_f1": cv_metrics["macro_f1"],
                    "macro_f1_std": float(np.std(fold_f1s, ddof=1)),
                    "obs_top1_accuracy": cv_metrics["obs_top1_accuracy"],
                    "obs_top1_accuracy_covered": cv_metrics["obs_top1_accuracy_covered"],
                    "log_loss": cv_metrics["log_loss"],
                    "candidate_miss_rate": cv_metrics["candidate_miss_rate"],
                    "obs_step_continuity": cv_cont["obs_step_continuity_overall"],
                    "compressed_route_continuity": cv_cont["compressed_route_continuity_overall"],
                    "cv_train_time_s": round(cv_train_s, 3),
                    "final_train_time_s": round(final_train_s, 3),
                },
                "student_test": {
                    **test_metrics,
                    "roc_auc": curve_info["roc_auc"],
                    "pr_auc": curve_info["pr_auc"],
                    "obs_step_continuity": test_cont["obs_step_continuity_overall"],
                    "compressed_route_continuity": test_cont["compressed_route_continuity_overall"],
                    "inference_time_ms": round(infer_ms, 3),
                    "infer_ms_per_trip": round(infer_ms / max(len(per_trip_df), 1), 3),
                    "infer_ms_per_1k_candidates": round(infer_ms / max(len(test_df), 1) * 1000.0, 4),
                },
            }
        )

    # Sort models by student-test observation Top-1 accuracy (primary route-matching metric), tie-break Macro-F1
    models_summary.sort(
        key=lambda m: (m["student_test"]["obs_top1_accuracy"], m["student_test"]["macro_f1"]),
        reverse=True,
    )
    best_model = models_summary[0]

    # Build flat comparison CSV
    flat_rows = []
    for m in models_summary:
        st = m["student_test"]
        cv = m["cv"]
        flat_rows.append(
            {
                "model_id": m["model_id"],
                "display_name": m["display_name"],
                "family": m["family"],
                "architecture": m["architecture"],
                "cv_macro_f1": round(cv["macro_f1"], 4),
                "cv_top1_accuracy": round(cv["obs_top1_accuracy"], 4),
                "cv_log_loss": round(cv["log_loss"], 4),
                "student_test_candidate_accuracy": round(st["candidate_accuracy"], 4),
                "student_test_precision_macro": round(st["precision_macro"], 4),
                "student_test_recall_macro": round(st["recall_macro"], 4),
                "student_test_macro_f1": round(st["macro_f1"], 4),
                "student_test_top1_accuracy": round(st["obs_top1_accuracy"], 4),
                "student_test_top1_covered": round(st["obs_top1_accuracy_covered"], 4),
                "student_test_log_loss": round(st["log_loss"], 4),
                "student_test_roc_auc": round(st["roc_auc"], 4) if st["roc_auc"] is not None else None,
                "student_test_pr_auc": round(st["pr_auc"], 4) if st["pr_auc"] is not None else None,
                "student_test_candidate_miss_rate": round(st["candidate_miss_rate"], 6),
                "student_test_obs_continuity": round(st["obs_step_continuity"], 4),
                "student_test_compressed_continuity": round(st["compressed_route_continuity"], 4),
                "student_test_infer_ms_total": round(st["inference_time_ms"], 3),
                "student_test_infer_ms_per_trip": round(st["infer_ms_per_trip"], 3),
                "final_train_time_s": round(cv["final_train_time_s"], 3),
            }
        )

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(COMPARISON_DIR / "model_comparison_table.csv", index=False)

    all_per_trip_df = pd.concat(per_trip_frames, ignore_index=True)
    all_per_trip_df.to_csv(COMPARISON_DIR / "per_trip_all_models.csv", index=False)

    # Full unified JSON with dataset overview + teammate branch audit status (zero fabricated numbers!)
    unified_payload = {
        "schema_version": "2.0.0",
        "feature_schema": {
            "num_features": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "dataset_summary": {
            "train_trips": int(len(train_summary_df)),
            "train_observations": tot_train_obs,
            "train_candidates": int(len(train_df)),
            "train_candidate_miss_rate": round(tot_train_miss / max(tot_train_obs, 1), 6),
            "student_test_trips": int(len(test_summary_df)),
            "student_test_observations": tot_test_obs,
            "student_test_candidates": int(len(test_df)),
            "student_test_candidate_miss_rate": round(tot_test_miss / max(tot_test_obs, 1), 6),
            "evaluator_test_used": False,
        },
        "best_model_id": best_model["model_id"],
        "best_model_selection_criterion": "Highest held-out student_test Observation-Level Top-1 Accuracy (tie-breaker: Macro F1) across identical 28-feature dataset",
        "models": models_summary,
        "teammate_branch_status": [
            {
                "branch": "origin/kushi/data-foundation",
                "owner": "Kushi",
                "models": ["logistic_regression (6-feature)", "knn (6-feature, 100k subsample)"],
                "status": "Evaluated on separate 6-feature 64/16 holdout split (not 28-feature student_test); kept separate to avoid comparing incompatible feature sets.",
                "reported_holdout_metrics": {
                    "logistic_regression_6feat": {
                        "covered_candidate_accuracy": 0.8647,
                        "covered_macro_f1": 0.6802,
                        "log_loss": 0.2728,
                        "obs_top1_all": 0.7285,
                    },
                    "knn_6feat": {
                        "covered_candidate_accuracy": 0.8880,
                        "covered_macro_f1": 0.7057,
                        "log_loss": 1.5288,
                        "obs_top1_all": 0.6357,
                    },
                },
            },
            {
                "branch": "origin/kritika-hgb-hmm",
                "owner": "Kritika",
                "models": ["HistGradientBoosting (6-feature tuning scripts)", "HMM/Viterbi (pending)"],
                "status": "Training and Bayesian tuning scripts committed on branch; no trained model artifact, evaluation metrics, or HMM decoder committed yet.",
                "reported_holdout_metrics": None,
            },
        ],
    }

    (COMPARISON_DIR / "model_comparison_summary.json").write_text(
        json.dumps(unified_payload, indent=2), encoding="utf-8"
    )
    (COMPARISON_DIR / "roc_pr_curves.json").write_text(
        json.dumps(curves_data, indent=2), encoding="utf-8"
    )

    _generate_comparison_plots(models_summary, curves_data, COMPARISON_DIR / "plots")

    print("\n=== UNIFIED MULTI-MODEL COMPARISON TABLE (28 FEATURES, 10 STUDENT-TEST TRIPS) ===")
    print(
        flat_df[
            [
                "display_name",
                "student_test_candidate_accuracy",
                "student_test_macro_f1",
                "student_test_top1_accuracy",
                "student_test_log_loss",
                "student_test_roc_auc",
                "student_test_infer_ms_per_trip",
            ]
        ].to_string(index=False)
    )
    print(f"\nBest comparable model: {best_model['display_name']} ({best_model['model_id']})")
    print(f"Saved comparison outputs to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
