"""Comprehensive evaluation, plotting, and Folium map generation for the PathSnap MLP baseline.

Computes actual metrics on both:
  1. 80-trip 5-Fold GroupKFold Cross-Validation (out-of-fold predictions)
  2. 10-trip Student-Testing partition (held-out development evaluation)

NEVER accesses evaluator-testing data.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List

# Ensure matplotlib has a writable config directory
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pathsnap_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.mlp.config import (
    ARTIFACTS_DIR,
    CandidateConfig,
    PLOTS_DIR,
    RESULTS_DIR,
    STUDENT_TEST_DIR,
    ensure_directories,
)
from ml.mlp.data_adapter import build_or_load_partition_dataset, load_raw_trip
from ml.mlp.metrics_utils import (
    compute_candidate_and_obs_metrics,
    compute_classifier_route_continuity,
)
from ml.mlp.predict import MLPPredictor


def plot_training_curves(cv_summary: Dict[str, object], plots_dir: Path) -> None:
    """Plot actual epoch-wise train/val loss and Macro-F1 / Top-1 curves across all 5 CV folds."""
    histories = cv_summary["fold_histories"]
    fold_metrics = cv_summary["fold_metrics"]
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]

    # 1. Loss curves (Left: Training Loss per fold, Right: Validation Loss per fold + best epoch stars)
    fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=150, sharey=True)
    for i, (h, fm) in enumerate(zip(histories, fold_metrics)):
        c = palette[i % len(palette)]
        eps = np.array(h["epochs"])
        tr_l = np.array(h["train_loss"])
        va_l = np.array(h["val_loss"])
        best_ep = int(fm["best_epoch"])
        best_idx = best_ep - 1

        lbl = f"Fold {i+1} (best ep={best_ep})"
        ax_tr.plot(eps, tr_l, color=c, linewidth=2.0, alpha=0.9, label=lbl)
        ax_va.plot(eps, va_l, color=c, linewidth=2.0, alpha=0.9, label=lbl)
        ax_va.scatter([best_ep], [va_l[best_idx]], color=c, marker="*", s=110, zorder=5, edgecolors="black", linewidths=0.5)

    ax_tr.set_title("Training BCE Loss by Fold (5-Fold GroupKFold)", fontsize=10.8, fontweight="bold")
    ax_tr.set_xlabel("Epoch")
    ax_tr.set_ylabel("Binary Cross-Entropy Loss (Log Loss)")
    ax_tr.grid(True, linestyle="--", alpha=0.4)
    ax_tr.legend(fontsize=8.5, loc="upper right")

    ax_va.set_title("Validation BCE Loss by Fold (* = Early-Stopping Best)", fontsize=10.8, fontweight="bold")
    ax_va.set_xlabel("Epoch")
    ax_va.grid(True, linestyle="--", alpha=0.4)
    ax_va.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        f"PathSnap MLP ({cv_summary['experiment_name']}) — Actual Epoch-Wise Loss Curves (80 Training Trips)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png")
    plt.close(fig)

    # 2. Performance curves (Left: Train vs Val Macro-F1 per fold, Right: Val Obs Top-1 Accuracy per fold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=150)
    for i, (h, fm) in enumerate(zip(histories, fold_metrics)):
        c = palette[i % len(palette)]
        eps = np.array(h["epochs"])
        tr_f1 = np.array(h["train_macro_f1"])
        va_f1 = np.array(h["val_macro_f1"])
        va_t1 = np.array(h["val_top1_acc"])
        best_ep = int(fm["best_epoch"])
        best_idx = best_ep - 1

        ax1.plot(eps, tr_f1, color=c, linestyle="--", linewidth=1.3, alpha=0.55)
        ax1.plot(eps, va_f1, color=c, linestyle="-", linewidth=2.1, alpha=0.95, label=f"Fold {i+1} Val (solid) / Train (dashed)")
        ax1.scatter([best_ep], [va_f1[best_idx]], color=c, marker="*", s=95, zorder=5, edgecolors="black", linewidths=0.5)

        ax2.plot(eps, va_t1, color=c, linewidth=2.1, alpha=0.95, label=f"Fold {i+1} (best Top-1={fm['obs_top1_accuracy']:.3f})")
        ax2.scatter([best_ep], [va_t1[best_idx]], color=c, marker="*", s=95, zorder=5, edgecolors="black", linewidths=0.5)

    ax1.set_title("Train (dashed) vs Validation (solid) Macro F1-Score", fontsize=10.8, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Macro F1-Score")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=7.8, loc="lower right")

    ax2.set_title("Validation Observation-Level Top-1 Accuracy (All Obs)", fontsize=10.8, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.3, loc="center right")

    fig.suptitle(
        f"PathSnap MLP ({cv_summary['experiment_name']}) — 5-Fold GroupKFold Performance Curves",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "performance_curves.png")
    plt.close(fig)


def plot_confusion_matrices(
    cv_cm: List[List[int]], test_cm: List[List[int]], plots_dir: Path
) -> None:
    """Plot candidate-level confusion matrices for 5-Fold CV (OOF) and Student-Test."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), dpi=150)
    titles = [
        "5-Fold GroupKFold CV (80 Train Trips, OOF)",
        "Student-Testing Set (10 Held-Out Trips)",
    ]
    cms = [np.array(cv_cm, dtype=np.int64), np.array(test_cm, dtype=np.int64)]

    for ax, cm, title in zip(axes, cms, titles):
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = cm / row_sums
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Non-Match (0)", "Pred: Match (1)"])
        ax.set_yticklabels(["True: Non-Match (0)", "True: Match (1)"])
        ax.set_title(title, fontsize=10.5, fontweight="bold")

        for r in range(2):
            for c in range(2):
                val_cnt = cm[r, c]
                val_pct = cm_norm[r, c] * 100.0
                color = "white" if cm_norm[r, c] > 0.55 else "black"
                ax.text(
                    c,
                    r,
                    f"{val_cnt:,}\n({val_pct:.2f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("PathSnap MLP Candidate-Level Confusion Matrices", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrices.png")
    plt.close(fig)


def plot_per_trip_student_test(per_trip_df: pd.DataFrame, plots_dir: Path) -> None:
    """Bar chart of observation Top-1 accuracy and route continuity across the 10 student_test trips."""
    df_s = per_trip_df.sort_values("trip_id").reset_index(drop=True)
    x = np.arange(len(df_s))
    w = 0.27

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150)
    ax.bar(x - w, df_s["top1_accuracy_obs"], width=w, label="Obs Top-1 Accuracy", color="#1f77b4")
    ax.bar(x, df_s["obs_step_continuity"], width=w, label="Obs-Step Route Continuity", color="#2ca02c")
    ax.bar(
        x + w,
        df_s["compressed_route_continuity"],
        width=w,
        label="Compressed Route Continuity",
        color="#ff7f0e",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_s["trip_id"], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score (0 to 1)")
    ax.set_xlabel("Student-Testing Trip ID")
    ax.set_title("PathSnap MLP Classifier-Only Performance Across All 10 Student-Testing Trips", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    for i, v in enumerate(df_s["top1_accuracy_obs"]):
        ax.text(i - w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(plots_dir / "student_test_per_trip_comparison.png")
    plt.close(fig)


def plot_architecture_comparison(comp_csv: Path, plots_dir: Path) -> None:
    """Plot comparison of baseline MLP architectures evaluated on 5-fold GroupKFold CV."""
    if not comp_csv.exists():
        return
    cdf = pd.read_csv(comp_csv)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=150)
    names = cdf["experiment"].tolist()
    colors = ["#4c78a8", "#f58518", "#54a24b"][: len(names)]

    metrics = [
        ("cv_macro_f1", "CV Macro F1-Score (higher is better)"),
        ("cv_top1_acc", "CV Obs Top-1 Accuracy (higher is better)"),
        ("cv_log_loss", "CV Log Loss (lower is better)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        vals = cdf[col].to_numpy()
        bars = ax.bar(names, vals, color=colors, width=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ymin = max(0.0, float(vals.min()) - 0.05)
        ymax = float(vals.max()) + 0.05
        ax.set_ylim(ymin, ymax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (ymax - ymin) * 0.02, f"{v:.4f}", ha="center", fontsize=9)

    fig.suptitle("Baseline MLP Architecture Comparison (80-Trip 5-Fold GroupKFold CV)", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "architecture_comparison.png")
    plt.close(fig)


def generate_folium_comparison_map(
    trip_dir: Path,
    pred_arc_ids: np.ndarray,
    out_html: Path,
    title_suffix: str = "",
) -> bool:
    """Generate an interactive Folium map comparing raw GPS, road network, reference route, and MLP route."""
    try:
        import folium
    except ImportError:
        print("WARNING: folium is not installed; skipping Folium map generation.")
        return False

    trip = load_raw_trip(trip_dir, load_route=True)
    nodes = trip.nodes_lonlat  # [lon, lat]
    arcs = trip.arcs
    track = trip.track  # [lon, lat, ts]

    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # 1. Relevant local road-network subset (arcs near the trip bounding box)
    lat_min, lat_max = float(track[:, 1].min()) - 0.002, float(track[:, 1].max()) + 0.002
    lon_min, lon_max = float(track[:, 0].min()) - 0.002, float(track[:, 0].max()) + 0.002
    u_ll = nodes[arcs[:, 0]]
    v_ll = nodes[arcs[:, 1]]
    in_box = (
        (u_ll[:, 0] >= lon_min)
        & (u_ll[:, 0] <= lon_max)
        & (u_ll[:, 1] >= lat_min)
        & (u_ll[:, 1] <= lat_max)
    )
    net_indices = np.flatnonzero(in_box)
    # Cap network background segments for smooth browser rendering
    if len(net_indices) > 2500:
        step = max(1, len(net_indices) // 2500)
        net_indices = net_indices[::step]

    net_group = folium.FeatureGroup(name="Local Road Network", show=True)
    net_lines = [
        [[float(u_ll[i, 1]), float(u_ll[i, 0])], [float(v_ll[i, 1]), float(v_ll[i, 0])]]
        for i in net_indices
    ]
    if net_lines:
        folium.PolyLine(net_lines, color="#999999", weight=1.2, opacity=0.45).add_to(net_group)
    net_group.add_to(fmap)

    # 2. Reference route (green)
    if trip.route is not None and len(trip.route) > 0:
        ref_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        ref_lines = [
            [
                [float(nodes[arcs[aid, 0], 1]), float(nodes[arcs[aid, 0], 0])],
                [float(nodes[arcs[aid, 1], 1]), float(nodes[arcs[aid, 1], 0])],
            ]
            for aid in trip.route
        ]
        folium.PolyLine(ref_lines, color="#2ca02c", weight=5.0, opacity=0.80, tooltip="Reference Route").add_to(
            ref_group
        )
        ref_group.add_to(fmap)

    # 3. MLP Predicted route (red, compressed unique consecutive segments)
    pred_group = folium.FeatureGroup(name="MLP Predicted Route (Classifier-Only Top-1)", show=True)
    comp_pred = [int(pred_arc_ids[0])]
    for a in pred_arc_ids[1:]:
        if int(a) != comp_pred[-1]:
            comp_pred.append(int(a))
    pred_lines = [
        [
            [float(nodes[arcs[aid, 0], 1]), float(nodes[arcs[aid, 0], 0])],
            [float(nodes[arcs[aid, 1], 1]), float(nodes[arcs[aid, 1], 0])],
        ]
        for aid in comp_pred
    ]
    folium.PolyLine(
        pred_lines,
        color="#d62728",
        weight=3.0,
        opacity=0.85,
        dash_array="6, 4",
        tooltip="MLP Classifier-Only Top-1 Route",
    ).add_to(pred_group)
    pred_group.add_to(fmap)

    # 4. Raw GPS observations (blue dots, subsampled slightly if huge for responsiveness)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    stride = max(1, len(track) // 600)
    for idx in range(0, len(track), stride):
        folium.CircleMarker(
            location=[float(track[idx, 1]), float(track[idx, 0])],
            radius=2.3,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.75,
            popup=f"Obs {idx} | t={track[idx, 2]:.0f}s",
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # Legend HTML overlay
    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                border: 1px solid #ccc; font-family: Arial, sans-serif; font-size: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
      <div style="font-weight: bold; margin-bottom: 6px;">PathSnap — Trip {trip.trip_id} {title_suffix}</div>
      <div><span style="color:#1f77b4; font-weight:bold;">●</span> Raw GPS Observations</div>
      <div><span style="color:#2ca02c; font-weight:bold;">━━</span> Reference Route (.route)</div>
      <div><span style="color:#d62728; font-weight:bold;">┅┅</span> MLP Top-1 Predicted Route</div>
      <div><span style="color:#999999; font-weight:bold;">──</span> Local Road Network</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(track[:, 1].min()), float(track[:, 0].min())], [float(track[:, 1].max()), float(track[:, 0].max())]])

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return True


def main() -> None:
    ensure_directories()
    predictor = MLPPredictor.from_artifacts(ARTIFACTS_DIR, device="cpu")
    cand_cfg_dict = predictor.model_config.get("candidate_config", {})
    cand_cfg = CandidateConfig(**cand_cfg_dict) if cand_cfg_dict else CandidateConfig()

    # Load 5-Fold CV summary
    cv_summary = json.loads((RESULTS_DIR / "cv_selected_summary.json").read_text(encoding="utf-8"))
    train_meta = json.loads((ARTIFACTS_DIR / "training_metadata.json").read_text(encoding="utf-8"))

    # Load Student-Testing partition (10 held-out trips)
    test_df, test_summary_df = build_or_load_partition_dataset(
        STUDENT_TEST_DIR, "student_test", cfg=cand_cfg, use_cache=True, load_labels=True
    )

    # Measure inference time on student_test
    t0 = time.perf_counter()
    _, test_probs, _, _, infer_ms = predictor.predict_scores(test_df)
    total_infer_s = time.perf_counter() - t0

    tot_test_obs = int(test_summary_df["num_observations"].sum())
    tot_test_miss = int(test_summary_df["candidate_misses"].sum())

    test_metrics = compute_candidate_and_obs_metrics(
        test_df, test_probs, total_obs_count=tot_test_obs, candidate_miss_count=tot_test_miss
    )
    test_continuity, per_trip_df = compute_classifier_route_continuity(
        test_df, test_probs, STUDENT_TEST_DIR
    )

    # Merge candidate miss rate per trip into per_trip_df
    per_trip_df = per_trip_df.merge(
        test_summary_df[["trip_id", "candidate_misses", "candidate_miss_rate", "avg_candidates_per_obs"]],
        on="trip_id",
        how="left",
    )
    per_trip_df.to_csv(RESULTS_DIR / "student_test_per_trip.csv", index=False)

    # Save prediction vs reference comparison sample (first 50 obs of trip 00000000 + summary across trips)
    eval_work = test_df[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    eval_work["pred_prob"] = test_probs
    top1_idx = eval_work.groupby(["trip_id", "obs_idx"], sort=False)["pred_prob"].idxmax()
    top1_df = eval_work.loc[top1_idx].sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)
    # Attach reference arc_id where present in candidates
    pos_df = eval_work[eval_work["label"] == 1][["trip_id", "obs_idx", "arc_id"]].rename(
        columns={"arc_id": "reference_arc_id"}
    )
    comp_obs_df = top1_df.merge(pos_df, on=["trip_id", "obs_idx"], how="left")
    comp_obs_df = comp_obs_df.rename(columns={"arc_id": "predicted_arc_id", "label": "is_correct"})
    comp_obs_df["reference_arc_id"] = comp_obs_df["reference_arc_id"].fillna(-1).astype(int)
    comp_obs_df.head(200).to_csv(RESULTS_DIR / "prediction_vs_reference_sample.csv", index=False)

    # Full Student-Test JSON record
    student_test_record = {
        "partition": "student_test (10 held-out trips)",
        "model_name": predictor.model_name,
        "metrics": test_metrics,
        "route_continuity": test_continuity,
        "inference_time_s": round(total_infer_s, 4),
        "inference_time_ms": round(infer_ms, 3),
        "infer_ms_per_trip": round(infer_ms / max(len(per_trip_df), 1), 3),
        "infer_ms_per_1k_candidates": round(infer_ms / max(len(test_df), 1) * 1000.0, 4),
    }
    (RESULTS_DIR / "student_test_metrics.json").write_text(
        json.dumps(student_test_record, indent=2), encoding="utf-8"
    )

    # Side-by-side CV vs Student-Test summary table
    cv_m = cv_summary["oof_metrics"]
    cv_c = cv_summary["oof_route_continuity"]
    summary_rows = [
        ("Candidate Accuracy", f"{cv_m['candidate_accuracy']:.4f}", f"{test_metrics['candidate_accuracy']:.4f}"),
        ("Macro Precision", f"{cv_m['precision_macro']:.4f}", f"{test_metrics['precision_macro']:.4f}"),
        ("Macro Recall", f"{cv_m['recall_macro']:.4f}", f"{test_metrics['recall_macro']:.4f}"),
        ("Macro F1-Score", f"{cv_m['macro_f1']:.4f}", f"{test_metrics['macro_f1']:.4f}"),
        ("Positive-Class Precision", f"{cv_m['precision_positive']:.4f}", f"{test_metrics['precision_positive']:.4f}"),
        ("Positive-Class Recall", f"{cv_m['recall_positive']:.4f}", f"{test_metrics['recall_positive']:.4f}"),
        ("Positive-Class F1-Score", f"{cv_m['binary_f1_positive']:.4f}", f"{test_metrics['binary_f1_positive']:.4f}"),
        ("Observation-Level Top-1 Accuracy (All Obs)", f"{cv_m['obs_top1_accuracy']:.4f}", f"{test_metrics['obs_top1_accuracy']:.4f}"),
        ("Observation-Level Top-1 Accuracy (Covered Obs)", f"{cv_m['obs_top1_accuracy_covered']:.4f}", f"{test_metrics['obs_top1_accuracy_covered']:.4f}"),
        ("Log Loss (Binary Cross-Entropy)", f"{cv_m['log_loss']:.4f}", f"{test_metrics['log_loss']:.4f}"),
        ("Candidate Miss Rate", f"{cv_m['candidate_miss_rate']*100:.2f}%", f"{test_metrics['candidate_miss_rate']*100:.2f}%"),
        ("Route Continuity (Obs-Step, Classifier-Only)", f"{cv_c['obs_step_continuity_overall']:.4f}", f"{test_continuity['obs_step_continuity_overall']:.4f}"),
        ("Route Continuity (Compressed, Classifier-Only)", f"{cv_c['compressed_route_continuity_overall']:.4f}", f"{test_continuity['compressed_route_continuity_overall']:.4f}"),
        ("Training Time (s)", f"{cv_summary['cv_train_time_s']:.2f} (5-fold) / {train_meta['final_train_time_s']:.2f} (final)", "N/A (inference only)"),
        ("Inference Time (ms / 1k candidates)", f"{cv_summary['infer_ms_per_1k_candidates']:.3f}", f"{student_test_record['infer_ms_per_1k_candidates']:.3f}"),
    ]
    summary_table_df = pd.DataFrame(
        summary_rows, columns=["Metric", "5-Fold GroupKFold CV (80 Train Trips)", "Student-Test (10 Held-Out Trips)"]
    )
    summary_table_df.to_csv(RESULTS_DIR / "performance_summary_table.csv", index=False)

    print("\n=== MLP EVALUATION SUMMARY (ACTUAL EXPERIMENTAL RESULTS) ===")
    print(summary_table_df.to_string(index=False))
    print("\n=== PER-TRIP STUDENT-TEST BREAKDOWN ===")
    print(per_trip_df.to_string(index=False))

    # Generate all plots
    plot_training_curves(cv_summary, PLOTS_DIR)
    plot_confusion_matrices(cv_m["confusion_matrix"], test_metrics["confusion_matrix"], PLOTS_DIR)
    plot_per_trip_student_test(per_trip_df, PLOTS_DIR)
    plot_architecture_comparison(RESULTS_DIR / "cv_architecture_comparison.csv", PLOTS_DIR)

    # Generate interactive Folium maps for 3 representative student_test trips (best, median, hardest)
    ordered_trips = per_trip_df.sort_values("top1_accuracy_obs")
    chosen_tids = [
        str(ordered_trips.iloc[0]["trip_id"]),   # hardest
        str(ordered_trips.iloc[len(ordered_trips) // 2]["trip_id"]),  # median
        str(ordered_trips.iloc[-1]["trip_id"]),  # best
    ]
    for tid in dict.fromkeys(chosen_tids):
        sub_arcs = top1_df[top1_df["trip_id"] == tid]["arc_id"].to_numpy(dtype=np.int64)
        row_info = per_trip_df[per_trip_df["trip_id"] == tid].iloc[0]
        suffix = f"(Top-1: {row_info['top1_accuracy_obs']*100:.1f}%)"
        ok = generate_folium_comparison_map(
            STUDENT_TEST_DIR / tid,
            sub_arcs,
            PLOTS_DIR / f"folium_student_test_{tid}.html",
            title_suffix=suffix,
        )
        if ok:
            print(f"Generated Folium map: {PLOTS_DIR / f'folium_student_test_{tid}.html'}")

    print(f"\nAll evaluation results saved to: {RESULTS_DIR}")
    print(f"All evaluation plots saved to:   {PLOTS_DIR}")


if __name__ == "__main__":
    main()
