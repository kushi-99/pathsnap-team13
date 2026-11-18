"""Shared evaluation metrics for candidate classification and route continuity."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def compute_candidate_and_obs_metrics(
    df: pd.DataFrame,
    probs: np.ndarray,
    total_obs_count: int,
    candidate_miss_count: int,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Compute candidate-level and observation-level metrics from predicted probabilities.

    Args:
        df: Candidate DataFrame containing ['trip_id', 'obs_idx', 'arc_id', 'label'].
        probs: Predicted P(match=1) for each row in `df`.
        total_obs_count: Total raw GPS observations across all trips in this partition.
        candidate_miss_count: Number of observations whose reference arc was absent from candidates.
        threshold: Probability threshold for candidate-level binary classification.
    """
    y_true = df["label"].to_numpy(dtype=np.int64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y_pred = (p >= threshold).astype(np.int64)

    cand_acc = float(accuracy_score(y_true, y_pred))
    prec_pos = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_pos = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_pos = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ll = float(log_loss(y_true, p, labels=[0, 1]))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # Observation-level Top-1 Accuracy: pick candidate with max prob per (trip_id, obs_idx)
    eval_frame = pd.DataFrame(
        {
            "trip_id": df["trip_id"].values,
            "obs_idx": df["obs_idx"].values,
            "arc_id": df["arc_id"].values,
            "label": y_true,
            "prob": p,
        }
    )
    top1_idx = eval_frame.groupby(["trip_id", "obs_idx"], sort=False)["prob"].idxmax()
    top1_rows = eval_frame.loc[top1_idx]

    top1_correct = int(top1_rows["label"].sum())
    covered_obs = total_obs_count - candidate_miss_count
    top1_acc_all = float(top1_correct / max(total_obs_count, 1))
    top1_acc_covered = float(top1_correct / max(covered_obs, 1))
    miss_rate = float(candidate_miss_count / max(total_obs_count, 1))

    return {
        "candidate_accuracy": cand_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "macro_f1": f1_macro,
        "precision_positive": prec_pos,
        "recall_positive": rec_pos,
        "binary_f1_positive": f1_pos,
        "log_loss": ll,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "obs_top1_accuracy": top1_acc_all,
        "obs_top1_accuracy_covered": top1_acc_covered,
        "obs_top1_correct": top1_correct,
        "total_observations": int(total_obs_count),
        "covered_observations": int(covered_obs),
        "candidate_misses": int(candidate_miss_count),
        "candidate_miss_rate": miss_rate,
        "threshold": float(threshold),
    }


def compute_classifier_route_continuity(
    df: pd.DataFrame,
    probs: np.ndarray,
    partition_dir: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute classifier-only route continuity per trip from Top-1 predicted arcs.

    A consecutive transition (arc_{t-1} -> arc_t) is considered topologically continuous if:
      - arc_t == arc_{t-1} (vehicle stayed on the same road segment), OR
      - arcs[arc_{t-1}].to_node_id == arcs[arc_t].from_node_id (direct graph edge).

    Also computes compressed-route continuity (after collapsing consecutive duplicate arcs,
    matching `.route` sequence format).
    """
    work = pd.DataFrame(
        {
            "trip_id": df["trip_id"].values,
            "obs_idx": df["obs_idx"].values,
            "arc_id": df["arc_id"].values,
            "label": df["label"].values if "label" in df.columns else 0,
            "prob": probs,
        }
    )
    top1_idx = work.groupby(["trip_id", "obs_idx"], sort=False)["prob"].idxmax()
    top1 = work.loc[top1_idx].sort_values(["trip_id", "obs_idx"])

    per_trip_rows: List[Dict[str, object]] = []
    tot_obs_trans = 0
    ok_obs_trans = 0
    tot_comp_trans = 0
    ok_comp_trans = 0

    for tid, grp in top1.groupby("trip_id", sort=True):
        arcs_path = partition_dir / str(tid) / f"{tid}.arcs"
        arcs = np.loadtxt(arcs_path, dtype=np.int64)
        if arcs.ndim == 1:
            arcs = arcs.reshape(1, -1)

        pred_arcs = grp["arc_id"].to_numpy(dtype=np.int64)
        obs_ok = 0
        obs_tot = max(len(pred_arcs) - 1, 0)
        for i in range(obs_tot):
            a1, a2 = pred_arcs[i], pred_arcs[i + 1]
            if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                obs_ok += 1

        # Compressed sequence (collapse consecutive identical arc_ids)
        comp = [pred_arcs[0]] if len(pred_arcs) > 0 else []
        for a in pred_arcs[1:]:
            if a != comp[-1]:
                comp.append(a)
        comp_ok = 0
        comp_tot = max(len(comp) - 1, 0)
        for i in range(comp_tot):
            a1, a2 = comp[i], comp[i + 1]
            if arcs[a1, 1] == arcs[a2, 0]:
                comp_ok += 1

        tot_obs_trans += obs_tot
        ok_obs_trans += obs_ok
        tot_comp_trans += comp_tot
        ok_comp_trans += comp_ok

        trip_top1 = float(grp["label"].mean()) if "label" in df.columns else float("nan")
        per_trip_rows.append(
            {
                "trip_id": str(tid),
                "observations": int(len(pred_arcs)),
                "top1_accuracy_obs": trip_top1,
                "obs_step_continuity": float(obs_ok / max(obs_tot, 1)),
                "compressed_route_len": int(len(comp)),
                "compressed_route_continuity": float(comp_ok / max(comp_tot, 1)),
            }
        )

    per_trip_df = pd.DataFrame(per_trip_rows)
    agg = {
        "obs_step_continuity_overall": float(ok_obs_trans / max(tot_obs_trans, 1)),
        "obs_step_continuity_mean_trip": float(per_trip_df["obs_step_continuity"].mean()),
        "compressed_route_continuity_overall": float(ok_comp_trans / max(tot_comp_trans, 1)),
        "compressed_route_continuity_mean_trip": float(per_trip_df["compressed_route_continuity"].mean()),
        "total_obs_transitions": int(tot_obs_trans),
        "continuous_obs_transitions": int(ok_obs_trans),
        "total_compressed_transitions": int(tot_comp_trans),
        "continuous_compressed_transitions": int(ok_comp_trans),
    }
    return agg, per_trip_df
