"""PathSnap Review 2 — Complete Multi-Page Streamlit Evaluator Application (Laptop 2).

Launch on Laptop 2:
    $env:PATHSNAP_API_URL = "http://<LAPTOP_1_IP>:8000"
    streamlit run ui/evaluator_dashboard/app.py

Pages:
  1. Dashboard          — Project overview, dataset statistics, available models, best model KPIs
  2. Model Comparison   — Multi-model table, comparison charts, confusion matrices, per-trip metrics
  3. GPS Map Matching   — Interactive trip selection, live FastAPI inference, multi-layer Folium map
  4. Evaluation         — Confusion matrix, Precision-Recall, real ROC/PR curves, continuity, downloads
  5. System Information — Model architecture, 28-feature schema, API connection & backend health
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Ensure local ui/evaluator_dashboard imports work regardless of CWD
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from api_client import APIClientError, PathSnapAPIClient, get_default_api_url
from map_renderer import build_prediction_map

APP_VERSION = "2.0.0"

st.set_page_config(
    page_title="PathSnap — Distributed Map-Matching Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _fmt_pct(val: Optional[float], decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{float(val) * 100:.{decimals}f}%"


def _fmt_num(val: Optional[float], decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{float(val):.{decimals}f}"


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_health(api_url: str) -> Dict[str, Any]:
    return PathSnapAPIClient(base_url=api_url).check_health()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_models(api_url: str) -> Dict[str, Any]:
    return PathSnapAPIClient(base_url=api_url).list_models()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_evaluation(api_url: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    return PathSnapAPIClient(base_url=api_url).run_evaluation(model_name=model_name)


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_trips(api_url: str, partition: str) -> List[Dict[str, Any]]:
    return PathSnapAPIClient(base_url=api_url).list_trips(partition=partition)


def _plot_confusion_matrix_fig(cm_list: List[List[int]], title: str):
    cm = np.array(cm_list, dtype=np.int64)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=130)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Non-Match (0)", "Pred: Match (1)"])
    ax.set_yticklabels(["True: Non-Match (0)", "True: Match (1)"])
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    for r in range(2):
        for c in range(2):
            color = "white" if cm_norm[r, c] > 0.55 else "black"
            ax.text(
                c,
                r,
                f"{cm[r, c]:,}\n({cm_norm[r, c] * 100:.2f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_offline_banner(api_url: str, err: str) -> None:
    st.error(
        f"**FastAPI Backend Offline / Unreachable (`{api_url}`)**\n\n"
        f"{err}\n\n"
        "**How to start Laptop 1 Backend (PowerShell):**\n"
        "```powershell\n"
        "uvicorn api.main:app --host 0.0.0.0 --port 8000\n"
        "```\n"
        "Then click **Test / Refresh Connection** in the sidebar."
    )


# ---------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------------------------
def render_dashboard(api_url: str, eval_data: Dict[str, Any], models_data: Dict[str, Any]) -> None:
    st.title("PathSnap — Distributed Map-Matching & Route Reconstruction")
    st.markdown(
        "Reconstructs realistic vehicle routes by matching noisy GPS observations to road-network "
        "segments using multi-model candidate scoring, FastAPI distributed inference (Laptop 1), "
        "and interactive Folium geographical visualization (Laptop 2)."
    )

    ds = eval_data.get("dataset_summary") or {}
    models = eval_data.get("models") or []
    best_id = eval_data.get("best_model_id") or models_data.get("best_model_id")
    best_model = next((m for m in models if m["model_id"] == best_id), models[0] if models else None)

    st.subheader("Dataset & Pipeline Overview (28 Observable Features)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Training Trips", f"{ds.get('train_trips', 80)}")
    c2.metric("Training GPS Obs", f"{ds.get('train_observations', 208417):,}")
    c3.metric("Generated Candidates (Train)", f"{ds.get('train_candidates', 1779885):,}")
    c4.metric("Student-Test Trips", f"{ds.get('student_test_trips', 10)}")
    c5.metric("Loaded ML Models", f"{len(models_data.get('models', []))}")

    st.markdown("---")
    if best_model:
        st_m = best_model.get("student_test", {})
        st.subheader(f"Best Comparable Model (Held-Out Student-Test): `{best_model['display_name']}`")
        st.caption(
            f"Architecture: {best_model.get('architecture', 'N/A')} | "
            f"Selection Criterion: {eval_data.get('best_model_selection_criterion', 'Highest Student-Test Top-1 Accuracy')}"
        )
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Candidate Accuracy", _fmt_pct(st_m.get("candidate_accuracy")))
        b2.metric("Macro F1-Score", _fmt_pct(st_m.get("macro_f1")))
        b3.metric("Obs-Level Top-1 Accuracy", _fmt_pct(st_m.get("obs_top1_accuracy")))
        b4.metric("Log Loss (BCE)", _fmt_num(st_m.get("log_loss"), 4))
        b5.metric("Candidate Miss Rate", _fmt_pct(st_m.get("candidate_miss_rate")))

        st.info(
            "**Metric Distinction:** *Candidate Accuracy* measures binary classification over all "
            "`172,897` candidate `(observation, arc)` pairs (where ~89% are non-matches). "
            "*Observation-Level Top-1 Accuracy* measures the fraction of all `19,160` GPS observations "
            "whose single highest-scoring candidate arc equals the aligned reference road segment."
        )

    st.subheader("Available Comparable Models (Evaluated on Identical 28-Feature Student-Test)")
    if models:
        rows = []
        for m in models:
            stm = m.get("student_test", {})
            cvm = m.get("cv", {})
            rows.append(
                {
                    "Model": m["display_name"],
                    "Model ID": m["model_id"],
                    "Family": m["family"],
                    "CV Macro F1": round(float(cvm.get("macro_f1", 0.0)), 4),
                    "Student-Test Cand Acc": round(float(stm.get("candidate_accuracy", 0.0)), 4),
                    "Student-Test Macro F1": round(float(stm.get("macro_f1", 0.0)), 4),
                    "Student-Test Obs Top-1": round(float(stm.get("obs_top1_accuracy", 0.0)), 4),
                    "Log Loss": round(float(stm.get("log_loss", 0.0)), 4),
                    "Infer (ms / trip)": round(float(stm.get("infer_ms_per_trip", 0.0)), 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 2: MODEL COMPARISON
# ---------------------------------------------------------------------------
def render_model_comparison(eval_data: Dict[str, Any]) -> None:
    st.title("Model Comparison (Equivalent 28-Feature Dataset & Metrics)")
    models = eval_data.get("models") or []
    per_trip_all = pd.DataFrame(eval_data.get("per_trip") or [])

    if not models:
        st.warning("No comparable model evaluation summaries found on the backend.")
        return

    st.caption(
        "All models below were trained on the exact same 80 training trips (5-fold trip-grouped GroupKFold CV + "
        "full 80-trip fit) and evaluated on the exact same 10 held-out student-test trips (19,160 GPS observations, "
        "172,897 candidate pairs, 0.18% candidate miss rate)."
    )

    table_rows = []
    for m in models:
        stm = m["student_test"]
        table_rows.append(
            {
                "Model": m["display_name"],
                "Family": m["family"],
                "Architecture": m["architecture"],
                "Candidate Acc": round(stm["candidate_accuracy"], 4),
                "Macro Precision": round(stm["precision_macro"], 4),
                "Macro Recall": round(stm["recall_macro"], 4),
                "Macro F1": round(stm["macro_f1"], 4),
                "Obs Top-1 Acc": round(stm["obs_top1_accuracy"], 4),
                "Log Loss": round(stm["log_loss"], 4),
                "ROC-AUC": round(stm["roc_auc"], 4) if stm.get("roc_auc") is not None else None,
                "Miss Rate (%)": round(stm["candidate_miss_rate"] * 100, 2),
                "Obs Continuity": round(stm["obs_step_continuity"], 4),
                "Infer (ms/trip)": round(stm["infer_ms_per_trip"], 2),
            }
        )
    comp_df = pd.DataFrame(table_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Interactive comparison charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("Classification & Top-1 Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=130)
        names = comp_df["Model"].tolist()
        x = np.arange(len(names))
        w = 0.25
        ax.bar(x - w, comp_df["Candidate Acc"], width=w, label="Candidate Acc", color="#1f77b4")
        ax.bar(x, comp_df["Macro F1"], width=w, label="Macro F1", color="#2ca02c")
        ax.bar(x + w, comp_df["Obs Top-1 Acc"], width=w, label="Obs Top-1 Acc", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0.60, 1.03)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with ch2:
        st.subheader("Log Loss vs Inference Latency")
        fig, ax1 = plt.subplots(figsize=(6.8, 4.2), dpi=130)
        ax2 = ax1.twinx()
        names = comp_df["Model"].tolist()
        x = np.arange(len(names))
        ax1.bar(x - 0.18, comp_df["Log Loss"], width=0.34, label="Log Loss (lower better)", color="#d62728", alpha=0.85)
        ax2.bar(x + 0.18, comp_df["Infer (ms/trip)"], width=0.34, label="Infer ms/trip", color="#9467bd", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha="right")
        ax1.set_ylabel("Log Loss", color="#d62728")
        ax2.set_ylabel("Inference ms / trip", color="#9467bd")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Inspect Individual Model Details, Confusion Matrix & Per-Trip Performance")
    model_options = {m["model_id"]: f"{m['display_name']} ({m['model_id']})" for m in models}
    selected_mid = st.selectbox(
        "Select Model to Inspect",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
    )
    sel_m = next(m for m in models if m["model_id"] == selected_mid)
    stm = sel_m["student_test"]

    d1, d2 = st.columns([1.0, 1.3])
    with d1:
        if stm.get("confusion_matrix"):
            fig_cm = _plot_confusion_matrix_fig(
                stm["confusion_matrix"],
                f"{sel_m['display_name']} — Student-Test Confusion Matrix",
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        with st.expander("Architecture & Hyperparameter Configuration"):
            st.write(f"**Family:** {sel_m['family']}")
            st.write(f"**Architecture:** {sel_m['architecture']}")
            st.write(f"**Artifact:** `{sel_m.get('artifact_path', 'N/A')}`")
            if sel_m.get("config"):
                st.json(sel_m["config"])

    with d2:
        st.markdown(f"**Per-Trip Student-Test Evaluation — `{sel_m['display_name']}`**")
        if not per_trip_all.empty and "model_id" in per_trip_all.columns:
            sub_pt = per_trip_all[per_trip_all["model_id"] == selected_mid].drop(columns=["model_id"])
            st.dataframe(sub_pt, use_container_width=True, hide_index=True)
        else:
            st.info("Per-trip CSV breakdown unavailable.")

    # Transparent teammate branch status
    tb_status = eval_data.get("teammate_branch_status") or []
    if tb_status:
        st.markdown("---")
        st.subheader("Other Team Branch Models (Status & Comparability Disclosure)")
        st.caption(
            "These models reside on separate teammate Git branches with a different 6-feature schema or "
            "incomplete artifacts; they are reported separately here so incompatible setups are never mixed."
        )
        for item in tb_status:
            with st.expander(f"{item['owner']} — Branch `{item['branch']}`"):
                st.write(f"**Models:** {', '.join(item.get('models', []))}")
                st.write(f"**Status:** {item.get('status', '')}")
                if item.get("reported_holdout_metrics"):
                    st.json(item["reported_holdout_metrics"])


# ---------------------------------------------------------------------------
# PAGE 3: GPS MAP MATCHING (LIVE FASTAPI + FOLIUM)
# ---------------------------------------------------------------------------
def render_gps_map_matching(api_url: str, models_data: Dict[str, Any]) -> None:
    st.title("GPS Map Matching & Folium Route Visualization")
    st.caption(
        "Select a real trip and model below to run live candidate generation and model inference on Laptop 1 "
        "and visualize the original GPS trajectory, candidate road segments, predicted route, and optional reference route."
    )

    client = PathSnapAPIClient(base_url=api_url)
    available_models = models_data.get("models") or []
    model_ids = [m["model_id"] for m in available_models] or ["mlp_128_64_32"]
    model_labels = {m["model_id"]: f"{m['display_name']} ({m['model_id']})" for m in available_models}

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.8, 1.8])
    with ctrl1:
        partition = st.selectbox("Dataset Partition", options=["student_test", "train", "evaluator_test"], index=0)

    trips: List[Dict[str, Any]] = []
    try:
        trips = _fetch_trips(api_url, partition)
    except APIClientError as exc:
        st.error(f"Could not fetch trips for `{partition}`: {exc}")

    with ctrl2:
        if trips:
            trip_fmt = {
                t["trip_id"]: f"{t['trip_id']} ({t['num_observations']:,} obs, {t['num_arcs']:,} arcs)"
                for t in trips
            }
            selected_trip = st.selectbox(
                f"Select Trip ({len(trips)} available)",
                options=[t["trip_id"] for t in trips],
                format_func=lambda tid: trip_fmt[tid],
            )
        else:
            st.selectbox("Select Trip", options=[f"No trips found in '{partition}'"], disabled=True)
            selected_trip = None

    with ctrl3:
        default_idx = 0
        best_id = models_data.get("best_model_id")
        if best_id in model_ids:
            default_idx = model_ids.index(best_id)
        selected_model = st.selectbox(
            "Select ML Model",
            options=model_ids,
            index=default_idx,
            format_func=lambda mid: model_labels.get(mid, mid),
        )

    opt1, opt2, opt3 = st.columns([1.6, 1.8, 1.2], vertical_alignment="bottom")
    with opt1:
        show_candidates = st.checkbox("Overlay Generated Candidate Road Segments", value=True)
    with opt2:
        allow_ref = partition != "evaluator_test"
        show_reference = st.checkbox(
            "Compare Against Reference Route (when available)",
            value=allow_ref,
            disabled=not allow_ref,
            help="Automatically disabled for evaluator_test to keep evaluator ground truth strictly server-side.",
        )
    with opt3:
        run_btn = st.button(
            "Run Map Matching",
            type="primary",
            use_container_width=True,
            disabled=(selected_trip is None),
        )

    if run_btn and selected_trip is not None:
        progress = st.progress(15, text=f"Sending trip `{selected_trip}` to Laptop 1 ({selected_model})...")
        try:
            with st.spinner("Generating candidates, running model inference, and reconstructing route geometry..."):
                progress.progress(55, text="Executing ML scoring on Laptop 1 FastAPI...")
                pred_resp = client.run_prediction(
                    trip_id=selected_trip,
                    model_name=selected_model,
                    partition=partition,
                    include_reference=show_reference,
                    include_candidate_segments=show_candidates,
                    include_observations=False,
                )
                progress.progress(100, text="Route reconstruction complete!")
            st.session_state["map_matching_result"] = pred_resp
            st.session_state["map_matching_error"] = None
        except APIClientError as exc:
            st.session_state["map_matching_error"] = str(exc)
        finally:
            progress.empty()

    if st.session_state.get("map_matching_error"):
        st.error(st.session_state["map_matching_error"])

    res = st.session_state.get("map_matching_result")
    if not res:
        st.info("Choose a trip and model above, then click **Run Map Matching** to render the live Folium map.")
        return

    metrics = res.get("metrics") or {}
    cont = res.get("route_continuity") or {}

    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("GPS Observations", f"{res.get('num_observations', 0):,}")
    k2.metric("Candidates Scored", f"{res.get('num_candidates', 0):,}")
    k3.metric("Obs Top-1 Accuracy", _fmt_pct(metrics.get("obs_top1_accuracy")))
    k4.metric("Candidate Macro F1", _fmt_pct(metrics.get("macro_f1")))
    k5.metric("Obs-Step Continuity", _fmt_pct(cont.get("obs_step_continuity")))
    k6.metric("Model Inference", f"{res.get('inference_time_ms', 0.0):.1f} ms")

    map_col, info_col = st.columns([2.5, 1.0])
    with map_col:
        fmap = build_prediction_map(
            gps_points=res.get("gps_points") or [],
            predicted_segments=res.get("predicted_segments") or [],
            trip_id=str(res.get("trip_id", "")),
            model_label=str(res.get("model_display_name", res.get("model_name", ""))),
            reference_segments=res.get("reference_segments"),
            candidate_segments=res.get("candidate_segments"),
            selected_candidate_segments=res.get("selected_candidate_segments"),
            top1_accuracy=metrics.get("obs_top1_accuracy"),
            step_continuity=cont.get("obs_step_continuity"),
        )
        st_folium(fmap, width=None, height=580, returned_objects=[])

    with info_col:
        st.subheader("Trip Prediction Summary")
        st.markdown(
            f"""
            - **Trip ID:** `{res.get('trip_id')}` (`{res.get('partition')}`)
            - **Model:** `{res.get('model_display_name')}`
            - **Unique Predicted Segments:** `{res.get('num_predicted_segments', 0):,}`
            - **Candidate Extraction:** `{res.get('candidate_extraction_ms', 0.0):.1f} ms`
            - **Inference Latency:** `{res.get('inference_time_ms', 0.0):.1f} ms`
            - **Candidate Accuracy:** `{_fmt_pct(metrics.get('candidate_accuracy'))}`
            - **Log Loss:** `{_fmt_num(metrics.get('log_loss'), 4)}`
            - **Candidate Miss Rate:** `{_fmt_pct(metrics.get('candidate_miss_rate'))}`
            - **Compressed Continuity:** `{_fmt_pct(cont.get('compressed_route_continuity'))}`
            - **Disconnected Steps:** `{cont.get('disconnected_transitions', 'N/A')}`
            - **Reference Overlay:** `{'Displayed' if res.get('reference_segments') else 'Not displayed'}`
            """
        )

        # First 25 observation predictions table
        pred_arcs = res.get("predicted_arc_ids") or []
        top1_mp = res.get("top1_match_probs") or []
        top1_ep = res.get("top1_emission_probs") or []
        top1_lp = res.get("top1_log_emission_probs") or []
        if pred_arcs:
            preview_n = min(30, len(pred_arcs))
            preview_df = pd.DataFrame(
                {
                    "obs_idx": list(range(preview_n)),
                    "predicted_arc_id": pred_arcs[:preview_n],
                    "match_prob": top1_mp[:preview_n],
                    "emission_prob": top1_ep[:preview_n],
                    "log_emission_prob": top1_lp[:preview_n],
                }
            )
            st.markdown("**Predicted Segments (First 30 Observations)**")
            st.dataframe(preview_df, use_container_width=True, hide_index=True, height=240)

    with st.expander("Inspect FastAPI Response Summary (JSON)"):
        sanitized = {
            k: v
            for k, v in res.items()
            if k
            not in (
                "gps_points",
                "predicted_segments",
                "candidate_segments",
                "selected_candidate_segments",
                "reference_segments",
                "predicted_arc_ids",
                "top1_match_probs",
                "top1_emission_probs",
                "top1_log_emission_probs",
            )
        }
        st.json(sanitized)


# ---------------------------------------------------------------------------
# PAGE 4: EVALUATION (CM, PR, REAL ROC, PER-TRIP, CONTINUITY, DOWNLOADS)
# ---------------------------------------------------------------------------
def render_evaluation_page(eval_data: Dict[str, Any]) -> None:
    st.title("Model Evaluation & Diagnostics (10 Held-Out Student-Test Trips)")
    models = eval_data.get("models") or []
    curves = eval_data.get("roc_pr_curves") or {}
    per_trip_df = pd.DataFrame(eval_data.get("per_trip") or [])

    if not models:
        st.warning("No evaluation data available from backend.")
        return

    model_map = {m["model_id"]: m for m in models}
    selected_mid = st.selectbox(
        "Select Model for Detailed Evaluation",
        options=list(model_map.keys()),
        format_func=lambda k: f"{model_map[k]['display_name']} ({k})",
    )
    m = model_map[selected_mid]
    stm = m["student_test"]
    cvm = m["cv"]
    m_curve = curves.get(selected_mid) or {}

    # Top KPIs
    r1 = st.columns(6)
    r1[0].metric("Candidate Accuracy", _fmt_pct(stm.get("candidate_accuracy")))
    r1[1].metric("Obs-Level Top-1 Acc", _fmt_pct(stm.get("obs_top1_accuracy")))
    r1[2].metric("Macro F1-Score", _fmt_pct(stm.get("macro_f1")))
    r1[3].metric("ROC-AUC", _fmt_num(stm.get("roc_auc"), 4))
    r1[4].metric("PR-AUC (Avg Prec)", _fmt_num(stm.get("pr_auc"), 4))
    r1[5].metric("Candidate Miss Rate", _fmt_pct(stm.get("candidate_miss_rate")))

    r2 = st.columns(5)
    r2[0].metric("Positive Precision", _fmt_pct(stm.get("precision_positive")))
    r2[1].metric("Positive Recall", _fmt_pct(stm.get("recall_positive")))
    r2[2].metric("Positive F1-Score", _fmt_pct(stm.get("binary_f1_positive")))
    r2[3].metric("Obs-Step Continuity", _fmt_pct(stm.get("obs_step_continuity")))
    r2[4].metric("Compressed Continuity", _fmt_pct(stm.get("compressed_route_continuity")))

    st.markdown("---")
    c_left, c_right = st.columns([1.0, 1.35])
    with c_left:
        st.subheader("Candidate-Level Confusion Matrix")
        if stm.get("confusion_matrix"):
            fig_cm = _plot_confusion_matrix_fig(
                stm["confusion_matrix"], f"{m['display_name']} — Student-Test"
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)

    with c_right:
        st.subheader("ROC & Precision-Recall Curves (Actual Probabilities)")
        if m_curve.get("roc_curve") and m_curve.get("pr_curve"):
            fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=130)
            ax_r.plot(
                m_curve["roc_curve"]["fpr"],
                m_curve["roc_curve"]["tpr"],
                color="#1f77b4",
                linewidth=2.2,
                label=f"AUC = {m_curve['roc_auc']:.4f}",
            )
            ax_r.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax_r.set_title("ROC Curve", fontweight="bold")
            ax_r.set_xlabel("False Positive Rate")
            ax_r.set_ylabel("True Positive Rate")
            ax_r.grid(True, linestyle="--", alpha=0.4)
            ax_r.legend(loc="lower right")

            ax_p.plot(
                m_curve["pr_curve"]["recall"],
                m_curve["pr_curve"]["precision"],
                color="#2ca02c",
                linewidth=2.2,
                label=f"AP = {m_curve['pr_auc']:.4f}",
            )
            ax_p.set_title("Precision-Recall Curve", fontweight="bold")
            ax_p.set_xlabel("Recall (Match Class)")
            ax_p.set_ylabel("Precision (Match Class)")
            ax_p.grid(True, linestyle="--", alpha=0.4)
            ax_p.legend(loc="lower left")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Probability curves unavailable for this model.")

    st.markdown("---")
    st.subheader(f"Per-Trip Performance Breakdown — `{m['display_name']}`")
    sub_pt = pd.DataFrame()
    if not per_trip_df.empty and "model_id" in per_trip_df.columns:
        sub_pt = per_trip_df[per_trip_df["model_id"] == selected_mid].copy()
        st.dataframe(sub_pt.drop(columns=["model_id"]), use_container_width=True, hide_index=True)

        fig_pt, ax = plt.subplots(figsize=(11, 3.8), dpi=130)
        x = np.arange(len(sub_pt))
        w = 0.27
        ax.bar(x - w, sub_pt["top1_accuracy_obs"], width=w, label="Obs Top-1 Acc", color="#1f77b4")
        ax.bar(x, sub_pt["obs_step_continuity"], width=w, label="Obs-Step Continuity", color="#2ca02c")
        ax.bar(x + w, sub_pt["compressed_route_continuity"], width=w, label="Compressed Continuity", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(sub_pt["trip_id"].tolist(), rotation=20)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")
        fig_pt.tight_layout()
        st.pyplot(fig_pt)
        plt.close(fig_pt)

    # Downloadable CSV & JSON evaluation exports
    st.markdown("---")
    st.subheader("Download Verified Evaluation Artifacts")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        if not sub_pt.empty:
            st.download_button(
                label=f"Download Per-Trip CSV ({selected_mid})",
                data=sub_pt.to_csv(index=False).encode("utf-8"),
                file_name=f"pathsnap_per_trip_{selected_mid}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with dl2:
        export_json = {
            "model_id": selected_mid,
            "display_name": m["display_name"],
            "cv_metrics": cvm,
            "student_test_metrics": stm,
            "roc_auc": m_curve.get("roc_auc"),
            "pr_auc": m_curve.get("pr_auc"),
        }
        st.download_button(
            label=f"Download Metrics JSON ({selected_mid})",
            data=json.dumps(export_json, indent=2).encode("utf-8"),
            file_name=f"pathsnap_evaluation_{selected_mid}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            label="Download Full Multi-Model Comparison JSON",
            data=json.dumps(eval_data, indent=2).encode("utf-8"),
            file_name="pathsnap_full_model_comparison.json",
            mime="application/json",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# PAGE 5: SYSTEM INFORMATION
# ---------------------------------------------------------------------------
def render_system_info(
    api_url: str,
    health_data: Optional[Dict[str, Any]],
    models_data: Optional[Dict[str, Any]],
    eval_data: Optional[Dict[str, Any]],
    conn_error: Optional[str],
) -> None:
    st.title("System Information & Two-Laptop Diagnostics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frontend Version", APP_VERSION)
    col2.metric("API Version", (health_data or {}).get("api_version", "Offline"))
    col3.metric("Backend Status", "Online" if health_data else "Offline")
    col4.metric(
        "Models Loaded",
        f"{len((health_data or {}).get('available_models', []))}" if health_data else "0",
    )

    st.subheader("FastAPI Backend Connection & Health Status")
    st.write(f"**Configured `PATHSNAP_API_URL`:** `{api_url}`")
    if health_data:
        st.success("Laptop 1 FastAPI backend is reachable and healthy.")
        st.json(health_data)
    else:
        _render_offline_banner(api_url, conn_error or "Connection failed.")

    if models_data:
        st.subheader("Loaded Model Architectures & Configurations")
        for m in models_data.get("models", []):
            with st.expander(f"{m['display_name']} (`{m['model_id']}`) — Loaded: {m['loaded']}"):
                st.write(f"**Family:** {m['family']}")
                st.write(f"**Architecture:** {m['architecture']}")
                st.write(f"**Features:** {m['num_features']}")
                st.write(f"**Artifact Path:** `{m['artifact_path']}`")
                if m.get("config"):
                    st.json(m["config"])

        st.subheader("Canonical 28-Feature Input Schema")
        fcols = models_data.get("feature_columns") or []
        st.dataframe(
            pd.DataFrame({"Index": list(range(1, len(fcols) + 1)), "Feature Name": fcols}),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

    if eval_data and eval_data.get("dataset_summary"):
        st.subheader("Dataset Partition Summary")
        st.json(eval_data["dataset_summary"])


# ---------------------------------------------------------------------------
# MAIN APPLICATION ROUTER
# ---------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.header("PathSnap Navigation")
        page = st.radio(
            "Select Page",
            options=[
                "Dashboard",
                "Model Comparison",
                "GPS Map Matching",
                "Evaluation",
                "System Information",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Two-Laptop API Config")
        api_url = st.text_input(
            "Laptop 1 FastAPI URL",
            value=get_default_api_url(),
            help="Set via PATHSNAP_API_URL env var or override here (e.g., http://192.168.1.100:8000).",
        ).rstrip("/")

        if st.button("Test / Refresh Connection", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        health_data: Optional[Dict[str, Any]] = None
        conn_error: Optional[str] = None
        try:
            health_data = _fetch_health(api_url)
            st.success(f"Connected (`v{health_data.get('api_version', '?')}`)")
            st.caption(
                f"Models: {', '.join(health_data.get('available_models', []))} | "
                f"Startup: {health_data.get('startup_time_s', 0)}s"
            )
        except APIClientError as exc:
            conn_error = str(exc)
            st.error("API Offline")

        st.divider()
        st.caption(f"PathSnap Evaluator UI v{APP_VERSION} • Person 4 (Kruthik Reddy)")

    models_data: Optional[Dict[str, Any]] = None
    eval_data: Optional[Dict[str, Any]] = None
    if health_data is not None:
        try:
            models_data = _fetch_models(api_url)
            eval_data = _fetch_evaluation(api_url)
        except APIClientError as exc:
            conn_error = str(exc)

    if page == "System Information":
        render_system_info(api_url, health_data, models_data, eval_data, conn_error)
        return

    if health_data is None or models_data is None or eval_data is None:
        st.title(f"PathSnap — {page}")
        _render_offline_banner(api_url, conn_error or "Could not reach Laptop 1 FastAPI backend.")
        return

    if page == "Dashboard":
        render_dashboard(api_url, eval_data, models_data)
    elif page == "Model Comparison":
        render_model_comparison(eval_data)
    elif page == "GPS Map Matching":
        render_gps_map_matching(api_url, models_data)
    elif page == "Evaluation":
        render_evaluation_page(eval_data)


if __name__ == "__main__":
    main()
