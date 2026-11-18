"""PathSnap Evaluator Dashboard (Laptop 2 Client UI).

Run locally or on Laptop 2:
    streamlit run ui/evaluator_dashboard/app.py

Connects dynamically to Laptop 1 FastAPI backend (GET /health, GET /trips, POST /predict).
Never loads or displays evaluator ground-truth labels.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st
from streamlit_folium import st_folium

from api_client import APIClientError, PathSnapAPIClient
from map_renderer import build_prediction_map
from mock_data import get_layout_preview_prediction, get_layout_preview_trips


st.set_page_config(
    page_title="PathSnap — Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _fmt_metric(val: Optional[float], as_pct: bool = True, decimals: int = 2) -> str:
    if val is None:
        return "N/A (API Only)"
    if as_pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"


def main() -> None:
    # 1. Header — PathSnap Title & Project Description
    st.title("PathSnap — Distributed Map-Matching & Route Reconstruction")
    st.caption(
        "Evaluator Dashboard (Laptop 2 Client) • Reconstructs realistic vehicle routes by matching noisy GPS "
        "observations to road-network segments via Laptop 1 FastAPI inference and HMM/Viterbi decoding."
    )

    # Sidebar — API Connection Configuration & Status
    with st.sidebar:
        st.header("Laptop 1 API Connection")
        default_url = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")
        api_url = st.text_input(
            "FastAPI Base URL",
            value=default_url,
            help="Enter Laptop 1's LAN address, e.g., http://192.168.1.10:8000",
        )

        # Layout-preview toggle strictly OFF by default and clearly separated
        layout_preview_mode = st.toggle(
            "UI Layout Preview Mode (Mock Data)",
            value=False,
            help="OFF by default. Enable ONLY for visual layout checks when Laptop 1 FastAPI is offline.",
        )

        client = PathSnapAPIClient(base_url=api_url)
        api_healthy = False
        health_payload: Dict[str, Any] = {}

        if layout_preview_mode:
            st.warning(
                "**Layout Preview Mode Active** — Using synthetic placeholder coordinates. "
                "Turn OFF to fetch real trips and predictions from Laptop 1 FastAPI."
            )
        else:
            try:
                health_payload = client.check_health()
                api_healthy = True
                st.success(f"Connected to Laptop 1 API (`{api_url}`)")
                if health_payload:
                    st.caption(
                        f"Model: `{health_payload.get('model_name', 'Unknown')}` | "
                        f"Decoder: `{health_payload.get('decoder', 'Unknown')}`"
                    )
            except APIClientError as exc:
                st.error(f"API Disconnected: {exc}")

        if st.button("Refresh Trip List", use_container_width=True):
            st.session_state.pop("cached_trips", None)
            st.rerun()

        st.divider()
        st.markdown(
            "**Two-Laptop Architecture**\n"
            "- **Laptop 1:** FastAPI + Model + HMM/Viterbi + Road Network + Server-side Evaluation\n"
            "- **Laptop 2 (This UI):** Dynamic Trip Selection + Visualization + Metrics Display"
        )

    # 2. Dynamically Fetch Ten Selectable Evaluation Trips from API
    trips: List[Dict[str, Any]] = []
    trip_fetch_error: Optional[str] = None

    if layout_preview_mode:
        trips = get_layout_preview_trips()
    elif api_healthy:
        if "cached_trips" not in st.session_state or st.session_state.get("cached_api_url") != api_url:
            try:
                st.session_state["cached_trips"] = client.list_trips()
                st.session_state["cached_api_url"] = api_url
            except APIClientError as exc:
                trip_fetch_error = str(exc)
        trips = st.session_state.get("cached_trips", [])
    else:
        trip_fetch_error = "Waiting for Laptop 1 FastAPI connection before loading evaluation trips."

    # Trip Selector & Run Prediction Control Bar
    ctrl_col1, ctrl_col2 = st.columns([3, 1], vertical_alignment="bottom")

    with ctrl_col1:
        if trips:
            trip_labels = [
                f"{t['trip_id']} ({t['num_observations']} GPS obs)"
                if "num_observations" in t
                else str(t["trip_id"])
                for t in trips
            ]
            selected_idx = st.selectbox(
                f"Select Evaluation Trip ({len(trips)} available from API)",
                options=list(range(len(trips))),
                format_func=lambda i: trip_labels[i],
            )
            selected_trip_id = str(trips[selected_idx]["trip_id"])
        else:
            st.selectbox(
                "Select Evaluation Trip",
                options=["No trips loaded — connect Laptop 1 API"],
                disabled=True,
            )
            selected_trip_id = None

    with ctrl_col2:
        run_clicked = st.button(
            "Run Prediction",
            type="primary",
            use_container_width=True,
            disabled=(selected_trip_id is None),
        )

    if trip_fetch_error:
        st.info(trip_fetch_error)

    # 3 & 4. Run Prediction with Loading / Progress Indicator
    if run_clicked and selected_trip_id is not None:
        if layout_preview_mode:
            with st.spinner(f"Generating layout preview for {selected_trip_id}..."):
                st.session_state["last_result"] = get_layout_preview_prediction(selected_trip_id)
                st.session_state["last_error"] = None
        else:
            progress = st.progress(10, text=f"Sending trip `{selected_trip_id}` to Laptop 1 FastAPI...")
            try:
                with st.spinner(f"Running candidate extraction, model inference, and route reconstruction on Laptop 1..."):
                    progress.progress(45, text="Executing ML scoring & HMM/Viterbi decoding on Laptop 1...")
                    result = client.run_prediction(selected_trip_id)
                    progress.progress(100, text="Prediction complete!")
                st.session_state["last_result"] = result
                st.session_state["last_error"] = None
            except APIClientError as exc:
                st.session_state["last_error"] = str(exc)
            finally:
                progress.empty()

    if st.session_state.get("last_error"):
        st.error(f"Prediction Request Error: {st.session_state['last_error']}")

    result: Optional[Dict[str, Any]] = st.session_state.get("last_result")
    if not result:
        st.markdown("---")
        st.info("Select an evaluation trip above and click **Run Prediction** to view the reconstructed route and metrics.")
        return

    if result.get("is_mock"):
        st.warning(
            "**LAYOUT PREVIEW ONLY (MOCK DATA):** Coordinates are synthetic placeholders and metrics are intentionally blank. "
            "Disable *UI Layout Preview Mode* to evaluate real trips against Laptop 1."
        )

    # 7 & 8. Model Prediction Metrics + Route Continuity Information
    st.markdown("---")
    st.subheader(f"Evaluation Results — Trip `{result.get('trip_id', 'N/A')}`")

    metrics = result.get("metrics", {}) or {}
    continuity = result.get("route_continuity", {}) or {}

    m_cols = st.columns(5)
    m_cols[0].metric(
        "Obs-Level Top-1 Accuracy",
        _fmt_metric(metrics.get("obs_top1_accuracy"), as_pct=True),
    )
    m_cols[1].metric(
        "Macro F1-Score",
        _fmt_metric(metrics.get("macro_f1"), as_pct=False, decimals=4),
    )
    m_cols[2].metric(
        "Log Loss",
        _fmt_metric(metrics.get("log_loss"), as_pct=False, decimals=4),
    )
    m_cols[3].metric(
        "Route Continuity (Step)",
        _fmt_metric(continuity.get("obs_step_continuity"), as_pct=True),
    )
    m_cols[4].metric(
        "Inference Latency",
        f"{result.get('inference_time_ms', 0.0):.1f} ms"
        if result.get("inference_time_ms") is not None
        else "N/A",
    )

    # 5 & 6. Interactive Folium Map: Original GPS Trajectory + Predicted Road Route
    map_col, summary_col = st.columns([2.4, 1.0])

    with map_col:
        st.subheader("Geographical Trajectory & Reconstructed Route (Folium)")
        gps_pts = result.get("gps_points", [])
        pred_segs = result.get("predicted_segments", [])
        fmap = build_prediction_map(
            gps_points=gps_pts,
            predicted_segments=pred_segs,
            trip_id=str(result.get("trip_id", "Unknown")),
            model_label=str(result.get("model_name", "PathSnap")),
        )
        st_folium(fmap, width=None, height=540, returned_objects=[])

    # 9. Prediction Summary & Continuity Details
    with summary_col:
        st.subheader("Prediction Summary")
        st.markdown(
            f"""
            - **Trip ID:** `{result.get('trip_id', 'N/A')}`
            - **Model:** `{result.get('model_name', 'N/A')}`
            - **Decoder:** `{result.get('decoder', 'N/A')}`
            - **GPS Observations:** `{result.get('num_observations', len(result.get('gps_points', [])))}`
            - **Predicted Route Segments:** `{result.get('num_predicted_segments', len(result.get('predicted_segments', [])))}`
            - **Candidate Miss Rate:** `{_fmt_metric(metrics.get('candidate_miss_rate'), as_pct=True)}`
            """
        )

        st.subheader("Route Continuity")
        st.markdown(
            f"""
            - **Observation-Step Continuity:** `{_fmt_metric(continuity.get('obs_step_continuity'), as_pct=True)}`
            - **Compressed Route Continuity:** `{_fmt_metric(continuity.get('compressed_route_continuity'), as_pct=True)}`
            - **Disconnected Transitions:** `{continuity.get('disconnected_transitions', 'N/A')}`
            """
        )

        if result.get("summary_note"):
            st.caption(result["summary_note"])

        with st.expander("Inspect Raw API Response (Sanitized)"):
            preview_payload = {
                k: v for k, v in result.items() if k not in ("gps_points", "predicted_segments")
            }
            preview_payload["gps_points_count"] = len(result.get("gps_points", []))
            preview_payload["predicted_segments_count"] = len(result.get("predicted_segments", []))
            st.json(preview_payload)


if __name__ == "__main__":
    main()
