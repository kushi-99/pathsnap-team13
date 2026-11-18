"""Streamlit front end for the PathSnap FastAPI scoring service."""
import os

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="PathSnap | GPS map matching", page_icon="🧭", layout="wide")
st.title("PathSnap")
st.caption("Inspect GPS trips and compare candidate-road scores from the Logistic Regression and KNN baselines.")
with st.sidebar:
    st.header("Trip and model")
    split = "evaluator_test"
    st.caption("Final evaluation dataset: evaluator_test")
    try:
        response = requests.get(f"{API_URL}/trips", params={"split": split}, timeout=10)
        response.raise_for_status()
        trip_ids = response.json()["trips"]
    except requests.RequestException as error:
        st.error(f"API unavailable at {API_URL}. Start the FastAPI server first. Details: {error}")
        st.stop()
    trip_id = st.selectbox("Trip", trip_ids, index=0)
    model = st.selectbox("Baseline model", ["logistic_regression", "knn"], format_func=lambda value: "Logistic Regression" if value == "logistic_regression" else "KNN")
    run = st.button("Score trip", type="primary", use_container_width=True)

if run:
    with st.spinner("Generating road candidates and scoring GPS observations…"):
        try:
            response = requests.get(f"{API_URL}/score", params={"split": split, "trip_id": trip_id, "model": model}, timeout=300)
            response.raise_for_status()
            st.session_state["scored_trip"] = response.json()
        except requests.RequestException as error:
            st.error(f"Trip scoring failed: {error}")

result = st.session_state.get("scored_trip")
if result and result["trip_id"] == trip_id and result["split"] == split and result["model"] == model:
    left, middle, right = st.columns(3)
    left.metric("GPS observations", f"{len(result['observations']):,}")
    middle.metric("Candidate roads scored", f"{result['candidate_count']:,}")
    right.metric("Model", "Logistic Regression" if model == "logistic_regression" else "KNN")
    if result.get("evaluation"):
        st.caption(
            "Evaluator route-alignment Top-1: "
            f"{result['evaluation']['top1_accuracy_all']:.1%} · candidate coverage: "
            f"{result['evaluation']['candidate_coverage']:.1%}"
        )
    gps = pd.DataFrame(result["gps"], columns=["lat", "lon"])
    roads = pd.DataFrame([{"path": [[lon, lat] for lat, lon in line]} for line in result["selected_roads"]])
    layers = [pdk.Layer("PathLayer", data=[{"path": gps[["lon", "lat"]].values.tolist()}], get_path="path", get_color=[15, 118, 140], width_scale=12, width_min_pixels=3), pdk.Layer("PathLayer", data=roads.to_dict("records"), get_path="path", get_color=[229, 129, 70], width_scale=12, width_min_pixels=4)]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=float(gps.lat.mean()), longitude=float(gps.lon.mean()), zoom=13, pitch=0), tooltip={"text": "GPS path (blue), model-selected road sequence (orange)"}), use_container_width=True)
    st.subheader("Top road candidate per observation")
    st.dataframe(pd.DataFrame(result["observations"]), use_container_width=True, hide_index=True)
else:
    st.info("Choose a split, trip, and baseline, then select Score trip.")
