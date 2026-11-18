"""Automated integration and verification tests for PathSnap Review 2.

Tests real artifacts, real student_test trips, FastAPI endpoints, Folium rendering,
API client error handling, and two-laptop configurable URL resolution.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
UI_DIR = REPO_ROOT / "ui" / "evaluator_dashboard"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from api.main import app, registry
from api_client import APIClientError, PathSnapAPIClient, get_default_api_url
from map_renderer import build_prediction_map
from ml.mlp.config import ARTIFACTS_DIR, COMPARISON_DIR, RESULTS_DIR
from ml.mlp.predict import MLPPredictor


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient executing startup lifespan once."""
    with TestClient(app) as test_client:
        yield test_client


def test_mlp_artifact_loading():
    predictor = MLPPredictor.from_artifacts(ARTIFACTS_DIR, device="cpu")
    assert predictor.model_name == "mlp_128_64_32"
    assert len(predictor.feature_columns) == 28


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert "mlp_128_64_32" in data["available_models"]
    assert data["data_root_available"] is True
    assert data["available_trips_count"] == 10


def test_models_endpoint(client: TestClient):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["feature_columns"]) == 28
    model_ids = [m["model_id"] for m in data["models"]]
    assert "mlp_128_64_32" in model_ids
    for m in data["models"]:
        assert m["loaded"] is True
        assert m["num_features"] == 28


def test_trips_listing_and_detail(client: TestClient):
    resp = client.get("/trips", params={"partition": "student_test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 10
    trip_ids = [t["trip_id"] for t in data["trips"]]
    assert "00000052" in trip_ids
    assert "00000000" in trip_ids

    detail_resp = client.get("/trips/00000052", params={"partition": "student_test"})
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["trip_id"] == "00000052"
    assert detail["num_observations"] == 262
    assert len(detail["gps_points"]) == 262
    # Verify [lat, lon, ts] coordinate bounds are plausible
    lat0, lon0 = detail["gps_points"][0][0], detail["gps_points"][0][1]
    assert -90.0 <= lat0 <= 90.0
    assert -180.0 <= lon0 <= 180.0

    # Unknown trip returns 404
    missing = client.get("/trips/99999999", params={"partition": "student_test"})
    assert missing.status_code == 404


def test_predict_candidate_records_mode(client: TestClient):
    sample_req_path = REPO_ROOT / "api_integration" / "sample_request.json"
    sample_req = json.loads(sample_req_path.read_text(encoding="utf-8"))
    payload = {
        "trip_id": sample_req["trip_id"],
        "model_name": "mlp_128_64_32",
        "candidates": sample_req["candidates"],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_observations"] == 2
    assert data["num_candidates"] == len(sample_req["candidates"])
    assert data["predicted_arc_ids"] == [17630, 17630]
    assert data["observations"] is not None
    # Emission probabilities per observation must sum to ~1.0
    for obs in data["observations"]:
        ep_sum = sum(c["emission_prob"] for c in obs["candidates"])
        assert abs(ep_sum - 1.0) < 1e-4

    # Validation error for missing features
    bad_resp = client.post(
        "/predict",
        json={
            "model_name": "mlp_128_64_32",
            "candidates": [{"obs_idx": 0, "arc_id": 1, "features": {"perp_dist_m": 1.0}}],
        },
    )
    assert bad_resp.status_code == 422

    # Unknown model returns 404
    unknown_m = client.post(
        "/predict",
        json={"trip_id": "00000052", "model_name": "nonexistent_model"},
    )
    assert unknown_m.status_code == 404


def test_predict_trip_mode_and_folium_rendering(client: TestClient):
    resp = client.post(
        "/predict",
        json={
            "trip_id": "00000052",
            "partition": "student_test",
            "model_name": "mlp_128_64_32",
            "include_reference": True,
            "include_candidate_segments": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_observations"] == 262
    assert len(data["gps_points"]) == 262
    assert len(data["predicted_segments"]) > 0
    assert len(data["candidate_segments"]) > 0
    assert data["reference_segments"] is not None and len(data["reference_segments"]) > 0
    assert data["metrics"]["obs_top1_accuracy"] > 0.90
    assert data["route_continuity"]["obs_step_continuity"] > 0.90

    # Render Folium map WITH reference route
    fmap_with_ref = build_prediction_map(
        gps_points=data["gps_points"],
        predicted_segments=data["predicted_segments"],
        trip_id="00000052",
        model_label=data["model_display_name"],
        reference_segments=data["reference_segments"],
        candidate_segments=data["candidate_segments"],
        selected_candidate_segments=data["selected_candidate_segments"],
        top1_accuracy=data["metrics"]["obs_top1_accuracy"],
        step_continuity=data["route_continuity"]["obs_step_continuity"],
    )
    html_with_ref = fmap_with_ref.get_root().render()
    assert "Reference Route" in html_with_ref
    assert "Predicted Matched Route" in html_with_ref

    # Render Folium map WITHOUT reference route (graceful fallback)
    fmap_no_ref = build_prediction_map(
        gps_points=data["gps_points"],
        predicted_segments=data["predicted_segments"],
        trip_id="00000052",
        model_label=data["model_display_name"],
        reference_segments=None,
    )
    html_no_ref = fmap_no_ref.get_root().render()
    assert "Not displayed / server-side" in html_no_ref


def test_evaluate_endpoints_and_verified_mlp_metrics(client: TestClient):
    # Dataset-level all models
    resp_all = client.post("/evaluate", json={"partition": "student_test"})
    assert resp_all.status_code == 200
    data_all = resp_all.json()
    assert data_all["scope"] == "dataset_all_models"
    assert len(data_all["models"]) >= 1

    # Verify saved MLP metrics match exact documented figures
    mlp_entry = next(m for m in data_all["models"] if m["model_id"] == "mlp_128_64_32")
    stm = mlp_entry["student_test"]
    assert round(stm["candidate_accuracy"], 4) == 0.9688
    assert round(stm["macro_f1"], 4) == 0.9186
    assert round(stm["obs_top1_accuracy"], 4) == 0.8641
    assert round(stm["log_loss"], 4) == 0.0779
    assert round(stm["candidate_miss_rate"], 4) == 0.0018

    # Live single-trip evaluation
    resp_trip = client.post(
        "/evaluate",
        json={"model_name": "mlp_128_64_32", "trip_id": "00000052", "partition": "student_test"},
    )
    assert resp_trip.status_code == 200
    dtrip = resp_trip.json()
    assert dtrip["scope"] == "single_trip"
    assert dtrip["metrics"]["total_observations"] == 262


def test_api_client_offline_error_and_env_url(monkeypatch):
    monkeypatch.setenv("PATHSNAP_API_URL", "http://192.168.1.100:8000/")
    assert get_default_api_url() == "http://192.168.1.100:8000"

    offline_client = PathSnapAPIClient(base_url="http://127.0.0.1:19999", timeout_s=2.0)
    with pytest.raises(APIClientError):
        offline_client.check_health()
