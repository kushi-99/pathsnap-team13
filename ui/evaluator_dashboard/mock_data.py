"""Strictly isolated layout-testing mock data provider (OFF by default).

WARNING:
- Used ONLY to inspect UI layout before Laptop 1 FastAPI is online.
- Completely separated from `api_client.py` and the real inference pipeline.
- Does NOT read any evaluator dataset, does NOT contain real evaluation predictions,
  and marks every response with `"is_mock": True` so the UI displays a prominent warning banner.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def get_layout_preview_trips() -> List[Dict[str, Any]]:
    """Return 10 clearly labelled placeholder trip options for UI layout preview only."""
    return [
        {"trip_id": f"PREVIEW_TRIP_{i:02d}", "num_observations": 120 + i * 15}
        for i in range(1, 11)
    ]


def get_layout_preview_prediction(trip_id: str) -> Dict[str, Any]:
    """Generate synthetic sinusoidal coordinates purely for layout rendering checks."""
    base_lat, base_lon = 12.9345, 77.5345
    n_pts = 40
    gps_points = []
    route_nodes = []

    for i in range(n_pts):
        t = i / (n_pts - 1)
        r_lat = base_lat + 0.008 * t
        r_lon = base_lon + 0.010 * t + 0.0012 * math.sin(t * math.pi * 2.0)
        route_nodes.append([r_lat, r_lon])
        # Slight synthetic jitter to visually demonstrate raw GPS vs snapped road route
        jitter_lat = 0.00018 * math.sin(i * 1.7)
        jitter_lon = 0.00018 * math.cos(i * 1.3)
        gps_points.append([r_lat + jitter_lat, r_lon + jitter_lon])

    predicted_segments = [
        [route_nodes[i], route_nodes[i + 1]] for i in range(len(route_nodes) - 1)
    ]

    return {
        "is_mock": True,
        "trip_id": trip_id,
        "model_name": "LAYOUT_PREVIEW_MOCK (Not Real Inference)",
        "decoder": "MOCK_PREVIEW",
        "num_observations": n_pts,
        "num_predicted_segments": len(predicted_segments),
        "inference_time_ms": 0.0,
        "gps_points": gps_points,
        "predicted_segments": predicted_segments,
        "predicted_arc_ids": list(range(len(predicted_segments))),
        "metrics": {
            "obs_top1_accuracy": None,
            "macro_f1": None,
            "log_loss": None,
            "candidate_miss_rate": None,
        },
        "route_continuity": {
            "obs_step_continuity": None,
            "compressed_route_continuity": None,
            "disconnected_transitions": None,
        },
        "summary_note": (
            "Synthetic layout-preview response only. Connect Laptop 1 FastAPI to run real inference "
            "and obtain actual server-side evaluation metrics."
        ),
    }
