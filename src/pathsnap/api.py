"""FastAPI service for PathSnap trip inspection and baseline scoring."""
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .candidates import generate_candidates, road_segments
from .data_loader import list_trip_ids, load_trip
from .features import FEATURE_COLUMNS, build_features
from .geo import project_trip
from .evaluate import evaluate_predictions

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="PathSnap API", version="1.0.0", description="GPS candidate-road scoring with Logistic Regression and KNN baselines.")
cors_origins = [
    origin.strip()
    for origin in os.getenv("PATHSNAP_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
MODELS = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/trips")
def trips(split: str = Query("evaluator_test", pattern="^(evaluator_test|train|student_test)$")):
    try:
        trip_ids = list_trip_ids(ROOT / "data", split)
        if not trip_ids:
            raise HTTPException(
                status_code=503, detail=f"No trips found in the {split} dataset."
            )
        return {"split": split, "trips": trip_ids}
    except HTTPException:
        raise
    except (FileNotFoundError, OSError) as error:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {error}") from error

@app.get("/score")
def score(trip_id: str = Query(pattern=r"^\d{8}$"), split: str = Query("evaluator_test", pattern="^(evaluator_test|train|student_test)$"), model_name: str = Query("logistic_regression", alias="model", pattern="^(logistic_regression|knn)$")):
    try:
        trip = load_trip(ROOT / "data", trip_id, split)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=404, detail=f"Trip unavailable: {error}") from error
    track, nodes, arcs = (trip[key] for key in ("track", "nodes", "arcs"))
    candidates = generate_candidates(track, nodes, arcs, trip_id=trip_id)
    features = build_features(track, nodes, arcs, candidates)
    if model_name not in MODELS:
        model_path = ROOT / "results" / "baseline_models" / f"{model_name}.joblib"
        if not model_path.is_file():
            raise HTTPException(status_code=503, detail=f"Saved model missing: {model_path.name}")
        MODELS[model_name] = joblib.load(model_path)
    features["score"] = MODELS[model_name].predict_proba(features[FEATURE_COLUMNS])[:, 1]
    evaluation = None
    if split == "evaluator_test":
        evaluation = evaluate_predictions(
            trip, features, features["score"].to_numpy(), model_name
        )
    ranked = features.sort_values(
        ["observation_index", "score"], ascending=[True, False]
    )
    best = ranked.groupby("observation_index", sort=True).head(1)
    gps_xy, transformer = project_trip(track.longitude, track.latitude)
    arc_ids, segments = road_segments(nodes, arcs, transformer)
    positions = {int(arc_id): index for index, arc_id in enumerate(arc_ids)}
    road_segments_ll = []
    for arc_id in best.candidate_arc_id.astype(int):
        segment = segments[positions[arc_id]]
        longitude, latitude = transformer.transform(segment[:, 0], segment[:, 1], direction="INVERSE")
        road_segments_ll.append([[float(lat), float(lon)] for lon, lat in zip(longitude, latitude)])
    best_by_observation = best.set_index("observation_index")
    observations = [
        {
            "index": int(index),
            "lat": float(row.latitude),
            "lon": float(row.longitude),
            "timestamp_s": float(row.timestamp_s),
            "candidate_arc_id": int(best_by_observation.loc[index, "candidate_arc_id"]),
            "score": float(best_by_observation.loc[index, "score"]),
            "distance_to_road_m": float(
                best_by_observation.loc[index, "distance_to_road_m"]
            ),
        }
        for index, row in enumerate(track.itertuples())
    ]
    return {"trip_id": trip_id, "split": split, "model": model_name, "observations": observations, "gps": [[float(row.latitude), float(row.longitude)] for row in track.itertuples()], "selected_roads": road_segments_ll, "candidate_count": int(len(features)), "features": FEATURE_COLUMNS, "evaluation": evaluation}
