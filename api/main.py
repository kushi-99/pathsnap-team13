"""PathSnap FastAPI Backend Server (Laptop 1).

Launch on Laptop 1:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health            -> API readiness, loaded models, dataset availability
    GET  /models            -> Available models, architectures, and verified metrics
    GET  /trips             -> Available trip IDs for a partition
    GET  /trips/{trip_id}   -> Trip metadata and GPS trajectory coordinates
    POST /predict           -> Run model inference on a trip_id or raw candidate records
    POST /evaluate          -> Retrieve verified model comparison or run live trip evaluation
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from api import API_VERSION
from api.model_registry import ModelNotFoundError, ModelRegistry
from api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    TripDetailResponse,
    TripsListResponse,
)
from api.trip_service import TripNotFoundError, TripService
from ml.mlp.predict import PredictionInputError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pathsnap.api")

registry = ModelRegistry()
trip_service = TripService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML artifacts ONCE at startup (never per request)."""
    logger.info("Starting PathSnap FastAPI backend v%s ...", API_VERSION)
    registry.load_all()
    logger.info(
        "Loaded %d model(s) in %.3fs: %s (default=%s)",
        len(registry.available_models),
        registry.startup_time_s,
        registry.available_models,
        registry.default_model,
    )
    logger.info(
        "Dataset root: %s (available=%s)",
        trip_service.data_root,
        trip_service.data_root_available(),
    )
    yield
    logger.info("Shutting down PathSnap FastAPI backend.")


app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description=(
        "Laptop 1 inference and server-side evaluation backend for PathSnap "
        "(Multi-Model Candidate Scoring + Route Reconstruction)."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "%s %s -> HTTP %d (%.1f ms) from %s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request.client.host if request.client else "unknown",
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Return server readiness, model loading status, and dataset availability."""
    trips = trip_service.list_trips("student_test")
    return HealthResponse(
        status="ok" if registry.available_models else "degraded",
        api_version=API_VERSION,
        models_loaded=len(registry.available_models) > 0,
        available_models=registry.available_models,
        default_model=registry.default_model,
        model_name=registry.default_model,
        decoder="classifier_top1",
        data_root_available=trip_service.data_root_available(),
        available_trips_count=len(trips),
        startup_time_s=registry.startup_time_s,
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
def get_models() -> ModelsListResponse:
    """List all loaded models with architectures and verified evaluation summaries."""
    return ModelsListResponse(
        api_version=API_VERSION,
        default_model=registry.default_model,
        best_model_id=registry.comparison_summary.get("best_model_id", registry.default_model),
        feature_columns=registry.feature_columns,
        models=registry.list_model_infos(),
        teammate_branch_status=registry.comparison_summary.get("teammate_branch_status", []),
    )


@app.get("/trips", response_model=TripsListResponse, tags=["Trips"])
def get_trips(
    partition: str = Query(
        default="student_test",
        description="Dataset partition ('student_test', 'train', or 'evaluator_test').",
    )
) -> TripsListResponse:
    """Dynamically list available trip identifiers in the requested partition."""
    try:
        trips = trip_service.list_trips(partition=partition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TripsListResponse(partition=partition, count=len(trips), trips=trips)


@app.get("/trips/{trip_id}", response_model=TripDetailResponse, tags=["Trips"])
def get_trip_detail(
    trip_id: str,
    partition: str = Query(default="student_test"),
) -> TripDetailResponse:
    """Return metadata, geographical bounds, and raw GPS coordinates for one trip."""
    try:
        detail = trip_service.get_trip_detail(trip_id=trip_id, partition=partition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TripNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TripDetailResponse(**detail)


def _assemble_observation_predictions(
    obs_ids: np.ndarray,
    arc_ids: np.ndarray,
    match_probs: np.ndarray,
    emission_probs: np.ndarray,
    log_emission_probs: np.ndarray,
    include_full_observations: bool,
):
    """Aggregate candidate scores per observation preserving exact (obs_idx, arc_id) mapping."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for oid, aid, mp, ep, lp in zip(
        obs_ids.tolist(),
        arc_ids.tolist(),
        match_probs.tolist(),
        emission_probs.tolist(),
        log_emission_probs.tolist(),
    ):
        grouped.setdefault(int(oid), []).append(
            {
                "arc_id": int(aid),
                "match_prob": round(float(mp), 6),
                "emission_prob": round(float(ep), 6),
                "log_emission_prob": round(float(lp), 6),
            }
        )

    predicted_arc_ids: List[int] = []
    top1_mp: List[float] = []
    top1_ep: List[float] = []
    top1_lp: List[float] = []
    observations_out: List[Dict[str, Any]] = []

    for oid in sorted(grouped.keys()):
        cands = sorted(grouped[oid], key=lambda x: (-x["match_prob"], x["arc_id"]))
        best = cands[0]
        predicted_arc_ids.append(best["arc_id"])
        top1_mp.append(best["match_prob"])
        top1_ep.append(best["emission_prob"])
        top1_lp.append(best["log_emission_prob"])
        if include_full_observations:
            observations_out.append(
                {
                    "obs_idx": oid,
                    "predicted_arc_id": best["arc_id"],
                    "top1_match_prob": best["match_prob"],
                    "top1_emission_prob": best["emission_prob"],
                    "candidates": cands,
                }
            )

    return (
        predicted_arc_ids,
        top1_mp,
        top1_ep,
        top1_lp,
        observations_out if include_full_observations else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def post_predict(payload: PredictRequest) -> PredictResponse:
    """Run candidate scoring & map-matching for either raw candidates or a server-side trip_id."""
    try:
        predictor = registry.get_predictor(payload.model_name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    display_name = registry.get_display_name(payload.model_name)

    # Mode A: Direct candidate-level records (e.g., HMM pipeline or API integration test)
    if payload.candidates is not None:
        raw_records = [c.model_dump() for c in payload.candidates]
        try:
            df, mp, ep, lp, infer_ms = predictor.predict_scores(raw_records)
        except PredictionInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        pred_arcs, top1_mp, top1_ep, top1_lp, obs_out = _assemble_observation_predictions(
            df["obs_idx"].to_numpy(dtype=np.int64),
            df["arc_id"].to_numpy(dtype=np.int64),
            mp,
            ep,
            lp,
            include_full_observations=True,
        )
        return PredictResponse(
            trip_id=payload.trip_id,
            partition=None,
            model_name=payload.model_name,
            model_display_name=display_name,
            num_observations=len(pred_arcs),
            num_candidates=int(len(df)),
            num_predicted_segments=len(set(pred_arcs)),
            candidate_extraction_ms=0.0,
            inference_time_ms=round(infer_ms, 3),
            predicted_arc_ids=pred_arcs,
            top1_match_probs=top1_mp,
            top1_emission_probs=top1_ep,
            top1_log_emission_probs=top1_lp,
            observations=obs_out,
        )

    # Mode B: Trip-level end-to-end prediction & server-side evaluation on Laptop 1
    try:
        trip, cand_df, cand_summary, extract_ms = trip_service.get_trip_candidates(
            trip_id=str(payload.trip_id),
            partition=payload.partition,
            load_labels=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TripNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        _, mp, ep, lp, infer_ms = predictor.predict_scores(cand_df)
    except PredictionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pred_arcs, top1_mp, top1_ep, top1_lp, obs_out = _assemble_observation_predictions(
        cand_df["obs_idx"].to_numpy(dtype=np.int64),
        cand_df["arc_id"].to_numpy(dtype=np.int64),
        mp,
        ep,
        lp,
        include_full_observations=payload.include_observations,
    )

    # Reconstruct geographical coordinates [lat, lon] from actual road network
    gps_points = [
        [round(float(r[1]), 6), round(float(r[0]), 6), float(r[2])] for r in trip.track
    ]
    predicted_segments = trip_service.arcs_to_latlon_segments(
        trip, pred_arcs, deduplicate_consecutive=True
    )

    candidate_segments: List[List[List[float]]] = []
    selected_candidate_segments: List[List[List[float]]] = []
    if payload.include_candidate_segments:
        unique_cand_arcs = cand_df["arc_id"].drop_duplicates().astype(int).tolist()
        # Cap background candidate segments slightly on massive trips to keep HTTP payload snappy
        if len(unique_cand_arcs) > 3500:
            step = max(1, len(unique_cand_arcs) // 3500)
            unique_cand_arcs = unique_cand_arcs[::step]
        candidate_segments = trip_service.arcs_to_latlon_segments(
            trip, unique_cand_arcs, deduplicate_consecutive=False
        )
        unique_selected_arcs = list(dict.fromkeys(pred_arcs))
        selected_candidate_segments = trip_service.arcs_to_latlon_segments(
            trip, unique_selected_arcs, deduplicate_consecutive=False
        )

    # Server-side evaluation (ground-truth route NEVER leaked for evaluator_test!)
    metrics_out, continuity_out = trip_service.compute_trip_evaluation(
        trip, cand_df, cand_summary, mp, partition=payload.partition
    )

    allow_ref_geometry = (
        payload.include_reference
        and payload.partition != "evaluator_test"
        and trip.route is not None
    )
    reference_segments = (
        trip_service.arcs_to_latlon_segments(
            trip, trip.route.tolist(), deduplicate_consecutive=False
        )
        if allow_ref_geometry
        else None
    )

    return PredictResponse(
        trip_id=trip.trip_id,
        partition=payload.partition,
        model_name=payload.model_name,
        model_display_name=display_name,
        num_observations=len(pred_arcs),
        num_candidates=int(len(cand_df)),
        num_predicted_segments=len(predicted_segments),
        candidate_extraction_ms=extract_ms,
        inference_time_ms=round(infer_ms, 3),
        predicted_arc_ids=pred_arcs,
        top1_match_probs=top1_mp,
        top1_emission_probs=top1_ep,
        top1_log_emission_probs=top1_lp,
        observations=obs_out,
        gps_points=gps_points,
        predicted_segments=predicted_segments,
        candidate_segments=candidate_segments,
        selected_candidate_segments=selected_candidate_segments,
        reference_segments=reference_segments,
        reference_available=bool(trip.route is not None and payload.partition != "evaluator_test"),
        metrics=metrics_out,
        route_continuity=continuity_out,
    )


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def post_evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    """Return verified dataset-level evaluation metrics or run live evaluation for one trip."""
    # Case 1: Live single-trip evaluation
    if payload.trip_id is not None:
        model_id = payload.model_name or registry.default_model
        try:
            predictor = registry.get_predictor(model_id)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            trip, cand_df, cand_summary, _ = trip_service.get_trip_candidates(
                trip_id=payload.trip_id, partition=payload.partition, load_labels=True
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TripNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        _, mp, _, _, infer_ms = predictor.predict_scores(cand_df)
        metrics_out, continuity_out = trip_service.compute_trip_evaluation(
            trip, cand_df, cand_summary, mp, partition=payload.partition
        )
        if metrics_out is not None:
            metrics_out["inference_time_ms"] = round(infer_ms, 3)

        return EvaluateResponse(
            scope="single_trip",
            model_name=model_id,
            trip_id=payload.trip_id,
            partition=payload.partition,
            metrics=metrics_out,
            route_continuity=continuity_out,
        )

    # Case 2: Dataset-level verified comparison / single-model evaluation summary
    comp = registry.comparison_summary
    all_models = comp.get("models", [])
    per_trip_records: List[Dict[str, Any]] = []
    if registry.per_trip_df is not None:
        df_pt = registry.per_trip_df
        if payload.model_name:
            df_pt = df_pt[df_pt["model_id"] == payload.model_name]
        per_trip_records = df_pt.to_dict(orient="records")

    if payload.model_name:
        if payload.model_name not in registry.model_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"No evaluation summary found for model '{payload.model_name}'.",
            )
        m_meta = registry.model_metadata[payload.model_name]
        return EvaluateResponse(
            scope="dataset_single_model",
            model_name=payload.model_name,
            partition=payload.partition,
            dataset_summary=comp.get("dataset_summary"),
            best_model_id=comp.get("best_model_id"),
            models=[m_meta],
            metrics=m_meta.get("student_test"),
            per_trip=per_trip_records,
            roc_pr_curves={
                payload.model_name: registry.roc_pr_curves.get(payload.model_name)
            }
            if payload.model_name in registry.roc_pr_curves
            else None,
            teammate_branch_status=comp.get("teammate_branch_status"),
        )

    return EvaluateResponse(
        scope="dataset_all_models",
        partition=payload.partition,
        dataset_summary=comp.get("dataset_summary"),
        best_model_id=comp.get("best_model_id"),
        models=all_models,
        per_trip=per_trip_records,
        roc_pr_curves=registry.roc_pr_curves,
        teammate_branch_status=comp.get("teammate_branch_status"),
    )
