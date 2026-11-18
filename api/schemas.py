"""Pydantic v2 request and response schemas for the PathSnap FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    api_version: str
    models_loaded: bool
    available_models: List[str]
    default_model: str
    model_name: str  # Backwards-compatible alias for earlier UI clients
    decoder: str = "classifier_top1"
    data_root_available: bool
    available_trips_count: int
    startup_time_s: float


class ModelInfo(BaseModel):
    model_id: str
    display_name: str
    family: str
    architecture: str
    num_features: int
    loaded: bool
    artifact_path: str
    is_best: bool = False
    student_test_metrics: Optional[Dict[str, Any]] = None
    cv_metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class ModelsListResponse(BaseModel):
    api_version: str
    default_model: str
    best_model_id: Optional[str] = None
    feature_columns: List[str]
    models: List[ModelInfo]
    teammate_branch_status: List[Dict[str, Any]] = Field(default_factory=list)


class TripSummary(BaseModel):
    trip_id: str
    partition: str
    num_observations: int
    num_nodes: int
    num_arcs: int
    has_reference_route: bool


class TripsListResponse(BaseModel):
    partition: str
    count: int
    trips: List[TripSummary]


class TripDetailResponse(BaseModel):
    trip_id: str
    partition: str
    num_observations: int
    num_nodes: int
    num_arcs: int
    has_reference_route: bool
    duration_s: float
    bounds: Dict[str, float]
    gps_points: List[List[float]] = Field(
        default_factory=list,
        description="List of [lat, lon, timestamp] GPS observations.",
    )


class CandidateRecordInput(BaseModel):
    obs_idx: int = Field(..., ge=0)
    arc_id: int = Field(..., ge=0)
    features: Dict[str, float]


class PredictRequest(BaseModel):
    trip_id: Optional[str] = Field(
        default=None,
        description="Trip identifier on Laptop 1 (e.g., '00000000'). Mutually exclusive with `candidates`.",
    )
    partition: str = Field(
        default="student_test",
        description="Dataset partition on server ('student_test', 'train', or 'evaluator_test').",
    )
    model_name: str = Field(
        default="mlp_128_64_32",
        description="Registered model ID from GET /models.",
    )
    candidates: Optional[List[CandidateRecordInput]] = Field(
        default=None,
        description="Optional direct candidate records for raw model scoring.",
    )
    include_reference: bool = Field(
        default=False,
        description="Include reference route geometry for development partitions (blocked for evaluator_test).",
    )
    include_candidate_segments: bool = Field(
        default=True,
        description="Include deduplicated candidate road-segment coordinates for Folium overlay.",
    )
    include_observations: bool = Field(
        default=False,
        description="Include full per-observation candidate score lists (auto-enabled in raw candidates mode).",
    )

    @model_validator(mode="after")
    def _validate_target(self) -> "PredictRequest":
        if not self.trip_id and not self.candidates:
            raise ValueError("Provide either 'trip_id' or 'candidates'.")
        if self.candidates is not None and len(self.candidates) == 0:
            raise ValueError("'candidates' list cannot be empty.")
        return self


class CandidateScoreOut(BaseModel):
    arc_id: int
    match_prob: float
    emission_prob: float
    log_emission_prob: float


class ObservationPredictionOut(BaseModel):
    obs_idx: int
    predicted_arc_id: int
    top1_match_prob: float
    top1_emission_prob: float
    candidates: List[CandidateScoreOut]


class PredictResponse(BaseModel):
    trip_id: Optional[str] = None
    partition: Optional[str] = None
    model_name: str
    model_display_name: str
    decoder: str = "classifier_top1"
    num_observations: int
    num_candidates: int
    num_predicted_segments: int
    candidate_extraction_ms: float = 0.0
    inference_time_ms: float
    predicted_arc_ids: List[int]
    top1_match_probs: List[float] = Field(default_factory=list)
    top1_emission_probs: List[float] = Field(default_factory=list)
    top1_log_emission_probs: List[float] = Field(default_factory=list)
    observations: Optional[List[ObservationPredictionOut]] = None
    gps_points: List[List[float]] = Field(default_factory=list)
    predicted_segments: List[List[List[float]]] = Field(default_factory=list)
    candidate_segments: List[List[List[float]]] = Field(default_factory=list)
    selected_candidate_segments: List[List[List[float]]] = Field(default_factory=list)
    reference_segments: Optional[List[List[List[float]]]] = None
    reference_available: bool = False
    metrics: Optional[Dict[str, Any]] = None
    route_continuity: Optional[Dict[str, Any]] = None


class EvaluateRequest(BaseModel):
    model_name: Optional[str] = Field(
        default=None,
        description="Model ID to evaluate/fetch metrics for. Omit for full multi-model comparison.",
    )
    trip_id: Optional[str] = Field(
        default=None,
        description="Optional trip ID to evaluate live on server.",
    )
    partition: str = Field(
        default="student_test",
        description="Partition to evaluate ('student_test' or 'train').",
    )


class EvaluateResponse(BaseModel):
    scope: str
    model_name: Optional[str] = None
    trip_id: Optional[str] = None
    partition: str
    dataset_summary: Optional[Dict[str, Any]] = None
    best_model_id: Optional[str] = None
    models: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    route_continuity: Optional[Dict[str, Any]] = None
    per_trip: Optional[List[Dict[str, Any]]] = None
    roc_pr_curves: Optional[Dict[str, Any]] = None
    teammate_branch_status: Optional[List[Dict[str, Any]]] = None
