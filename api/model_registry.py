"""Startup singleton model registry for PathSnap FastAPI inference.

Loads all trained model artifacts ONCE at server startup and exposes a uniform
candidate-scoring interface across PyTorch MLP and scikit-learn classifiers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from ml.mlp.config import (
    ARTIFACTS_DIR,
    BASELINES_ARTIFACTS_DIR,
    COMPARISON_DIR,
    DEFAULT_FEATURE_COLUMNS,
    RESULTS_DIR,
)
from ml.mlp.model import normalize_probs_per_observation
from ml.mlp.predict import MLPPredictor, PredictionInputError


class ModelNotFoundError(KeyError):
    """Raised when a requested model_id is not loaded in the registry."""


class SklearnCandidatePredictor:
    """Uniform wrapper around scikit-learn binary candidate classifiers (.joblib)."""

    def __init__(
        self,
        model_id: str,
        display_name: str,
        estimator: Any,
        feature_columns: List[str],
    ) -> None:
        self.model_id = model_id
        self.model_name = model_id
        self.display_name = display_name
        self.estimator = estimator
        self.feature_columns = list(feature_columns)

    def _prepare_dataframe(
        self, candidates: Union[pd.DataFrame, Sequence[Mapping[str, Any]]]
    ) -> pd.DataFrame:
        if isinstance(candidates, pd.DataFrame):
            df = candidates
        elif isinstance(candidates, Sequence):
            if len(candidates) == 0:
                raise PredictionInputError("Candidate list is empty.")
            rows = []
            for rec in candidates:
                if "features" in rec and isinstance(rec["features"], Mapping):
                    flat = {"obs_idx": rec["obs_idx"], "arc_id": rec["arc_id"]}
                    if "trip_id" in rec:
                        flat["trip_id"] = rec["trip_id"]
                    flat.update(rec["features"])
                    rows.append(flat)
                else:
                    rows.append(dict(rec))
            df = pd.DataFrame(rows)
        else:
            raise PredictionInputError(f"Unsupported candidates input type: {type(candidates).__name__}")

        if len(df) == 0:
            raise PredictionInputError("Candidate DataFrame is empty.")
        for col in ("obs_idx", "arc_id"):
            if col not in df.columns:
                raise PredictionInputError(f"Missing required identifier column: '{col}'")
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise PredictionInputError(f"Missing required feature columns ({len(missing)}): {missing[:8]}")
        return df

    def predict_scores(
        self, candidates: Union[pd.DataFrame, Sequence[Mapping[str, Any]]]
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        df = self._prepare_dataframe(candidates)
        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(X).all():
            bad = [
                self.feature_columns[i]
                for i in range(X.shape[1])
                if not np.isfinite(X[:, i]).all()
            ]
            raise PredictionInputError(f"Non-finite (NaN/Inf) values in features: {bad}")

        probs = self.estimator.predict_proba(X)[:, 1].astype(np.float64)
        obs_ids = df["obs_idx"].to_numpy(dtype=np.int64)
        emission_probs, log_emission_probs = normalize_probs_per_observation(probs, obs_ids)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return df, probs, emission_probs, log_emission_probs, elapsed_ms


class ModelRegistry:
    """Manages loaded ML predictors and verified evaluation metadata."""

    def __init__(self) -> None:
        self.predictors: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.comparison_summary: Dict[str, Any] = {}
        self.roc_pr_curves: Dict[str, Any] = {}
        self.per_trip_df: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = list(DEFAULT_FEATURE_COLUMNS)
        self.default_model: str = "mlp_128_64_32"
        self.startup_time_s: float = 0.0

    def load_all(self) -> None:
        """Load MLP artifact, available scikit-learn baselines, and verified results once."""
        t0 = time.perf_counter()
        self.predictors.clear()
        self.model_metadata.clear()

        # 1. Load comparison summary if generated
        comp_json = COMPARISON_DIR / "model_comparison_summary.json"
        if comp_json.exists():
            self.comparison_summary = json.loads(comp_json.read_text(encoding="utf-8"))
            for m in self.comparison_summary.get("models", []):
                self.model_metadata[m["model_id"]] = m
            self.feature_columns = list(
                self.comparison_summary.get("feature_schema", {}).get(
                    "feature_columns", DEFAULT_FEATURE_COLUMNS
                )
            )

        curves_json = COMPARISON_DIR / "roc_pr_curves.json"
        if curves_json.exists():
            self.roc_pr_curves = json.loads(curves_json.read_text(encoding="utf-8"))

        per_trip_csv = COMPARISON_DIR / "per_trip_all_models.csv"
        if per_trip_csv.exists():
            self.per_trip_df = pd.read_csv(per_trip_csv, dtype={"trip_id": str})
        elif (RESULTS_DIR / "student_test_per_trip.csv").exists():
            df_pt = pd.read_csv(RESULTS_DIR / "student_test_per_trip.csv", dtype={"trip_id": str})
            df_pt.insert(0, "model_id", "mlp_128_64_32")
            self.per_trip_df = df_pt

        # 2. Load finalized PyTorch MLP artifact
        if (ARTIFACTS_DIR / "mlp_model.pt").exists():
            mlp_pred = MLPPredictor.from_artifacts(ARTIFACTS_DIR, device="cpu")
            self.predictors["mlp_128_64_32"] = mlp_pred
            self.feature_columns = list(mlp_pred.feature_columns)
            if "mlp_128_64_32" not in self.model_metadata:
                st_path = RESULTS_DIR / "student_test_metrics.json"
                st_data = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {}
                self.model_metadata["mlp_128_64_32"] = {
                    "model_id": "mlp_128_64_32",
                    "display_name": "MLP (128-64-32)",
                    "family": "Neural Network (PyTorch)",
                    "architecture": "28 -> 128 -> 64 -> 32 -> 1 (BatchNorm1d + ReLU + Dropout 0.20, 14,529 params)",
                    "artifact_path": "ml/mlp/artifacts/mlp_model.pt",
                    "num_features": len(self.feature_columns),
                    "config": mlp_pred.model_config.get("mlp_config"),
                    "student_test": st_data.get("metrics"),
                }

        # 3. Load trained scikit-learn baseline artifacts from artifacts/baselines/
        baseline_display = {
            "hist_gradient_boosting": ("HistGradientBoosting", "Gradient Boosting"),
            "random_forest": ("Random Forest", "Bagging / Ensemble"),
            "logistic_regression": ("Logistic Regression", "Linear / Probabilistic"),
        }
        if BASELINES_ARTIFACTS_DIR.exists():
            for joblib_file in sorted(BASELINES_ARTIFACTS_DIR.glob("*.joblib")):
                mid = joblib_file.stem
                est = joblib.load(joblib_file)
                disp, fam = baseline_display.get(mid, (mid.replace("_", " ").title(), "Scikit-Learn"))
                if mid in self.model_metadata:
                    disp = self.model_metadata[mid].get("display_name", disp)
                self.predictors[mid] = SklearnCandidatePredictor(
                    model_id=mid,
                    display_name=disp,
                    estimator=est,
                    feature_columns=self.feature_columns,
                )
                if mid not in self.model_metadata:
                    self.model_metadata[mid] = {
                        "model_id": mid,
                        "display_name": disp,
                        "family": fam,
                        "architecture": str(est),
                        "artifact_path": f"artifacts/baselines/{joblib_file.name}",
                        "num_features": len(self.feature_columns),
                    }

        # Pick default model (prefer best comparable model if loaded, else mlp_128_64_32)
        best_id = self.comparison_summary.get("best_model_id")
        if best_id and best_id in self.predictors:
            self.default_model = best_id
        elif "mlp_128_64_32" in self.predictors:
            self.default_model = "mlp_128_64_32"
        elif self.predictors:
            self.default_model = next(iter(self.predictors))

        self.startup_time_s = round(time.perf_counter() - t0, 3)

    @property
    def available_models(self) -> List[str]:
        return list(self.predictors.keys())

    def get_predictor(self, model_id: str) -> Any:
        if model_id not in self.predictors:
            raise ModelNotFoundError(
                f"Model '{model_id}' is not loaded. Available models: {self.available_models}"
            )
        return self.predictors[model_id]

    def get_display_name(self, model_id: str) -> str:
        meta = self.model_metadata.get(model_id, {})
        return str(meta.get("display_name", model_id))

    def list_model_infos(self) -> List[Dict[str, Any]]:
        best_id = self.comparison_summary.get("best_model_id", self.default_model)
        ordered_ids = [
            m["model_id"]
            for m in self.comparison_summary.get("models", [])
            if m["model_id"] in self.predictors
        ]
        for mid in self.predictors:
            if mid not in ordered_ids:
                ordered_ids.append(mid)

        out: List[Dict[str, Any]] = []
        for mid in ordered_ids:
            meta = self.model_metadata.get(mid, {})
            out.append(
                {
                    "model_id": mid,
                    "display_name": meta.get("display_name", mid),
                    "family": meta.get("family", "Unknown"),
                    "architecture": meta.get("architecture", "N/A"),
                    "num_features": int(meta.get("num_features", len(self.feature_columns))),
                    "loaded": mid in self.predictors,
                    "artifact_path": meta.get("artifact_path", ""),
                    "is_best": mid == best_id,
                    "student_test_metrics": meta.get("student_test"),
                    "cv_metrics": meta.get("cv"),
                    "config": meta.get("config"),
                }
            )
        return out
