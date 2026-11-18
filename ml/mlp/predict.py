"""Clean, framework-agnostic MLP inference interface for FastAPI and HMM/Viterbi.

Usage (Backend / FastAPI integration):
    from ml.mlp.predict import MLPPredictor

    predictor = MLPPredictor.from_artifacts()
    result = predictor.predict_candidates(candidate_records)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
import torch

from ml.mlp.config import ARTIFACTS_DIR, CandidateConfig
from ml.mlp.data_adapter import extract_trip_candidates, load_raw_trip
from ml.mlp.model import PathSnapMLP, normalize_probs_per_observation


class PredictionInputError(ValueError):
    """Raised when candidate inference payloads fail schema or value validation."""


class MLPPredictor:
    """Thread-safe inference wrapper around the trained PathSnap MLP artifact."""

    def __init__(
        self,
        model: PathSnapMLP,
        scaler: Any,
        feature_columns: List[str],
        label_mapping: Dict[str, Any],
        model_config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.feature_columns = list(feature_columns)
        self.label_mapping = dict(label_mapping)
        self.model_config = dict(model_config)
        self.device = device
        self.model_name = str(
            model_config.get("mlp_config", {}).get("experiment_name", "pathsnap_mlp")
        )

    @classmethod
    def from_artifacts(
        cls, artifacts_dir: Path = ARTIFACTS_DIR, device: str = "cpu"
    ) -> "MLPPredictor":
        """Load trained weights, scaler, feature schema, and metadata from disk."""
        artifacts_dir = Path(artifacts_dir)
        required = [
            "mlp_model.pt",
            "scaler.joblib",
            "feature_columns.json",
            "label_mapping.json",
            "model_config.json",
        ]
        missing = [f for f in required if not (artifacts_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required MLP artifacts in {artifacts_dir}: {missing}"
            )

        ckpt = torch.load(artifacts_dir / "mlp_model.pt", map_location=device, weights_only=True)
        model = PathSnapMLP(
            input_dim=int(ckpt["input_dim"]),
            hidden_dims=tuple(ckpt["hidden_dims"]),
            dropout=float(ckpt["dropout"]),
            use_batch_norm=bool(ckpt["use_batch_norm"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        scaler = joblib.load(artifacts_dir / "scaler.joblib")
        feat_data = json.loads((artifacts_dir / "feature_columns.json").read_text(encoding="utf-8"))
        feature_columns = feat_data["feature_columns"]
        label_mapping = json.loads((artifacts_dir / "label_mapping.json").read_text(encoding="utf-8"))
        model_config = json.loads((artifacts_dir / "model_config.json").read_text(encoding="utf-8"))

        return cls(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            label_mapping=label_mapping,
            model_config=model_config,
            device=device,
        )

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
                    flat = {
                        "obs_idx": rec["obs_idx"],
                        "arc_id": rec["arc_id"],
                    }
                    if "trip_id" in rec:
                        flat["trip_id"] = rec["trip_id"]
                    flat.update(rec["features"])
                    rows.append(flat)
                else:
                    rows.append(dict(rec))
            df = pd.DataFrame(rows)
        else:
            raise PredictionInputError(
                f"Unsupported candidates input type: {type(candidates).__name__}"
            )

        if len(df) == 0:
            raise PredictionInputError("Candidate DataFrame is empty.")

        for id_col in ("obs_idx", "arc_id"):
            if id_col not in df.columns:
                raise PredictionInputError(f"Missing required identifier column: '{id_col}'")

        missing_feats = [c for c in self.feature_columns if c not in df.columns]
        if missing_feats:
            raise PredictionInputError(
                f"Missing required feature columns ({len(missing_feats)}): {missing_feats[:8]}"
            )

        return df

    def predict_scores(
        self, candidates: Union[pd.DataFrame, Sequence[Mapping[str, Any]]]
    ) -> Tuple_Scores:
        """Low-level vectorized scoring returning (df, match_probs, emission_probs, log_emission_probs, elapsed_ms)."""
        t0 = time.perf_counter()
        df = self._prepare_dataframe(candidates)
        X_raw = df[self.feature_columns].to_numpy(dtype=np.float32)

        if not np.isfinite(X_raw).all():
            bad_cols = [
                self.feature_columns[i]
                for i in range(X_raw.shape[1])
                if not np.isfinite(X_raw[:, i]).all()
            ]
            raise PredictionInputError(f"Non-finite (NaN/Inf) values in features: {bad_cols}")

        X_scaled = self.scaler.transform(X_raw).astype(np.float32)
        match_probs = self.model.predict_proba_numpy(X_scaled, device=self.device)
        obs_ids = df["obs_idx"].to_numpy(dtype=np.int64)
        emission_probs, log_emission_probs = normalize_probs_per_observation(match_probs, obs_ids)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return df, match_probs, emission_probs, log_emission_probs, elapsed_ms

    def predict_candidates(
        self,
        candidates: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
        trip_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict candidate probabilities and Top-1 road segments preserving (obs_idx, arc_id) mapping.

        Output is directly consumable by both FastAPI endpoints and downstream HMM/Viterbi decoders.
        """
        df, match_probs, emission_probs, log_emission_probs, elapsed_ms = self.predict_scores(
            candidates
        )

        if trip_id is None and "trip_id" in df.columns:
            trip_id = str(df["trip_id"].iloc[0])

        obs_ids = df["obs_idx"].to_numpy(dtype=np.int64)
        arc_ids = df["arc_id"].to_numpy(dtype=np.int64)

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for oid, aid, mp, ep, lp in zip(
            obs_ids.tolist(),
            arc_ids.tolist(),
            match_probs.tolist(),
            emission_probs.tolist(),
            log_emission_probs.tolist(),
        ):
            grouped.setdefault(oid, []).append(
                {
                    "arc_id": int(aid),
                    "match_prob": round(float(mp), 6),
                    "emission_prob": round(float(ep), 6),
                    "log_emission_prob": round(float(lp), 6),
                }
            )

        observations_out: List[Dict[str, Any]] = []
        predicted_arc_sequence: List[int] = []

        for oid in sorted(grouped.keys()):
            cands = sorted(grouped[oid], key=lambda item: item["match_prob"], reverse=True)
            best = cands[0]
            predicted_arc_sequence.append(best["arc_id"])
            observations_out.append(
                {
                    "obs_idx": int(oid),
                    "predicted_arc_id": int(best["arc_id"]),
                    "top1_match_prob": best["match_prob"],
                    "top1_emission_prob": best["emission_prob"],
                    "candidates": cands,
                }
            )

        return {
            "trip_id": trip_id,
            "model_name": self.model_name,
            "num_observations": len(observations_out),
            "num_candidates": int(len(df)),
            "inference_time_ms": round(elapsed_ms, 3),
            "predicted_arc_ids": predicted_arc_sequence,
            "observations": observations_out,
        }

    def predict_trip_dir(self, trip_dir: Path) -> Dict[str, Any]:
        """Run leak-free candidate extraction + MLP scoring on a raw trip directory (ignores .route)."""
        cand_cfg_dict = self.model_config.get("candidate_config", {})
        cand_cfg = CandidateConfig(**cand_cfg_dict) if cand_cfg_dict else CandidateConfig()
        raw_trip = load_raw_trip(Path(trip_dir), load_route=False)
        cand_df, _ = extract_trip_candidates(raw_trip, cfg=cand_cfg, load_labels=False)
        return self.predict_candidates(cand_df, trip_id=raw_trip.trip_id)


Tuple_Scores = tuple
