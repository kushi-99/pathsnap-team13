"""Trip discovery, candidate caching, and road-network geometry reconstruction for FastAPI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.mlp.config import CandidateConfig, DATA_ROOT
from ml.mlp.data_adapter import RawTrip, extract_trip_candidates, load_raw_trip
from ml.mlp.metrics_utils import (
    compute_candidate_and_obs_metrics,
    compute_classifier_route_continuity,
)


class TripNotFoundError(FileNotFoundError):
    """Raised when a requested trip_id does not exist in the specified partition."""


class TripService:
    """Manages trip listing, geometry extraction, and server-side evaluation."""

    def __init__(self, data_root: Path = DATA_ROOT) -> None:
        self.data_root = Path(data_root)
        self.cand_cfg = CandidateConfig()
        self._trip_cache: Dict[Tuple[str, str], RawTrip] = {}
        self._cand_cache: Dict[Tuple[str, str, bool], Tuple[pd.DataFrame, Dict[str, Any]]] = {}

    def get_partition_dir(self, partition: str) -> Path:
        allowed = {"student_test", "train", "evaluator_test"}
        if partition not in allowed:
            raise ValueError(f"Unsupported partition '{partition}'. Allowed: {sorted(allowed)}")
        return self.data_root / partition

    def data_root_available(self) -> bool:
        return (self.data_root / "student_test").is_dir() or (self.data_root / "train").is_dir()

    def list_trips(self, partition: str = "student_test") -> List[Dict[str, Any]]:
        pdir = self.get_partition_dir(partition)
        if not pdir.is_dir():
            return []

        summaries: List[Dict[str, Any]] = []
        for tdir in sorted(d for d in pdir.iterdir() if d.is_dir()):
            tid = tdir.name
            track_f = tdir / f"{tid}.track"
            nodes_f = tdir / f"{tid}.nodes"
            arcs_f = tdir / f"{tid}.arcs"
            if not (track_f.is_file() and nodes_f.is_file() and arcs_f.is_file()):
                continue
            route_f = tdir / f"{tid}.route"
            # Fast line counts for metadata listing
            n_obs = sum(1 for _ in track_f.open("rb"))
            n_nodes = sum(1 for _ in nodes_f.open("rb"))
            n_arcs = sum(1 for _ in arcs_f.open("rb"))
            # Hide route existence on evaluator_test to preserve strict evaluation integrity
            has_ref = route_f.is_file() and (partition != "evaluator_test")
            summaries.append(
                {
                    "trip_id": tid,
                    "partition": partition,
                    "num_observations": int(n_obs),
                    "num_nodes": int(n_nodes),
                    "num_arcs": int(n_arcs),
                    "has_reference_route": bool(has_ref),
                }
            )
        return summaries

    def load_trip(self, trip_id: str, partition: str = "student_test") -> RawTrip:
        key = (partition, str(trip_id))
        if key in self._trip_cache:
            return self._trip_cache[key]

        pdir = self.get_partition_dir(partition)
        tdir = pdir / str(trip_id)
        if not tdir.is_dir():
            raise TripNotFoundError(f"Trip '{trip_id}' not found in partition '{partition}'.")

        # Load route server-side if present (never exposed to evaluator_test responses!)
        trip = load_raw_trip(tdir, load_route=True)
        # Keep memory bounded
        if len(self._trip_cache) > 25:
            self._trip_cache.clear()
        self._trip_cache[key] = trip
        return trip

    def get_trip_detail(self, trip_id: str, partition: str = "student_test") -> Dict[str, Any]:
        trip = self.load_trip(trip_id, partition=partition)
        track = trip.track  # [lon, lat, timestamp]
        duration = float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0
        # Convert [lon, lat, ts] -> [lat, lon, ts] for Folium/Leaflet
        gps_latlon = [
            [round(float(row[1]), 6), round(float(row[0]), 6), float(row[2])]
            for row in track
        ]
        return {
            "trip_id": trip.trip_id,
            "partition": partition,
            "num_observations": int(len(track)),
            "num_nodes": int(len(trip.nodes_lonlat)),
            "num_arcs": int(len(trip.arcs)),
            "has_reference_route": bool(trip.route is not None and partition != "evaluator_test"),
            "duration_s": round(duration, 2),
            "bounds": {
                "min_lat": float(track[:, 1].min()),
                "max_lat": float(track[:, 1].max()),
                "min_lon": float(track[:, 0].min()),
                "max_lon": float(track[:, 0].max()),
            },
            "gps_points": gps_latlon,
        }

    def get_trip_candidates(
        self, trip_id: str, partition: str = "student_test", load_labels: bool = True
    ) -> Tuple[RawTrip, pd.DataFrame, Dict[str, Any], float]:
        """Extract or fetch cached candidate DataFrame for a trip."""
        t0 = time.perf_counter()
        trip = self.load_trip(trip_id, partition=partition)
        has_route = trip.route is not None
        use_labels = bool(load_labels and has_route)
        key = (partition, str(trip_id), use_labels)

        if key in self._cand_cache:
            cand_df, summary = self._cand_cache[key]
            return trip, cand_df, summary, round((time.perf_counter() - t0) * 1000.0, 3)

        cand_df, summary = extract_trip_candidates(trip, cfg=self.cand_cfg, load_labels=use_labels)
        if len(self._cand_cache) > 20:
            self._cand_cache.clear()
        self._cand_cache[key] = (cand_df, summary)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return trip, cand_df, summary, elapsed_ms

    @staticmethod
    def arcs_to_latlon_segments(
        trip: RawTrip, arc_ids: List[int], deduplicate_consecutive: bool = True
    ) -> List[List[List[float]]]:
        """Convert 0-based arc_ids into Folium-ready [[lat1, lon1], [lat2, lon2]] segments.

        Explicitly maps raw `.nodes` [longitude, latitude] -> Folium [latitude, longitude].
        """
        if not arc_ids:
            return []
        seq = [int(arc_ids[0])]
        if deduplicate_consecutive:
            for a in arc_ids[1:]:
                ia = int(a)
                if ia != seq[-1]:
                    seq.append(ia)
        else:
            seq = [int(a) for a in arc_ids]

        nodes = trip.nodes_lonlat  # [lon, lat]
        arcs = trip.arcs  # [from_node_id, to_node_id]
        segments: List[List[List[float]]] = []
        n_arcs = len(arcs)
        for aid in seq:
            if 0 <= aid < n_arcs:
                u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
                segments.append(
                    [
                        [round(float(nodes[u, 1]), 6), round(float(nodes[u, 0]), 6)],
                        [round(float(nodes[v, 1]), 6), round(float(nodes[v, 0]), 6)],
                    ]
                )
        return segments

    def compute_trip_evaluation(
        self,
        trip: RawTrip,
        cand_df: pd.DataFrame,
        cand_summary: Dict[str, Any],
        probs: np.ndarray,
        partition: str,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Compute server-side metrics (when reference route exists) and classifier route continuity."""
        pdir = self.get_partition_dir(partition)
        continuity_agg, _ = compute_classifier_route_continuity(cand_df, probs, pdir)
        continuity_out = {
            "obs_step_continuity": round(float(continuity_agg["obs_step_continuity_overall"]), 6),
            "compressed_route_continuity": round(
                float(continuity_agg["compressed_route_continuity_overall"]), 6
            ),
            "total_obs_transitions": int(continuity_agg["total_obs_transitions"]),
            "continuous_obs_transitions": int(continuity_agg["continuous_obs_transitions"]),
            "disconnected_transitions": int(
                continuity_agg["total_obs_transitions"] - continuity_agg["continuous_obs_transitions"]
            ),
            "total_compressed_transitions": int(continuity_agg["total_compressed_transitions"]),
            "continuous_compressed_transitions": int(
                continuity_agg["continuous_compressed_transitions"]
            ),
        }

        metrics_out: Optional[Dict[str, Any]] = None
        if "label" in cand_df.columns and trip.route is not None:
            tot_obs = int(cand_summary.get("num_observations", len(trip.track)))
            misses = int(cand_summary.get("candidate_misses", 0))
            m = compute_candidate_and_obs_metrics(
                cand_df, probs, total_obs_count=tot_obs, candidate_miss_count=misses
            )
            metrics_out = m

        return metrics_out, continuity_out
