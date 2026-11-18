"""Modular candidate-generation, label-alignment, and feature-extraction adapter.

This module bridges raw PathSnap trip folders (.track, .nodes, .arcs, optional .route)
and the standardized candidate-level DataFrame expected by the MLP pipeline:

    Columns:
      - trip_id (str)
      - obs_idx (int)
      - arc_id (int, original 0-based row index in <trip>.arcs)
      - <feature_columns> (float32, purely observable from .track/.nodes/.arcs)
      - label (int, 0 or 1; only present when load_labels=True)

If another team member supplies a shared preprocessed candidate table (CSV/Parquet)
with this schema, `load_external_candidate_dataset()` can load it directly without
modifying any MLP model or training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ml.mlp.config import (
    CACHE_DIR,
    CandidateConfig,
    DEFAULT_FEATURE_COLUMNS,
)

EARTH_RADIUS_M = 6371000.0


@dataclass
class RawTrip:
    """In-memory representation of a raw PathSnap trip."""

    trip_id: str
    nodes_lonlat: np.ndarray  # (N, 2) float64: [lon, lat]
    arcs: np.ndarray  # (A, 2) int64: [from_node_id, to_node_id]
    track: np.ndarray  # (T, 3) float64: [lon, lat, timestamp]
    route: Optional[np.ndarray] = None  # (R,) int64: arc_ids


def load_raw_trip(trip_dir: Path, load_route: bool = True) -> RawTrip:
    """Load raw .track, .nodes, .arcs (and optionally .route) from a trip directory."""
    tid = trip_dir.name
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes", dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)

    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    track = np.loadtxt(trip_dir / f"{tid}.track", dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)

    route = None
    route_path = trip_dir / f"{tid}.route"
    if load_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64)
        if route.ndim == 0:
            route = route.reshape(1)

    return RawTrip(
        trip_id=tid,
        nodes_lonlat=nodes,
        arcs=arcs,
        track=track,
        route=route,
    )


def project_local_tangent_plane(
    lonlat: np.ndarray, lon0: float, lat0: float
) -> np.ndarray:
    """Project (lon, lat) degrees to local tangent-plane meters centered at (lon0, lat0)."""
    cos_lat0 = math.cos(math.radians(lat0))
    x = np.radians(lonlat[:, 0] - lon0) * cos_lat0 * EARTH_RADIUS_M
    y = np.radians(lonlat[:, 1] - lat0) * EARTH_RADIUS_M
    return np.column_stack((x, y))


def compute_track_kinematics(
    track_xy: np.ndarray, timestamps: np.ndarray, cfg: CandidateConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local heading (rad), heading_valid, speed (m/s), dt_prev, is_time_gap."""
    n = len(track_xy)
    dt_prev = np.zeros(n, dtype=np.float64)
    dt_next = np.zeros(n, dtype=np.float64)
    if n > 1:
        diffs = np.diff(timestamps)
        dt_prev[1:] = diffs
        dt_next[:-1] = diffs
        dt_prev[0] = diffs[0]
        dt_next[-1] = diffs[-1]

    is_gap_prev = dt_prev > cfg.time_gap_threshold_s
    is_gap_next = dt_next > cfg.time_gap_threshold_s
    is_time_gap = (is_gap_prev | is_gap_next).astype(np.float64)

    # Local displacement vector for speed & heading (guard against time gaps)
    headings = np.zeros(n, dtype=np.float64)
    heading_valid = np.zeros(n, dtype=np.float64)
    speeds = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Pick best local window without crossing a large time gap
        i_prev = i - 1 if (i > 0 and not is_gap_prev[i]) else i
        i_next = i + 1 if (i < n - 1 and not is_gap_next[i]) else i

        if i_next > i_prev:
            dx = track_xy[i_next, 0] - track_xy[i_prev, 0]
            dy = track_xy[i_next, 1] - track_xy[i_prev, 1]
            dt = timestamps[i_next] - timestamps[i_prev]
            dist = math.hypot(dx, dy)
            if dt > 0:
                speeds[i] = min(dist / dt, cfg.max_speed_mps)
        else:
            dx, dy, dist = 0.0, 0.0, 0.0

        # If immediate window displacement is tiny (<1.5 m), widen slightly up to +-3 steps
        if dist < 1.5:
            best_dx, best_dy, best_d = dx, dy, dist
            for w in (2, 3):
                j0 = max(0, i - w)
                j1 = min(n - 1, i + w)
                # Ensure no big gap inside [j0, j1]
                if j1 > j0 and (timestamps[j1] - timestamps[j0]) <= (2 * w * cfg.time_gap_threshold_s):
                    wdx = track_xy[j1, 0] - track_xy[j0, 0]
                    wdy = track_xy[j1, 1] - track_xy[j0, 1]
                    wd = math.hypot(wdx, wdy)
                    if wd > best_d:
                        best_dx, best_dy, best_d = wdx, wdy, wd
                    if best_d >= 1.5:
                        break
            dx, dy, dist = best_dx, best_dy, best_d

        if dist >= 1.5:
            headings[i] = math.atan2(dy, dx)
            heading_valid[i] = 1.0

    return headings, heading_valid, speeds, np.minimum(dt_prev, 60.0), is_time_gap


def point_to_segments_geometry(
    px: float,
    py: float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized geometry from one point (px, py) to M segments (a -> b).

    Returns: (perp_dist, proj_ratio, is_interior, dist_start, dist_end)
    """
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    raw_t = np.where(l2 > 1e-12, ((px - ax) * dx + (py - ay) * dy) / np.maximum(l2, 1e-12), 0.0)
    is_interior = ((raw_t >= 0.0) & (raw_t <= 1.0)).astype(np.float64)
    t = np.clip(raw_t, 0.0, 1.0)
    cx = ax + t * dx
    cy = ay + t * dy
    perp_dist = np.hypot(px - cx, py - cy)
    dist_start = np.hypot(px - ax, py - ay)
    dist_end = np.hypot(px - bx, py - by)
    return perp_dist, t, is_interior, dist_start, dist_end


def align_track_to_route(
    track_xy: np.ndarray,
    track_headings: np.ndarray,
    heading_valid: np.ndarray,
    route_arc_ids: np.ndarray,
    node_xy: np.ndarray,
    arcs: np.ndarray,
    cfg: CandidateConfig,
) -> np.ndarray:
    """Monotonic dynamic-programming alignment from GPS observations to `.route` arcs.

    Finds a non-decreasing sequence of indices j_0 <= j_1 <= ... <= j_{T-1} into
    `route_arc_ids` minimizing cumulative geometric + heading alignment cost.
    Runs in exact O(T * R) using vectorized NumPy prefix-min scanning.
    """
    T = len(track_xy)
    R = len(route_arc_ids)
    u_ids = arcs[route_arc_ids, 0]
    v_ids = arcs[route_arc_ids, 1]
    ax = node_xy[u_ids, 0]
    ay = node_xy[u_ids, 1]
    bx = node_xy[v_ids, 0]
    by = node_xy[v_ids, 1]
    dx = bx - ax
    dy = by - ay
    l2 = np.maximum(dx * dx + dy * dy, 1e-12)
    arc_headings = np.arctan2(dy, dx)

    j_idx = np.arange(R, dtype=np.int64)
    alpha = cfg.align_step_penalty_m
    alpha_j = alpha * j_idx

    backptr = np.zeros((T, R), dtype=np.int32)
    dp_prev = np.zeros(R, dtype=np.float64)

    for t in range(T):
        px, py = track_xy[t, 0], track_xy[t, 1]
        proj = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
        dist = np.hypot(px - (ax + proj * dx), py - (ay + proj * dy))
        if heading_valid[t] > 0.5:
            cos_diff = np.cos(track_headings[t] - arc_headings)
            # Heading penalty in [0, align_heading_weight_m]
            h_cost = 0.5 * (1.0 - cos_diff) * cfg.align_heading_weight_m
            cost_t = dist + h_cost
        else:
            cost_t = dist

        if t == 0:
            dp_prev = cost_t
            backptr[0] = j_idx
        else:
            shifted = dp_prev - alpha_j
            best_val = np.minimum.accumulate(shifted)
            changed = np.empty(R, dtype=bool)
            changed[0] = True
            changed[1:] = best_val[1:] < best_val[:-1]
            best_idx = np.maximum.accumulate(np.where(changed, j_idx, 0))
            dp_prev = best_val + alpha_j + cost_t
            backptr[t] = best_idx

    aligned_j = np.empty(T, dtype=np.int64)
    curr = int(np.argmin(dp_prev))
    aligned_j[T - 1] = curr
    for t in range(T - 1, 0, -1):
        curr = int(backptr[t, curr])
        aligned_j[t - 1] = curr

    return route_arc_ids[aligned_j]


def extract_trip_candidates(
    trip: RawTrip,
    cfg: CandidateConfig = CandidateConfig(),
    load_labels: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate spatial candidates and candidate-level features for a single trip.

    Returns:
        df: Candidate-level DataFrame (one row per (obs_idx, candidate arc_id)).
        summary: Trip-level candidate-generation diagnostics (coverage, miss rate, etc.).
    """
    track = trip.track
    T = len(track)
    lon0 = float(np.mean(track[:, 0]))
    lat0 = float(np.mean(track[:, 1]))

    track_xy = project_local_tangent_plane(track[:, :2], lon0, lat0)
    node_xy = project_local_tangent_plane(trip.nodes_lonlat, lon0, lat0)
    arcs = trip.arcs
    num_arcs = len(arcs)

    # Pre-compute full-network topology (node degrees and reverse-arc existence)
    num_nodes = len(node_xy)
    node_deg = np.zeros(num_nodes, dtype=np.int32)
    np.add.at(node_deg, arcs[:, 0], 1)
    np.add.at(node_deg, arcs[:, 1], 1)

    # Filter arcs to padded bounding box of the GPS track (exact optimization for huge graphs)
    pad = cfg.fallback_radius_m + 50.0
    xmin, ymin = track_xy.min(axis=0) - pad
    xmax, ymax = track_xy.max(axis=0) + pad

    u_xy_all = node_xy[arcs[:, 0]]
    v_xy_all = node_xy[arcs[:, 1]]
    seg_xmin = np.minimum(u_xy_all[:, 0], v_xy_all[:, 0])
    seg_xmax = np.maximum(u_xy_all[:, 0], v_xy_all[:, 0])
    seg_ymin = np.minimum(u_xy_all[:, 1], v_xy_all[:, 1])
    seg_ymax = np.maximum(u_xy_all[:, 1], v_xy_all[:, 1])

    local_mask = (
        (seg_xmax >= xmin)
        & (seg_xmin <= xmax)
        & (seg_ymax >= ymin)
        & (seg_ymin <= ymax)
    )
    local_arc_ids = np.flatnonzero(local_mask)
    if len(local_arc_ids) == 0:
        local_arc_ids = np.arange(num_arcs, dtype=np.int64)

    local_arcs = arcs[local_arc_ids]
    local_u_xy = u_xy_all[local_arc_ids]
    local_v_xy = v_xy_all[local_arc_ids]
    local_dx = local_v_xy[:, 0] - local_u_xy[:, 0]
    local_dy = local_v_xy[:, 1] - local_u_xy[:, 1]
    local_len = np.hypot(local_dx, local_dy)
    local_heading = np.arctan2(local_dy, local_dx)

    # Reverse-arc lookup only on local arcs (fast set membership)
    local_edge_set = set(zip(local_arcs[:, 0].tolist(), local_arcs[:, 1].tolist()))
    local_has_rev = np.fromiter(
        (1.0 if (v, u) in local_edge_set else 0.0 for u, v in local_arcs),
        dtype=np.float64,
        count=len(local_arcs),
    )

    # Build sub-segment sampled spatial KD-tree over local arcs
    n_samples = np.maximum(2, np.ceil(local_len / cfg.segment_sample_step_m).astype(np.int64) + 1)
    total_samples = int(n_samples.sum())
    sample_pts = np.empty((total_samples, 2), dtype=np.float64)
    sample_local_idx = np.empty(total_samples, dtype=np.int64)

    offset = 0
    for idx_i, ns in enumerate(n_samples):
        ts = np.linspace(0.0, 1.0, int(ns))
        ux, uy = local_u_xy[idx_i]
        dx_i, dy_i = local_dx[idx_i], local_dy[idx_i]
        end = offset + int(ns)
        sample_pts[offset:end, 0] = ux + ts * dx_i
        sample_pts[offset:end, 1] = uy + ts * dy_i
        sample_local_idx[offset:end] = idx_i
        offset = end

    kdtree = cKDTree(sample_pts)

    # Track kinematics
    headings, heading_valid, speeds, dt_prev_clipped, is_time_gap = compute_track_kinematics(
        track_xy, track[:, 2], cfg
    )

    # Optional label alignment (never called on unlabeled evaluator trips)
    aligned_ref_arcs: Optional[np.ndarray] = None
    if load_labels and trip.route is not None and len(trip.route) > 0:
        aligned_ref_arcs = align_track_to_route(
            track_xy, headings, heading_valid, trip.route, node_xy, arcs, cfg
        )

    # Step 1: Query candidates per observation
    radius_hits = kdtree.query_ball_point(track_xy, r=cfg.search_radius_m)

    obs_cand_local_idx: List[np.ndarray] = []
    obs_cand_dists: List[np.ndarray] = []
    obs_cand_proj: List[np.ndarray] = []
    obs_cand_interior: List[np.ndarray] = []
    obs_cand_dstart: List[np.ndarray] = []
    obs_cand_dend: List[np.ndarray] = []
    obs_top3_arcs: List[set] = []
    obs_top3_from: List[set] = []
    obs_top3_to: List[set] = []

    for t in range(T):
        hit_indices = radius_hits[t]
        if len(hit_indices) == 0:
            # Fallback to larger radius or K nearest sample points
            hit_indices = kdtree.query_ball_point(track_xy[t], r=cfg.fallback_radius_m)
        if len(hit_indices) == 0:
            _, nn_idx = kdtree.query(track_xy[t], k=min(cfg.max_candidates * 4, total_samples))
            hit_indices = np.atleast_1d(nn_idx)

        cand_lidx = np.unique(sample_local_idx[hit_indices])
        px, py = track_xy[t, 0], track_xy[t, 1]
        pdist, proj, is_int, dstart, dend = point_to_segments_geometry(
            px,
            py,
            local_u_xy[cand_lidx, 0],
            local_u_xy[cand_lidx, 1],
            local_v_xy[cand_lidx, 0],
            local_v_xy[cand_lidx, 1],
        )

        # Keep top max_candidates closest arcs
        if len(cand_lidx) > cfg.max_candidates:
            order = np.argpartition(pdist, cfg.max_candidates)[: cfg.max_candidates]
            order = order[np.argsort(pdist[order])]
        else:
            order = np.argsort(pdist)

        cand_lidx = cand_lidx[order]
        pdist = pdist[order]
        proj = proj[order]
        is_int = is_int[order]
        dstart = dstart[order]
        dend = dend[order]

        obs_cand_local_idx.append(cand_lidx)
        obs_cand_dists.append(pdist)
        obs_cand_proj.append(proj)
        obs_cand_interior.append(is_int)
        obs_cand_dstart.append(dstart)
        obs_cand_dend.append(dend)

        top_k = cand_lidx[:3]
        obs_top3_arcs.append(set(local_arc_ids[top_k].tolist()))
        obs_top3_from.append(set(local_arcs[top_k, 0].tolist()))
        obs_top3_to.append(set(local_arcs[top_k, 1].tolist()))

    # Step 2: Assemble candidate feature matrix
    total_rows = sum(len(c) for c in obs_cand_local_idx)
    n_feats = len(DEFAULT_FEATURE_COLUMNS)
    X = np.empty((total_rows, n_feats), dtype=np.float32)
    obs_idx_arr = np.empty(total_rows, dtype=np.int32)
    arc_id_arr = np.empty(total_rows, dtype=np.int64)
    y_arr = np.zeros(total_rows, dtype=np.int8) if aligned_ref_arcs is not None else None
    ref_arc_per_obs = aligned_ref_arcs

    candidate_hits = 0
    row_ptr = 0

    for t in range(T):
        cand_lidx = obs_cand_local_idx[t]
        k = len(cand_lidx)
        end = row_ptr + k

        pdist = obs_cand_dists[t]
        proj = obs_cand_proj[t]
        is_int = obs_cand_interior[t]
        dstart = obs_cand_dstart[t]
        dend = obs_cand_dend[t]
        alen = local_len[cand_lidx]
        c_arc_ids = local_arc_ids[cand_lidx]
        c_from = local_arcs[cand_lidx, 0]
        c_to = local_arcs[cand_lidx, 1]

        # Directional alignment
        if heading_valid[t] > 0.5:
            diff_rad = headings[t] - local_heading[cand_lidx]
            cos_h = np.cos(diff_rad)
            sin_h_abs = np.abs(np.sin(diff_rad))
            ang_deg = np.degrees(np.arccos(np.clip(cos_h, -1.0, 1.0)))
            h_norm = ang_deg / 180.0
        else:
            cos_h = np.zeros(k, dtype=np.float64)
            sin_h_abs = np.zeros(k, dtype=np.float64)
            h_norm = np.full(k, 0.5, dtype=np.float64)

        # Temporal context (distances from t-1 and t+1 to candidate arcs)
        ax_c = local_u_xy[cand_lidx, 0]
        ay_c = local_u_xy[cand_lidx, 1]
        bx_c = local_v_xy[cand_lidx, 0]
        by_c = local_v_xy[cand_lidx, 1]

        if t > 0 and (track[t, 2] - track[t - 1, 2]) <= cfg.time_gap_threshold_s:
            prev_d, prev_proj, _, _, _ = point_to_segments_geometry(
                track_xy[t - 1, 0], track_xy[t - 1, 1], ax_c, ay_c, bx_c, by_c
            )
            prev_a_set = obs_top3_arcs[t - 1]
            prev_to_set = obs_top3_to[t - 1]
            is_prev_conn = np.fromiter(
                (1.0 if (aid in prev_a_set or fn in prev_to_set) else 0.0 for aid, fn in zip(c_arc_ids, c_from)),
                dtype=np.float64,
                count=k,
            )
        else:
            prev_d = pdist
            prev_proj = proj
            is_prev_conn = np.zeros(k, dtype=np.float64)

        if t < T - 1 and (track[t + 1, 2] - track[t, 2]) <= cfg.time_gap_threshold_s:
            next_d, next_proj, _, _, _ = point_to_segments_geometry(
                track_xy[t + 1, 0], track_xy[t + 1, 1], ax_c, ay_c, bx_c, by_c
            )
            next_a_set = obs_top3_arcs[t + 1]
            next_from_set = obs_top3_from[t + 1]
            is_next_conn = np.fromiter(
                (1.0 if (aid in next_a_set or tn in next_from_set) else 0.0 for aid, tn in zip(c_arc_ids, c_to)),
                dtype=np.float64,
                count=k,
            )
        else:
            next_d = pdist
            next_proj = proj
            is_next_conn = np.zeros(k, dtype=np.float64)

        progress = next_proj - prev_proj

        # Competitive / relative features within observation t
        min_d = float(pdist[0])  # sorted ascending
        ranks = np.arange(k, dtype=np.float64) / max(k - 1, 1)
        diff_min = pdist - min_d
        ratio_min = (pdist + 1.0) / (min_d + 1.0)
        best_cos = float(np.max(cos_h))
        cos_vs_best = cos_h - best_cos

        X[row_ptr:end, 0] = pdist
        X[row_ptr:end, 1] = np.log1p(pdist)
        X[row_ptr:end, 2] = proj
        X[row_ptr:end, 3] = is_int
        X[row_ptr:end, 4] = dstart
        X[row_ptr:end, 5] = dend
        X[row_ptr:end, 6] = alen
        X[row_ptr:end, 7] = np.log1p(alen)
        X[row_ptr:end, 8] = cos_h
        X[row_ptr:end, 9] = sin_h_abs
        X[row_ptr:end, 10] = h_norm
        X[row_ptr:end, 11] = heading_valid[t]
        X[row_ptr:end, 12] = speeds[t]
        X[row_ptr:end, 13] = dt_prev_clipped[t]
        X[row_ptr:end, 14] = is_time_gap[t]
        X[row_ptr:end, 15] = prev_d
        X[row_ptr:end, 16] = next_d
        X[row_ptr:end, 17] = progress
        X[row_ptr:end, 18] = is_prev_conn
        X[row_ptr:end, 19] = is_next_conn
        X[row_ptr:end, 20] = local_has_rev[cand_lidx]
        X[row_ptr:end, 21] = node_deg[c_from]
        X[row_ptr:end, 22] = node_deg[c_to]
        X[row_ptr:end, 23] = ranks
        X[row_ptr:end, 24] = diff_min
        X[row_ptr:end, 25] = ratio_min
        X[row_ptr:end, 26] = cos_vs_best
        X[row_ptr:end, 27] = float(k)

        obs_idx_arr[row_ptr:end] = t
        arc_id_arr[row_ptr:end] = c_arc_ids

        if y_arr is not None and ref_arc_per_obs is not None:
            ref_aid = int(ref_arc_per_obs[t])
            matches = (c_arc_ids == ref_aid)
            if np.any(matches):
                y_arr[row_ptr:end] = matches.astype(np.int8)
                candidate_hits += 1

        row_ptr = end

    df = pd.DataFrame(X, columns=DEFAULT_FEATURE_COLUMNS)
    df.insert(0, "arc_id", arc_id_arr)
    df.insert(0, "obs_idx", obs_idx_arr)
    df.insert(0, "trip_id", trip.trip_id)
    if y_arr is not None:
        df["label"] = y_arr

    summary = {
        "trip_id": trip.trip_id,
        "num_observations": T,
        "num_candidate_rows": total_rows,
        "avg_candidates_per_obs": total_rows / max(T, 1),
    }
    if aligned_ref_arcs is not None:
        miss_count = T - candidate_hits
        summary["candidate_hits"] = candidate_hits
        summary["candidate_misses"] = miss_count
        summary["candidate_miss_rate"] = miss_count / max(T, 1)

    return df, summary


def build_or_load_partition_dataset(
    partition_dir: Path,
    partition_name: str,
    cfg: CandidateConfig = CandidateConfig(),
    use_cache: bool = True,
    load_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build or load cached candidate dataset and per-trip summary for a partition."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{partition_name}_k{cfg.max_candidates}_r{int(cfg.search_radius_m)}"
    data_cache = CACHE_DIR / f"{cache_key}.pkl"
    summary_cache = CACHE_DIR / f"{cache_key}_summary.csv"

    if use_cache and data_cache.exists() and summary_cache.exists():
        df = pd.read_pickle(data_cache)
        summary_df = pd.read_csv(summary_cache, dtype={"trip_id": str})
        return df, summary_df

    trip_dirs = sorted([d for d in partition_dir.iterdir() if d.is_dir()])
    dfs: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []

    t0 = time.perf_counter()
    for idx, tdir in enumerate(trip_dirs, 1):
        trip = load_raw_trip(tdir, load_route=load_labels)
        tdf, tsum = extract_trip_candidates(trip, cfg=cfg, load_labels=load_labels)
        dfs.append(tdf)
        summaries.append(tsum)
        if idx % 10 == 0 or idx == len(trip_dirs):
            elapsed = time.perf_counter() - t0
            print(f"[{partition_name}] Processed {idx}/{len(trip_dirs)} trips ({elapsed:.1f}s)")

    full_df = pd.concat(dfs, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    if use_cache:
        full_df.to_pickle(data_cache)
        summary_df.to_csv(summary_cache, index=False)

    return full_df, summary_df


def load_external_candidate_dataset(path: Path) -> pd.DataFrame:
    """Load an external team-supplied candidate table (CSV, Parquet, or Pickle)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype={"trip_id": str})
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported dataset extension: {suffix}")


if __name__ == "__main__":
    from ml.mlp.config import TRAIN_DIR, STUDENT_TEST_DIR

    cfg = CandidateConfig()
    for name, pdir in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        df, sdf = build_or_load_partition_dataset(pdir, name, cfg=cfg, use_cache=True)
        tot_obs = int(sdf["num_observations"].sum())
        tot_miss = int(sdf["candidate_misses"].sum())
        pos_rate = float(df["label"].mean())
        print(
            f"\n=== {name.upper()} SUMMARY ===\n"
            f"  Trips: {len(sdf)}, Total obs: {tot_obs:,}, Candidate rows: {len(df):,}\n"
            f"  Positive class rate: {pos_rate:.4f} ({int(df['label'].sum()):,} positives)\n"
            f"  Overall Candidate Miss Rate: {tot_miss}/{tot_obs} ({tot_miss/tot_obs*100:.2f}%)\n"
            f"  Any NaN/Inf in features: {not np.isfinite(df[DEFAULT_FEATURE_COLUMNS].to_numpy()).all()}"
        )
