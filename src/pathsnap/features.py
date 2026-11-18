import numpy as np

from .candidates import road_segments
from .geo import project_trip


FEATURE_COLUMNS = [
    "distance_to_road_m",
    "heading_difference_deg",
    "speed_mps",
    "time_gap_s",
    "incoming_degree",
    "outgoing_degree",
]


def build_features(track, nodes, arcs, candidates):
    """Add GPS-motion and road-connectivity features to candidate rows."""
    gps_xy, transformer = project_trip(
        track["longitude"],
        track["latitude"],
    )

    # Movement from the previous GPS point to the current one.
    movement = np.diff(gps_xy, axis=0, prepend=gps_xy[[0]])
    step_distance_m = np.linalg.norm(movement, axis=1)

    timestamps = track["timestamp_s"].to_numpy(dtype=float)
    time_gap_s = np.diff(timestamps, prepend=np.nan)
    time_gap_s[0] = np.nan

    speed_mps = np.full(len(track), np.nan)
    valid_time_gap = np.isfinite(time_gap_s) & (time_gap_s > 0)
    speed_mps[valid_time_gap] = (
        step_distance_m[valid_time_gap] / time_gap_s[valid_time_gap]
    )

    gps_heading_deg = (
        np.degrees(np.arctan2(movement[:, 0], movement[:, 1])) % 360.0
    )
    gps_heading_deg[step_distance_m < 0.5] = np.nan

    arc_ids, segments_xy = road_segments(nodes, arcs, transformer)

    road_heading_deg = (
        np.degrees(
            np.arctan2(
                segments_xy[:, 1, 0] - segments_xy[:, 0, 0],
                segments_xy[:, 1, 1] - segments_xy[:, 0, 1],
            )
        )
        % 360.0
    )
    heading_by_arc = dict(zip(arc_ids.tolist(), road_heading_deg.tolist()))

    incoming_degree = arcs.groupby("to_node_id").size().to_dict()
    outgoing_degree = arcs.groupby("from_node_id").size().to_dict()
    endpoints_by_arc = arcs.set_index("arc_id")[["from_node_id", "to_node_id"]]

    result = candidates.copy()

    obs = result["observation_index"].to_numpy(dtype=int)
    candidate_arcs = result["candidate_arc_id"].to_numpy(dtype=int)

    candidate_heading = np.array([
        heading_by_arc.get(int(arc_id), np.nan)
        for arc_id in candidate_arcs
    ])

    heading_difference = np.abs(
        (gps_heading_deg[obs] - candidate_heading + 180.0) % 360.0 - 180.0
    )

    from_nodes = np.array([
        endpoints_by_arc.loc[int(arc_id), "from_node_id"]
        for arc_id in candidate_arcs
    ])
    to_nodes = np.array([
        endpoints_by_arc.loc[int(arc_id), "to_node_id"]
        for arc_id in candidate_arcs
    ])

    result["heading_difference_deg"] = heading_difference
    result["speed_mps"] = speed_mps[obs]
    result["time_gap_s"] = time_gap_s[obs]
    result["incoming_degree"] = [
        int(incoming_degree.get(int(node_id), 0))
        for node_id in from_nodes
    ]
    result["outgoing_degree"] = [
        int(outgoing_degree.get(int(node_id), 0))
        for node_id in to_nodes
    ]

    return result
