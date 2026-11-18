import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import point_segment_distances, project_trip


def road_segments(nodes, arcs, transformer):
    """Convert arc rows into projected, directed road segments."""
    node_x, node_y = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy(),
    )
    node_xy = np.column_stack((node_x, node_y))

    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)

    valid = (
        (endpoints >= 0).all(axis=1)
        & (endpoints < len(node_xy)).all(axis=1)
    )

    valid_arcs = arcs.loc[valid]
    valid_endpoints = endpoints[valid]

    arc_ids = valid_arcs["arc_id"].to_numpy(dtype=int)
    segments_xy = node_xy[valid_endpoints]

    return arc_ids, segments_xy


def generate_candidates(
    track,
    nodes,
    arcs,
    trip_id="",
    k=40,
    radius_m=100.0,
):
    """Return nearby candidate arcs for each GPS observation."""
    gps_xy, transformer = project_trip(
        track["longitude"],
        track["latitude"],
    )

    arc_ids, segments_xy = road_segments(nodes, arcs, transformer)

    if len(arc_ids) == 0:
        raise ValueError("This trip has no valid road arcs.")

    midpoints = segments_xy.mean(axis=1)
    midpoint_tree = cKDTree(midpoints)

    k = min(int(k), len(arc_ids))
    rows = []

    for observation_index, point_xy in enumerate(gps_xy):
        _, nearby_indices = midpoint_tree.query(point_xy, k=k)
        nearby_indices = np.atleast_1d(nearby_indices).astype(int)

        distances = point_segment_distances(
            point_xy,
            segments_xy[nearby_indices],
        )

        within_radius = distances <= radius_m

        # Ensure each observation has at least one candidate.
        if not within_radius.any():
            within_radius[np.argmin(distances)] = True

        for local_index in np.flatnonzero(within_radius):
            arc_position = nearby_indices[local_index]

            rows.append({
                "trip_id": str(trip_id),
                "observation_index": observation_index,
                "candidate_arc_id": int(arc_ids[arc_position]),
                "distance_to_road_m": float(distances[local_index]),
            })

    return pd.DataFrame(rows)
