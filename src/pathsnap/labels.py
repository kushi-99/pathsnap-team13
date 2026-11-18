import numpy as np

from .candidates import road_segments
from .geo import point_segment_distances, project_trip


def assign_reference_labels(
    track,
    nodes,
    arcs,
    route,
    candidates,
    trip_id="",
):
    """Attach provisional binary labels to candidate rows.

    For each GPS point:
      1. Find its nearest arc among the reference route arcs.
      2. Enforce nondecreasing position through the route sequence.
      3. Mark the matching candidate arc as label 1.

    Inspect this alignment on trips before treating it as final.
    """
    gps_xy, transformer = project_trip(
        track["longitude"],
        track["latitude"],
    )

    all_arc_ids, all_segments = road_segments(nodes, arcs, transformer)
    segment_by_arc = {
        int(arc_id): all_segments[position]
        for position, arc_id in enumerate(all_arc_ids)
    }

    route_arc_ids = np.asarray(route, dtype=int)

    missing_route_arcs = [
        int(arc_id)
        for arc_id in route_arc_ids
        if int(arc_id) not in segment_by_arc
    ]
    if missing_route_arcs:
        raise ValueError(
            f"Route contains invalid arc IDs, such as {missing_route_arcs[:5]}"
        )

    route_segments = np.stack([
        segment_by_arc[int(arc_id)]
        for arc_id in route_arc_ids
    ])

    nearest_route_position = np.empty(len(gps_xy), dtype=int)

    for observation_index, point_xy in enumerate(gps_xy):
        distances = point_segment_distances(point_xy, route_segments)
        nearest_route_position[observation_index] = int(np.argmin(distances))

    # GPS noise can make nearest route positions jump backward.
    ordered_route_position = np.maximum.accumulate(nearest_route_position)
    reference_arc_by_observation = route_arc_ids[ordered_route_position]

    result = candidates.copy()
    observation_indices = result["observation_index"].to_numpy(dtype=int)

    result["reference_arc_id"] = reference_arc_by_observation[observation_indices]
    result["label"] = (
        result["candidate_arc_id"].to_numpy(dtype=int)
        == result["reference_arc_id"].to_numpy(dtype=int)
    ).astype(int)
    result["trip_id"] = str(trip_id)

    return result
