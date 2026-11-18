import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .candidates import road_segments
from .geo import point_segment_distances, project_trip


def _align_points_to_route(
    gps_xy,
    route_segments,
    timestamps_s,
    max_advance=6,
    skip_penalty=0.35,
    start_penalty=0.01,
    end_penalty=0.01,
):
    """Align GPS points to route positions with a forward-only dynamic program."""
    n_points = len(gps_xy)
    n_route = len(route_segments)
    if n_points == 0 or n_route == 0:
        raise ValueError("Cannot align an empty GPS trace or empty route")

    positions = np.arange(n_route, dtype=np.int32)
    back = np.zeros((n_points, n_route), dtype=np.int32)

    first_emission = point_segment_distances(gps_xy[0], route_segments) / 10.0
    previous_cost = first_emission + start_penalty * positions

    timestamps_s = np.asarray(timestamps_s, dtype=float)
    time_gaps = np.diff(timestamps_s, prepend=np.nan)
    if n_points > 1:
        time_gaps[0] = time_gaps[1]
    else:
        time_gaps[0] = 1.0

    for point_index in range(1, n_points):
        gap = time_gaps[point_index]
        allowed_advance = max_advance
        if np.isfinite(gap) and gap > 1.0:
            allowed_advance = max(allowed_advance, int(np.ceil(gap * 2.0)))
        allowed_advance = min(allowed_advance, n_route - 1)

        # Transition from route position k to j costs
        # previous_cost[k] + skip_penalty * (j-k), with k <= j.
        transformed = previous_cost - skip_penalty * positions
        padded = np.pad(
            transformed,
            (allowed_advance, 0),
            mode="constant",
            constant_values=np.inf,
        )
        windows = sliding_window_view(padded, allowed_advance + 1)
        best_offset = np.argmin(windows, axis=1)
        best_transformed_cost = windows.min(axis=1)
        previous_position = positions + best_offset - allowed_advance
        back[point_index] = np.maximum(previous_position, 0)

        emission = point_segment_distances(
            gps_xy[point_index], route_segments
        ) / 10.0
        previous_cost = (
            emission
            + skip_penalty * positions
            + best_transformed_cost
        )

    final_cost = previous_cost + end_penalty * (n_route - 1 - positions)
    aligned_positions = np.empty(n_points, dtype=np.int32)
    aligned_positions[-1] = int(np.argmin(final_cost))

    for point_index in range(n_points - 1, 0, -1):
        aligned_positions[point_index - 1] = back[
            point_index, aligned_positions[point_index]
        ]

    return aligned_positions


def assign_reference_labels(
    track,
    nodes,
    arcs,
    route,
    candidates,
    trip_id="",
):
    """Attach inferred binary labels after forward-only route alignment.

    `.route` gives an ordered sequence, not pointwise labels. Dynamic alignment
    uses GPS-to-route segment distances and route order. It remains an inferred
    alignment and should be visually checked on representative trips.
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
    if len(route_arc_ids) == 0:
        raise ValueError("Reference route is empty")

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

    route_positions = _align_points_to_route(
        gps_xy,
        route_segments,
        track["timestamp_s"].to_numpy(dtype=float),
    )
    reference_arc_by_observation = route_arc_ids[route_positions]

    result = candidates.copy()
    observation_indices = result["observation_index"].to_numpy(dtype=int)
    result["reference_arc_id"] = reference_arc_by_observation[observation_indices]
    result["label"] = (
        result["candidate_arc_id"].to_numpy(dtype=int)
        == result["reference_arc_id"].to_numpy(dtype=int)
    ).astype(int)
    result["trip_id"] = str(trip_id)
    return result
