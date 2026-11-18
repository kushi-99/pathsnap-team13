import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import list_trip_ids, load_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--split",
        choices=["train", "student_test"],
        default="train",
    )
    parser.add_argument("--out", default="results/eda_inventory.csv")
    args = parser.parse_args()

    rows = []

    for trip_id in list_trip_ids(args.data_root, args.split):
        trip = load_trip(args.data_root, trip_id, args.split)

        track = trip["track"]
        nodes = trip["nodes"]
        arcs = trip["arcs"]
        route = trip["route"]

        timestamps = track["timestamp_s"].to_numpy(dtype=float)
        time_gaps = np.diff(timestamps)

        valid_arc_endpoints = (
            (arcs["from_node_id"] >= 0)
            & (arcs["to_node_id"] >= 0)
            & (arcs["from_node_id"] < len(nodes))
            & (arcs["to_node_id"] < len(nodes))
        )

        invalid_coordinates = (
            ~track["longitude"].between(-180, 180)
            | ~track["latitude"].between(-90, 90)
        )

        rows.append({
            "split": args.split,
            "trip_id": trip_id,
            "gps_observations": len(track),
            "road_nodes": len(nodes),
            "road_arcs": len(arcs),
            "route_arcs": len(route),
            "missing_track_cells": int(track.isna().sum().sum()),
            "invalid_coordinates": int(invalid_coordinates.sum()),
            "timestamps_strictly_increasing": bool((time_gaps > 0).all()),
            "median_time_gap_s": (
                float(np.median(time_gaps)) if len(time_gaps) else np.nan
            ),
            "max_time_gap_s": (
                float(np.max(time_gaps)) if len(time_gaps) else np.nan
            ),
            "invalid_arc_endpoints": int((~valid_arc_endpoints).sum()),
            "min_longitude": float(track["longitude"].min()),
            "max_longitude": float(track["longitude"].max()),
            "min_latitude": float(track["latitude"].min()),
            "max_latitude": float(track["latitude"].max()),
        })

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Saved EDA inventory to {output_path}")


if __name__ == "__main__":
    main()