from pathlib import Path

import numpy as np
import pandas as pd


def locate_split(data_root, split):
    """Find a split folder in either supported extracted-ZIP layout."""
    root = Path(data_root)

    possible_paths = [
        root / split,
        root / "Student folder" / split,
    ]

    for path in possible_paths:
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Could not find '{split}' under '{root}'. "
        "Check where you extracted the dataset."
    )


def list_trip_ids(data_root, split="train"):
    split_folder = locate_split(data_root, split)

    return sorted(
        folder.name
        for folder in split_folder.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    )


def load_trip(data_root, trip_id, split="train"):
    """Load one trip's track, nodes, arcs and reference route."""
    trip_folder = locate_split(data_root, split) / str(trip_id)
    prefix = trip_folder / str(trip_id)

    track = pd.read_csv(
        prefix.with_suffix(".track"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp_s"],
    )

    nodes = pd.read_csv(
        prefix.with_suffix(".nodes"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
    )

    arcs = pd.read_csv(
        prefix.with_suffix(".arcs"),
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
    )

    # In this dataset, an arc's ID is its row number in the .arcs file.
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int64))

    route = pd.read_csv(
        prefix.with_suffix(".route"),
        sep=r"\s+",
        header=None,
        names=["arc_id"],
    )["arc_id"].astype("int64")

    return {
        "trip_id": str(trip_id),
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }
    