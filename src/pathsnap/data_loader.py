
"""Raw dataset loading utilities for the PathSnap project."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class RawTrip:
    """In-memory representation of one PathSnap trip."""

    trip_id: str
    nodes_lonlat: np.ndarray
    arcs: np.ndarray
    track: np.ndarray
    route: Optional[np.ndarray] = None

    @property
    def num_observations(self) -> int:
        return int(len(self.track))

    @property
    def num_nodes(self) -> int:
        return int(len(self.nodes_lonlat))

    @property
    def num_arcs(self) -> int:
        return int(len(self.arcs))


def _load_array(path: Path, dtype: type, name: str) -> np.ndarray:
    """Load a whitespace-separated numerical file with validation."""

    if not path.is_file():
        raise FileNotFoundError(f"Required {name} file not found: {path}")

    try:
        data = np.loadtxt(path, dtype=dtype, ndmin=2)
    except (ValueError, OSError) as exc:
        raise ValueError(f"Could not read {name} file {path}: {exc}") from exc

    if data.size == 0:
        raise ValueError(f"{name} file is empty: {path}")

    if not np.isfinite(data).all():
        raise ValueError(f"{name} file contains NaN or Inf values: {path}")

    return data


def load_raw_trip(
    trip_dir: Path,
    load_route: bool = True,
) -> RawTrip:
    """
    Load one trip from its directory.

    Expected filenames:
        <trip_id>.track
        <trip_id>.nodes
        <trip_id>.arcs
        <trip_id>.route  (optional)
    """

    trip_dir = Path(trip_dir)

    if not trip_dir.is_dir():
        raise NotADirectoryError(f"Trip directory does not exist: {trip_dir}")

    trip_id = trip_dir.name

    nodes = _load_array(
        trip_dir / f"{trip_id}.nodes", np.float64, "nodes"
    )
    arcs = _load_array(
        trip_dir / f"{trip_id}.arcs", np.int64, "arcs"
    )
    track = _load_array(
        trip_dir / f"{trip_id}.track", np.float64, "track"
    )

    if nodes.shape[1] != 2:
        raise ValueError(
            f"{trip_id}: nodes must have 2 columns [longitude, latitude]; "
            f"got {nodes.shape}"
        )

    if arcs.shape[1] != 2:
        raise ValueError(
            f"{trip_id}: arcs must have 2 columns "
            "[from_node_id, to_node_id]; got {arcs.shape}"
        )

    if track.shape[1] != 3:
        raise ValueError(
            f"{trip_id}: track must have 3 columns "
            "[longitude, latitude, timestamp]; got {track.shape}"
        )

    if len(nodes) == 0 or len(arcs) == 0 or len(track) == 0:
        raise ValueError(f"{trip_id}: nodes, arcs and track must be non-empty")

    if np.any(nodes[:, 1] < -90) or np.any(nodes[:, 1] > 90):
        raise ValueError(f"{trip_id}: invalid node latitude")

    if np.any(nodes[:, 0] < -180) or np.any(nodes[:, 0] > 180):
        raise ValueError(f"{trip_id}: invalid node longitude")

    if np.any(track[:, 1] < -90) or np.any(track[:, 1] > 90):
        raise ValueError(f"{trip_id}: invalid GPS latitude")

    if np.any(track[:, 0] < -180) or np.any(track[:, 0] > 180):
        raise ValueError(f"{trip_id}: invalid GPS longitude")

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(
            f"{trip_id}: arc references a node outside the nodes array"
        )

    route = None
    route_path = trip_dir / f"{trip_id}.route"

    if load_route and route_path.is_file():
        try:
            route = np.loadtxt(
                route_path, dtype=np.int64, ndmin=1
            ).reshape(-1)
        except (ValueError, OSError) as exc:
            raise ValueError(
                f"Could not read route file {route_path}: {exc}"
            ) from exc

        if not np.isfinite(route).all():
            raise ValueError(f"{trip_id}: route contains invalid values")

        if np.any(route < 0) or np.any(route >= len(arcs)):
            raise ValueError(
                f"{trip_id}: route contains an invalid arc index"
            )

    return RawTrip(
        trip_id=trip_id,
        nodes_lonlat=nodes,
        arcs=arcs,
        track=track,
        route=route,
    )


def discover_trip_dirs(partition_dir: Path) -> List[Path]:
    """Find valid trip directories inside a dataset partition."""

    partition_dir = Path(partition_dir)

    if not partition_dir.is_dir():
        raise NotADirectoryError(
            f"Dataset partition does not exist: {partition_dir}"
        )

    trip_dirs = []

    for directory in sorted(partition_dir.iterdir()):
        if not directory.is_dir():
            continue

        trip_id = directory.name

        required_files = [
            directory / f"{trip_id}.track",
            directory / f"{trip_id}.nodes",
            directory / f"{trip_id}.arcs",
        ]

        if all(path.is_file() for path in required_files):
            trip_dirs.append(directory)

    return trip_dirs


def load_partition(
    partition_dir: Path,
    load_route: bool = True,
    strict: bool = True,
) -> List[RawTrip]:
    """
    Load all valid trips in a partition.

    strict=True:
        Stop and raise an error if any trip fails.

    strict=False:
        Skip invalid trips and continue loading.
    """

    trip_dirs = discover_trip_dirs(partition_dir)

    if not trip_dirs:
        raise FileNotFoundError(
            f"No valid trip directories found in {partition_dir}"
        )

    trips = []

    for trip_dir in trip_dirs:
        try:
            trip = load_raw_trip(trip_dir, load_route=load_route)
            trips.append(trip)

        except (ValueError, OSError) as exc:
            if strict:
                raise RuntimeError(
                    f"Failed loading trip {trip_dir.name}: {exc}"
                ) from exc

            print(f"[WARNING] Skipping {trip_dir.name}: {exc}")

    if not trips:
        raise RuntimeError(
            f"No trips could be loaded from {partition_dir}"
        )

    return trips
