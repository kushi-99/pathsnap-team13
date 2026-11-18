import numpy as np
from pyproj import CRS, Transformer


def project_trip(longitude, latitude):
    """Project one local trip into its UTM zone."""
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)

    mean_lon = float(np.nanmean(longitude))
    mean_lat = float(np.nanmean(latitude))

    zone = int(np.floor((mean_lon + 180.0) / 6.0) + 1)
    epsg = (32600 if mean_lat >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_epsg(epsg),
        always_xy=True,
    )

    x, y = transformer.transform(longitude, latitude)
    return np.column_stack((x, y)), transformer


def point_segment_distances(point_xy, segments_xy):
    """Return the distance from one point to each line segment."""
    start = segments_xy[:, 0, :]
    end = segments_xy[:, 1, :]

    direction = end - start
    length_squared = np.einsum("ij,ij->i", direction, direction)
    length_squared = np.maximum(length_squared, 1e-12)

    position = np.einsum("ij,ij->i", point_xy - start, direction)
    position = position / length_squared
    position = np.clip(position, 0.0, 1.0)

    nearest_points = start + position[:, None] * direction
    return np.linalg.norm(nearest_points - point_xy, axis=1)
