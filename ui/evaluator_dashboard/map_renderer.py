"""Folium geographical map renderer for the PathSnap Evaluator Dashboard.

Visualises ONLY:
  1. Original raw GPS trajectory observations (blue dots + faint connecting trace)
  2. Reconstructed / predicted road route segments (crimson polyline)
  3. Start and End trip markers

Does NOT render or accept evaluator ground-truth routes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import folium


def build_prediction_map(
    gps_points: Sequence[Sequence[float]],
    predicted_segments: Sequence[Sequence[Sequence[float]]],
    trip_id: str,
    model_label: str = "PathSnap Model",
) -> folium.Map:
    """Create an interactive Folium map from API-returned coordinates.

    Args:
        gps_points: Sequence of [lat, lon] pairs (or [lat, lon, timestamp]).
        predicted_segments: Sequence of road segments, each as [[lat1, lon1], [lat2, lon2]].
        trip_id: Selected trip identifier for legend/tooltips.
        model_label: Model/decoder label for legend.
    """
    lats: List[float] = []
    lons: List[float] = []

    for pt in gps_points:
        lats.append(float(pt[0]))
        lons.append(float(pt[1]))
    for seg in predicted_segments:
        for pt in seg:
            lats.append(float(pt[0]))
            lons.append(float(pt[1]))

    if lats and lons:
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    else:
        center = [12.9345, 77.5345]

    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Predicted Road Route (from Laptop 1 API)
    route_layer = folium.FeatureGroup(name="Predicted Road Route", show=True)
    if predicted_segments:
        folium.PolyLine(
            locations=predicted_segments,
            color="#d62728",
            weight=4.5,
            opacity=0.88,
            tooltip=f"Predicted Route — {model_label}",
        ).add_to(route_layer)
    route_layer.add_to(fmap)

    # Layer 2: Original Raw GPS Trajectory
    gps_layer = folium.FeatureGroup(name="Original GPS Observations", show=True)
    if gps_points:
        gps_latlon = [[float(p[0]), float(p[1])] for p in gps_points]
        folium.PolyLine(
            locations=gps_latlon,
            color="#1f77b4",
            weight=1.5,
            opacity=0.35,
            dash_array="4, 4",
        ).add_to(gps_layer)

        # Subsample circle markers if trip is very long so browser remains responsive
        stride = max(1, len(gps_latlon) // 600)
        for idx in range(0, len(gps_latlon), stride):
            lat, lon = gps_latlon[idx]
            folium.CircleMarker(
                location=[lat, lon],
                radius=2.8,
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.8,
                popup=f"GPS Obs #{idx} ({lat:.5f}, {lon:.5f})",
            ).add_to(gps_layer)

        # Start / End pins
        folium.Marker(
            location=gps_latlon[0],
            popup=f"Trip {trip_id} — Start",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(gps_layer)
        folium.Marker(
            location=gps_latlon[-1],
            popup=f"Trip {trip_id} — End",
            icon=folium.Icon(color="darkred", icon="flag-checkered", prefix="fa"),
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: rgba(255,255,255,0.95); padding: 10px 14px;
                border-radius: 8px; border: 1px solid #d0d7de;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
      <div style="font-weight: 700; margin-bottom: 5px;">PathSnap — Trip {trip_id}</div>
      <div><span style="color:#1f77b4; font-weight:bold;">●</span> Original Raw GPS Trajectory</div>
      <div><span style="color:#d62728; font-weight:bold;">━━</span> Predicted Road Route ({model_label})</div>
      <div style="margin-top:4px; color:#57606a; font-size:11px;">Ground truth stays server-side (Laptop 1)</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    if lats and lons:
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    return fmap
