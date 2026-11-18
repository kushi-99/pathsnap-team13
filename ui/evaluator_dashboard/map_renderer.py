"""Interactive Folium geographical route visualization for PathSnap Review 2.

Renders genuine geographic coordinates ([lat, lon]) reconstructed from `.track`,
`.nodes`, `.arcs`, and (when permitted) `.route`:
  1. Candidate road segments (light slate gray, toggleable)
  2. Reference route (emerald green solid line, gracefully omitted if unavailable)
  3. Selected candidate road segments (amber highlight halo)
  4. Predicted matched road route (crimson line)
  5. Original GPS trajectory (royal blue line + interactive observation markers + start/end pins)
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import folium


def build_prediction_map(
    gps_points: Sequence[Sequence[float]],
    predicted_segments: Sequence[Sequence[Sequence[float]]],
    trip_id: str,
    model_label: str = "PathSnap Model",
    reference_segments: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    candidate_segments: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    selected_candidate_segments: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    top1_accuracy: Optional[float] = None,
    step_continuity: Optional[float] = None,
) -> folium.Map:
    """Build a multi-layer interactive Folium map from real trip & road-network coordinates.

    Args:
        gps_points: Sequence of [lat, lon] or [lat, lon, timestamp] GPS coordinates.
        predicted_segments: Sequence of [[lat1, lon1], [lat2, lon2]] predicted road segments.
        trip_id: Trip identifier for title/legend.
        model_label: Display name of the selected model.
        reference_segments: Optional [[lat1, lon1], [lat2, lon2]] reference route segments.
        candidate_segments: Optional [[lat1, lon1], [lat2, lon2]] generated candidate segments.
        selected_candidate_segments: Optional [[lat1, lon1], [lat2, lon2]] selected unique segments.
        top1_accuracy: Optional observation-level Top-1 accuracy for trip info badge.
        step_continuity: Optional observation-step route continuity for trip info badge.
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
    if reference_segments:
        for seg in reference_segments:
            for pt in seg:
                lats.append(float(pt[0]))
                lons.append(float(pt[1]))

    center = [sum(lats) / len(lats), sum(lons) / len(lons)] if lats and lons else [12.9345, 77.5345]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # Layer 1: Generated Candidate Road Segments (road-network context)
    if candidate_segments:
        cand_layer = folium.FeatureGroup(
            name=f"Candidate Road Segments ({len(candidate_segments):,})", show=True
        )
        folium.PolyLine(
            locations=candidate_segments,
            color="#6c757d",
            weight=2.0,
            opacity=0.45,
            tooltip="Generated Candidate Road Segment",
        ).add_to(cand_layer)
        cand_layer.add_to(fmap)

    # Layer 2: Reference Route (when available & permitted)
    has_ref = bool(reference_segments)
    if has_ref:
        ref_layer = folium.FeatureGroup(
            name=f"Reference Route ({len(reference_segments):,} arcs)", show=True
        )
        folium.PolyLine(
            locations=reference_segments,
            color="#2ca02c",
            weight=6.5,
            opacity=0.72,
            tooltip=f"Reference Route — Trip {trip_id}",
        ).add_to(ref_layer)
        ref_layer.add_to(fmap)

    # Layer 3: Selected Candidate Segments (highlighted corridor)
    if selected_candidate_segments:
        sel_layer = folium.FeatureGroup(
            name=f"Selected Candidate Segments ({len(selected_candidate_segments):,})", show=False
        )
        folium.PolyLine(
            locations=selected_candidate_segments,
            color="#ff9900",
            weight=5.5,
            opacity=0.75,
            tooltip="Selected Top-1 Candidate Segment",
        ).add_to(sel_layer)
        sel_layer.add_to(fmap)

    # Layer 4: Predicted Matched Road Route
    pred_layer = folium.FeatureGroup(
        name=f"Predicted Route — {model_label} ({len(predicted_segments):,} segs)", show=True
    )
    if predicted_segments:
        folium.PolyLine(
            locations=predicted_segments,
            color="#d62728",
            weight=3.8,
            opacity=0.92,
            dash_array="8, 4" if has_ref else None,
            tooltip=f"Predicted Matched Route ({model_label})",
        ).add_to(pred_layer)
    pred_layer.add_to(fmap)

    # Layer 5: Original Raw GPS Trajectory + Observation Markers
    gps_layer = folium.FeatureGroup(
        name=f"Original GPS Trajectory ({len(gps_points):,} obs)", show=True
    )
    if gps_points:
        gps_latlon = [[float(p[0]), float(p[1])] for p in gps_points]
        folium.PolyLine(
            locations=gps_latlon,
            color="#1f77b4",
            weight=2.0,
            opacity=0.55,
            tooltip="Original Raw GPS Path",
        ).add_to(gps_layer)

        # Subsample interactive markers for smooth browser performance on long trips
        stride = max(1, len(gps_points) // 450)
        for idx in range(0, len(gps_points), stride):
            pt = gps_points[idx]
            lat, lon = float(pt[0]), float(pt[1])
            ts_str = f" | t={float(pt[2]):.0f}s" if len(pt) >= 3 else ""
            folium.CircleMarker(
                location=[lat, lon],
                radius=2.8,
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.82,
                popup=f"GPS Obs #{idx}{ts_str}<br>({lat:.6f}, {lon:.6f})",
            ).add_to(gps_layer)

        folium.Marker(
            location=gps_latlon[0],
            popup=f"Trip {trip_id} — Start (Obs #0)",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(gps_layer)
        folium.Marker(
            location=gps_latlon[-1],
            popup=f"Trip {trip_id} — End (Obs #{len(gps_latlon) - 1})",
            icon=folium.Icon(color="darkred", icon="flag-checkered", prefix="fa"),
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # Legend & Trip Info Overlay
    metric_lines = ""
    if top1_accuracy is not None:
        metric_lines += f"<div>Obs Top-1 Acc: <b>{top1_accuracy * 100:.2f}%</b></div>"
    if step_continuity is not None:
        metric_lines += f"<div>Route Continuity: <b>{step_continuity * 100:.2f}%</b></div>"

    ref_legend = (
        '<div><span style="color:#2ca02c; font-weight:bold;">━━</span> Reference Route (.route)</div>'
        if has_ref
        else '<div style="color:#6c757d;">Reference Route: Not displayed / server-side</div>'
    )
    cand_legend = (
        f'<div><span style="color:#6c757d; font-weight:bold;">──</span> Candidate Segments ({len(candidate_segments):,})</div>'
        if candidate_segments
        else ""
    )

    legend_html = f"""
    <div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999;
                background: rgba(255,255,255,0.96); padding: 11px 15px;
                border-radius: 8px; border: 1px solid #d0d7de;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 12px; line-height: 1.5;
                box-shadow: 0 2px 10px rgba(0,0,0,0.16); max-width: 290px;">
      <div style="font-weight: 700; font-size: 13px; margin-bottom: 4px; color:#111;">
        PathSnap — Trip {trip_id}
      </div>
      <div style="color:#555; font-size:11px; margin-bottom: 6px;">
        Model: <b>{model_label}</b> | GPS Obs: <b>{len(gps_points):,}</b>
      </div>
      <div><span style="color:#1f77b4; font-weight:bold;">●──</span> Original GPS Trajectory</div>
      <div><span style="color:#d62728; font-weight:bold;">┅┅</span> Predicted Matched Route</div>
      {ref_legend}
      {cand_legend}
      <div><span style="color:#ff9900; font-weight:bold;">━━</span> Selected Candidate Highlight (layer)</div>
      {f'<hr style="margin:6px 0; border:none; border-top:1px solid #e5e7eb;">{metric_lines}' if metric_lines else ''}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    if lats and lons:
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    return fmap
