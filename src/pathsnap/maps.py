"""Build interactive trajectory, candidate, and road-network maps."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_trip


def _arc_coordinates(nodes, arcs, arc_ids):
    endpoints = arcs.set_index("arc_id")[["from_node_id", "to_node_id"]]
    lines = []
    for arc_id in arc_ids:
        if int(arc_id) not in endpoints.index:
            continue
        start, end = endpoints.loc[int(arc_id)].to_numpy(dtype=int)
        if 0 <= start < len(nodes) and 0 <= end < len(nodes):
            lines.append([
                [float(nodes.iloc[start].latitude), float(nodes.iloc[start].longitude)],
                [float(nodes.iloc[end].latitude), float(nodes.iloc[end].longitude)],
            ])
    return lines


def _nearby_arc_ids(track, nodes, arcs, limit):
    latitude = track.latitude.to_numpy()
    longitude = track.longitude.to_numpy()
    lat_margin = max(float(latitude.max() - latitude.min()) * 0.08, 0.002)
    lon_margin = max(float(longitude.max() - longitude.min()) * 0.08, 0.002)
    valid = arcs[
        arcs.from_node_id.between(0, len(nodes) - 1)
        & arcs.to_node_id.between(0, len(nodes) - 1)
    ]
    start = valid.from_node_id.to_numpy(dtype=int)
    node_lat = nodes.latitude.to_numpy()
    node_lon = nodes.longitude.to_numpy()
    keep = (
        (node_lat[start] >= latitude.min() - lat_margin)
        & (node_lat[start] <= latitude.max() + lat_margin)
        & (node_lon[start] >= longitude.min() - lon_margin)
        & (node_lon[start] <= longitude.max() + lon_margin)
    )
    ids = valid.loc[keep, "arc_id"].to_numpy(dtype=int)
    if len(ids) > limit:
        ids = ids[np.linspace(0, len(ids) - 1, limit).astype(int)]
    return ids


def _candidate_rows_by_trip(feature_path, trip_ids):
    wanted = {str(trip_id) for trip_id in trip_ids}
    selected = {trip_id: [] for trip_id in wanted}
    columns = ["trip_id", "observation_index", "candidate_arc_id", "reference_arc_id", "label"]
    for chunk in pd.read_csv(feature_path, dtype={"trip_id": str}, usecols=columns, chunksize=100_000):
        matches = chunk.loc[chunk.trip_id.isin(wanted), columns]
        for trip_id, rows in matches.groupby("trip_id", sort=False):
            selected[str(trip_id)].append(rows)
    return {trip_id: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
            for trip_id, parts in selected.items()}


def _write_leaflet(path, title, center, gps, roads, route, candidates, show_gps=True):
    payload = json.dumps({"gps": gps if show_gps else [], "roads": roads, "route": route, "candidates": candidates}, separators=(",", ":"))
    html = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{{margin:0;font:14px Arial}}#title{{padding:10px 14px;background:#172b4d;color:white}}#map{{height:calc(100vh - 42px)}}.legend{{background:white;padding:8px;line-height:1.6}}</style></head><body><div id="title">{title} — inferred labels are not verified ground truth</div><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>const D={payload};const m=L.map('map').setView([{center[0]},{center[1]}],14);L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}',{{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}}).addTo(m);let layers={{}};
layers['Road network']=L.layerGroup(D.roads.map(x=>L.polyline(x,{{color:'#778899',weight:2,opacity:.6}}))).addTo(m);
if(D.gps.length){{layers['GPS trajectory']=L.polyline(D.gps.map(p=>[p[0],p[1]]),{{color:'#1769aa',weight:4}}).addTo(m);D.gps.forEach((p,i)=>L.circleMarker([p[0],p[1]],{{radius:3,color:'#064789'}}).bindPopup('GPS observation '+i+'<br>timestamp '+p[2]+' s').addTo(m));let a=D.gps[0],b=D.gps[D.gps.length-1];L.circleMarker([a[0],a[1]],{{radius:7,color:'#16803c'}}).bindPopup('Start').addTo(m);L.circleMarker([b[0],b[1]],{{radius:7,color:'#b42318'}}).bindPopup('End').addTo(m);}}
if(D.route.length)layers['Ordered reference route']=L.layerGroup(D.route.map(x=>L.polyline(x,{{color:'#e63946',weight:5,opacity:.85}}))).addTo(m);
if(D.candidates.length)layers['Candidates at sampled observations']=L.layerGroup(D.candidates.map(x=>L.polyline(x.xy,{{color:x.positive?'#18a558':'#f4a261',weight:x.positive?5:3,dashArray:x.positive?'':'5 5'}}).bindPopup(x.text))).addTo(m);
L.control.layers(null,layers,{{collapsed:false}}).addTo(m);L.control.scale().addTo(m);if(D.gps.length)m.fitBounds(L.latLngBounds(D.gps.map(p=>[p[0],p[1]])).pad(.12));</script></body></html>'''
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")


def _write_folium(path, title, center, gps, roads, route, candidates, show_gps=True):
    import folium
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True)
    fmap.get_root().header.add_child(folium.Element(f"<title>{title}</title>"))
    if roads:
        road_layer = folium.FeatureGroup(name="Nearby road network")
        folium.PolyLine(roads, color="#778899", weight=2, opacity=.6).add_to(road_layer)
        road_layer.add_to(fmap)
    if show_gps:
        gps_layer = folium.FeatureGroup(name="GPS trajectory", show=True)
        folium.PolyLine([[p[0], p[1]] for p in gps], color="#1769aa", weight=4).add_to(gps_layer)
        for index, point in enumerate(gps):
            folium.CircleMarker([point[0], point[1]], radius=3, tooltip=f"Observation {index}; timestamp {point[2]} s").add_to(gps_layer)
        if gps:
            folium.Marker(gps[0][:2], tooltip="Start", icon=folium.Icon(color="green")).add_to(gps_layer)
            folium.Marker(gps[-1][:2], tooltip="End", icon=folium.Icon(color="red")).add_to(gps_layer)
        gps_layer.add_to(fmap)
    if route:
        route_layer = folium.FeatureGroup(name="Ordered reference route", show=True)
        folium.PolyLine(route, color="#e63946", weight=5, opacity=.85).add_to(route_layer)
        route_layer.add_to(fmap)
    if candidates:
        candidate_layer = folium.FeatureGroup(name="Candidates at sampled observations", show=True)
        for candidate in candidates:
            folium.PolyLine(
                candidate["xy"],
                color="#18a558" if candidate["positive"] else "#f4a261",
                weight=4 if candidate["positive"] else 3,
                dash_array=None if candidate["positive"] else "5 5",
                popup=candidate["text"],
            ).add_to(candidate_layer)
        candidate_layer.add_to(fmap)
    bounds = [point[:2] for point in gps] if show_gps else [point for line in roads for point in line]
    if bounds:
        fmap.fit_bounds(bounds, padding=(20, 20), max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(path))


def build_trip_maps(data_root, feature_path, trip_id, output_dir, network_limit=1500):
    candidate_data = _candidate_rows_by_trip(feature_path, {trip_id, "00000054", "00000033"})
    trip = load_trip(data_root, trip_id, "train")
    track, nodes, arcs, route = (trip[key] for key in ("track", "nodes", "arcs", "route"))
    gps = [[float(row.latitude), float(row.longitude), float(row.timestamp_s)] for row in track.itertuples()]
    center = [float(track.latitude.mean()), float(track.longitude.mean())]
    network_ids = _nearby_arc_ids(track, nodes, arcs, network_limit)
    road_lines = _arc_coordinates(nodes, arcs, network_ids)
    route_lines = _arc_coordinates(nodes, arcs, route.to_numpy(dtype=int))
    rows = candidate_data[str(trip_id)]
    candidates = []
    if not rows.empty:
        observations = sorted(rows.observation_index.unique())
        selected_observations = set(observations[::max(1, len(observations) // 3)][:3])
        for row in rows[rows.observation_index.isin(selected_observations)].itertuples():
            line = _arc_coordinates(nodes, arcs, [int(row.candidate_arc_id)])
            if line:
                candidates.append({
                    "xy": line[0], "positive": int(row.label) == 1,
                    "text": f"Trip {trip_id}; observation {row.observation_index}; arc {row.candidate_arc_id}; label {row.label} (inferred)",
                })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = [
        ("gps_trajectory.html", "GPS trajectory", gps, [], [], [], True),
        ("road_network.html", "Nearby road network", [], road_lines, [], [], False),
        ("gps_network_overlay.html", "GPS trajectory over road network", gps, road_lines, route_lines, [], True),
        ("candidate_roads.html", "Candidate roads at sampled GPS observations", gps, road_lines, route_lines, candidates, True),
    ]
    for filename, title, points, roads, reference, candidate_lines, include_gps in maps:
        path = output_dir / filename
        try:
            _write_folium(path, title, center, points, roads, reference, candidate_lines, include_gps)
        except ImportError:
            _write_leaflet(path, title, center, points, roads, reference, candidate_lines, include_gps)
    for problematic_id in ("00000054", "00000033"):
        problem = load_trip(data_root, problematic_id, "train")
        problem_track, problem_nodes, problem_arcs, problem_route = (problem[key] for key in ("track", "nodes", "arcs", "route"))
        problem_gps = [[float(row.latitude), float(row.longitude), float(row.timestamp_s)] for row in problem_track.itertuples()]
        problem_center = [float(problem_track.latitude.mean()), float(problem_track.longitude.mean())]
        problem_road_ids = _nearby_arc_ids(problem_track, problem_nodes, problem_arcs, network_limit)
        problem_roads = _arc_coordinates(problem_nodes, problem_arcs, problem_road_ids)
        problem_route_lines = _arc_coordinates(problem_nodes, problem_arcs, problem_route.to_numpy(dtype=int))
        problem_rows = candidate_data[problematic_id]
        problem_candidates = []
        if not problem_rows.empty:
            observations = sorted(problem_rows.observation_index.unique())
            selected = set(observations[::max(1, len(observations) // 3)][:3])
            for row in problem_rows[problem_rows.observation_index.isin(selected)].itertuples():
                line = _arc_coordinates(problem_nodes, problem_arcs, [int(row.candidate_arc_id)])
                if line:
                    problem_candidates.append({"xy": line[0], "positive": int(row.label) == 1,
                                               "text": f"Trip {problematic_id}; observation {row.observation_index}; label {row.label} (inferred)"})
        path = output_dir / f"problematic_trip_{problematic_id}.html"
        try:
            _write_folium(path, f"Problematic trip {problematic_id}", problem_center, problem_gps, problem_roads, problem_route_lines, problem_candidates)
        except ImportError:
            _write_leaflet(path, f"Problematic trip {problematic_id}", problem_center, problem_gps, problem_roads, problem_route_lines, problem_candidates)
    print(f"Maps saved under {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--features", default="results/train_features_aligned.csv")
    parser.add_argument("--trip-id", default="00000001")
    parser.add_argument("--output-dir", default="results/maps")
    parser.add_argument("--network-limit", type=int, default=1500)
    args = parser.parse_args()
    build_trip_maps(args.data_root, args.features, args.trip_id, args.output_dir, args.network_limit)


if __name__ == "__main__":
    main()
