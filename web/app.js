const apiInput = document.querySelector("#api-url");
const trip = document.querySelector("#trip");
const model = document.querySelector("#model");
const status = document.querySelector("#status");
const run = document.querySelector("#run");
const stats = document.querySelector("#stats");
const evaluation = document.querySelector("#evaluation");
const params = new URLSearchParams(window.location.search);
const initialApi = params.get("api") || localStorage.getItem("pathsnap-api-url") || "";
apiInput.value = initialApi;

const mapElement = document.querySelector("#map");
let map;
let layer;
if (window.L) {
  map = L.map(mapElement).setView([37.77, -122.42], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);
  layer = L.layerGroup().addTo(map);
}

function apiBase() {
  const value = apiInput.value.trim().replace(/\/+$/, "");
  if (!value) {
    throw new Error("Enter the backend address (for example, http://192.168.1.10:8000).");
  }
  try {
    return new URL(value).origin;
  } catch {
    throw new Error("Backend address must include http:// or https://.");
  }
}

async function apiGet(path, query = {}) {
  const url = new URL(path, `${apiBase()}/`);
  url.search = new URLSearchParams(query);
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Backend returned HTTP ${response.status}.`);
  }
  return body;
}

function renderOfflineMap(data) {
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", "0 0 1000 800");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "GPS trace and selected road candidates");

  const allPoints = [
    ...data.gps,
    ...data.selected_roads.flat(),
  ];
  const meanLatitude = data.gps.reduce((sum, point) => sum + point[0], 0) /
    data.gps.length;
  const longitudeScale = Math.cos((meanLatitude * Math.PI) / 180);
  const projected = allPoints.map(([latitude, longitude]) => [
    longitude * longitudeScale,
    latitude,
  ]);
  const bounds = projected.reduce((range, [x, y]) => ({
    minX: Math.min(range.minX, x),
    maxX: Math.max(range.maxX, x),
    minY: Math.min(range.minY, y),
    maxY: Math.max(range.maxY, y),
  }), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
  const { minX, maxX, minY, maxY } = bounds;
  const rangeX = Math.max(maxX - minX, 1e-8);
  const rangeY = Math.max(maxY - minY, 1e-8);
  const scale = Math.min(920 / rangeX, 720 / rangeY);
  const offsetX = (1000 - rangeX * scale) / 2;
  const offsetY = (800 - rangeY * scale) / 2;
  const xy = ([latitude, longitude]) => [
    offsetX + (longitude * longitudeScale - minX) * scale,
    800 - offsetY - (latitude - minY) * scale,
  ];

  const background = document.createElementNS(namespace, "rect");
  background.setAttribute("width", "1000");
  background.setAttribute("height", "800");
  background.setAttribute("fill", "#eef2ef");
  svg.append(background);

  const trace = document.createElementNS(namespace, "polyline");
  trace.setAttribute("points", data.gps.map((point) => xy(point).join(",")).join(" "));
  trace.setAttribute("fill", "none");
  trace.setAttribute("stroke", "#1683a0");
  trace.setAttribute("stroke-width", "4");
  trace.setAttribute("stroke-linecap", "round");
  trace.setAttribute("stroke-linejoin", "round");
  svg.append(trace);

  const markerStride = Math.max(1, Math.ceil(data.observations.length / 250));
  data.observations.forEach((point, index) => {
    if (index % markerStride !== 0 && index !== data.observations.length - 1) return;
    const marker = document.createElementNS(namespace, "circle");
    const [x, y] = xy([point.lat, point.lon]);
    marker.setAttribute("cx", x);
    marker.setAttribute("cy", y);
    marker.setAttribute("r", "4");
    marker.setAttribute("fill", "#17677b");
    marker.setAttribute("stroke", "white");
    marker.setAttribute("stroke-width", "1.5");
    const title = document.createElementNS(namespace, "title");
    title.textContent =
      `Observation ${point.index}: road ${point.candidate_arc_id}, ` +
      `score ${point.score.toFixed(3)}, ` +
      `distance ${point.distance_to_road_m.toFixed(1)} m`;
    marker.append(title);
    svg.append(marker);
  });

  data.selected_roads.forEach((road) => {
    const path = document.createElementNS(namespace, "polyline");
    path.setAttribute("points", road.map((point) => xy(point).join(",")).join(" "));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#e05c22");
    path.setAttribute("stroke-width", "7");
    path.setAttribute("stroke-dasharray", "8 5");
    path.setAttribute("stroke-opacity", "1");
    svg.append(path);
  });

  const legend = document.createElementNS(namespace, "g");
  const panel = document.createElementNS(namespace, "rect");
  panel.setAttribute("x", "24");
  panel.setAttribute("y", "24");
  panel.setAttribute("width", "220");
  panel.setAttribute("height", "78");
  panel.setAttribute("rx", "10");
  panel.setAttribute("fill", "white");
  panel.setAttribute("fill-opacity", "0.92");
  legend.append(panel);
  [["#1683a0", "GPS trace"], ["#e05c22", "Selected roads"]]
    .forEach(([color, label], index) => {
      const line = document.createElementNS(namespace, "line");
      line.setAttribute("x1", "40");
      line.setAttribute("x2", "72");
      line.setAttribute("y1", String(50 + index * 28));
      line.setAttribute("y2", String(50 + index * 28));
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", "5");
      legend.append(line);
      const text = document.createElementNS(namespace, "text");
      text.setAttribute("x", "84");
      text.setAttribute("y", String(55 + index * 28));
      text.setAttribute("fill", "#17252b");
      text.setAttribute("font-size", "16");
      text.textContent = label;
      legend.append(text);
    });
  svg.append(legend);

  mapElement.replaceChildren(svg);
  mapElement.classList.add("offline-map");
}

async function loadTrips() {
  try {
    localStorage.setItem("pathsnap-api-url", apiBase());
    status.textContent = "Loading trips…";
    const data = await apiGet("/trips", { split: "evaluator_test" });
    trip.replaceChildren(...data.trips.map((id) => new Option(id, id)));
    run.disabled = data.trips.length === 0;
    status.textContent = `${data.trips.length} trips available.`;
  } catch (error) {
    trip.replaceChildren();
    run.disabled = true;
    status.textContent = error.message;
  }
}

apiInput.addEventListener("change", loadTrips);
apiInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadTrips();
});
run.addEventListener("click", async () => {
  run.disabled = true;
  status.textContent = "Generating candidates and scoring the trip…";
  try {
    const data = await apiGet("/score", {
      split: "evaluator_test",
      trip_id: trip.value,
      model: model.value,
    });
    if (map) {
      layer.clearLayers();
      const gps = L.polyline(data.gps, { color: "#1683a0", weight: 3 }).addTo(layer);
      data.selected_roads.forEach((road) => {
        L.polyline(road, { color: "#e58146", weight: 4 }).addTo(layer);
      });
      data.observations.forEach((point) => {
        L.circleMarker([point.lat, point.lon], { radius: 3, color: "#17677b" })
          .bindPopup(
            `Observation ${point.index}<br>Road arc ${point.candidate_arc_id}` +
            `<br>Model score ${point.score.toFixed(3)}` +
            `<br>Distance ${point.distance_to_road_m.toFixed(1)} m`,
          )
          .addTo(layer);
      });
      map.fitBounds(gps.getBounds().pad(0.12));
    } else {
      renderOfflineMap(data);
    }
    const scores = data.observations.map((point) => point.score);
    const meanScore = scores.reduce((sum, score) => sum + score, 0) /
      scores.length;
    stats.textContent =
      `${data.observations.length.toLocaleString()} GPS observations · ` +
      `${data.candidate_count.toLocaleString()} candidates · ` +
      `mean selected score ${meanScore.toFixed(3)}`;
    evaluation.textContent = "";
    if (data.evaluation) {
      evaluation.textContent =
        `Evaluator route-alignment Top-1: ${(100 * data.evaluation.top1_accuracy_all).toFixed(1)}% · ` +
        `candidate coverage: ${(100 * data.evaluation.candidate_coverage).toFixed(1)}%`;
    }
    const rows = document.querySelector("#rows");
    rows.replaceChildren();
    const sampleNote = document.createElement("p");
    sampleNote.textContent =
      `First 10 of ${data.observations.length.toLocaleString()} observations`;
    rows.append(sampleNote);
    const table = document.createElement("table");
    table.innerHTML =
      "<thead><tr><th>Observation</th><th>Road arc</th><th>Score</th>" +
      "<th>Distance (m)</th></tr></thead>";
    const body = document.createElement("tbody");
    data.observations.slice(0, 10).forEach((point) => {
      const row = body.insertRow();
      [point.index, point.candidate_arc_id, point.score.toFixed(3),
        point.distance_to_road_m.toFixed(1)].forEach((value) => {
        row.insertCell().textContent = value;
      });
    });
    table.append(body);
    rows.append(table);
    status.textContent = map
      ? "Scoring complete. GPS is blue and the selected roads are orange."
      : "Scoring complete. GPS (blue), selected roads (orange), and scores are shown.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    run.disabled = false;
  }
});

if (initialApi) loadTrips();