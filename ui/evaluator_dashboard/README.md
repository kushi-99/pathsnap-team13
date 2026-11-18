# PathSnap Evaluator Dashboard (Laptop 2 Client UI)

This directory contains the evaluator-facing frontend designed for the PathSnap two-laptop deployment architecture.

## Architecture

- **Laptop 1 (Server):** Runs the team's FastAPI backend hosting the trained classifier, HMM/Viterbi route decoder, road-network files, and server-side evaluation against reserved reference routes.
- **Laptop 2 (This UI):** Dynamically retrieves the 10 evaluation trip IDs from Laptop 1 (`GET /trips`), sends the evaluator's selected `trip_id` to `POST /predict`, and renders the raw GPS trajectory, predicted road route (Folium), server-computed metrics, and route continuity.

**Strict Integrity Rules Enforced:**
- Evaluator ground-truth labels are **never** loaded, requested, or rendered in the UI.
- Predictions are **never** hardcoded.
- Layout preview mode (`mock_data.py`) is strictly isolated, disabled by default, and clearly watermarked if toggled on for offline UI checks.

## Running on Laptop 2

```powershell
# Optional: point directly to Laptop 1's LAN IP
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_IP>:8000"

streamlit run ui/evaluator_dashboard/app.py
```

## Expected Laptop 1 API Contract

1. `GET /health` → `{"status": "ok", "model_name": "...", "decoder": "hmm_viterbi"}`
2. `GET /trips` → `{"trips": [{"trip_id": "...", "num_observations": 1200}, ...]}`
3. `POST /predict` with body `{"trip_id": "..."}` → returns:
   - `trip_id`, `model_name`, `decoder`, `num_observations`, `num_predicted_segments`, `inference_time_ms`
   - `gps_points`: `[[lat, lon], ...]`
   - `predicted_segments`: `[[[lat1, lon1], [lat2, lon2]], ...]`
   - `metrics`: `{"obs_top1_accuracy": ..., "macro_f1": ..., "log_loss": ..., "candidate_miss_rate": ...}`
   - `route_continuity`: `{"obs_step_continuity": ..., "compressed_route_continuity": ..., "disconnected_transitions": ...}`

Endpoint paths can be adjusted in one place in `api_client.py` (`APIEndpoints`) to match the backend developer's finalized routes.
