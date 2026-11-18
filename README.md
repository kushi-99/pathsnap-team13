# PathSnap Team 13 - data foundation

Kushi's Person 1 work covers trip loading, EDA, candidate-road generation, shared features, route-label alignment, and Logistic Regression/KNN baselines.

## Two-laptop demo

The backend runs on the laptop holding the dataset and model files. The browser UI can run on a second laptop on the same trusted Wi-Fi/LAN; only that laptop needs the small `web/` folder and Python's built-in HTTP server. For the final demo, the backend laptop must have the evaluator trips under `data/evaluator_test/` and both trained models under `results/baseline_models/`.

Install dependencies on the backend laptop and extract the provided `Student folder.zip` under `data/`. In PowerShell from the repository root, find the backend laptop's LAN IPv4 address with `ipconfig`, then start the API:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
$env:PATHSNAP_CORS_ORIGINS = "http://FRONTEND-LAN-IP:8501"
python -m pathsnap.webapp --host 0.0.0.0 --port 8000
```

Replace `FRONTEND-LAN-IP` with the frontend laptop's IPv4 address. Allow inbound TCP ports `8000` and `8501` on the **Private** network in Windows Firewall if prompted. Keep both laptops on the same trusted network; the demo API has no authentication.

On the frontend laptop, serve the static UI from the repository root:

```powershell
python -m http.server 8501 --bind 0.0.0.0 --directory web
```

Open `http://FRONTEND-LAN-IP:8501/?api=http://BACKEND-LAN-IP:8000` in a browser, replacing both addresses. The backend URL is saved in that browser; it can also be entered in the **Backend address** field. Use **Logistic Regression** for the live demo; KNN inference can be substantially slower. With internet access, the UI shows an interactive Leaflet map and road tiles; if the CDN is unavailable, an offline GPS/road plot and observation scores are still displayed.

For a single-laptop demo, open `http://localhost:8501` after starting the static server and enter `http://127.0.0.1:8000` as the backend address.

An alternative Streamlit UI is available: on the frontend laptop set `$env:PATHSNAP_API_URL = "http://BACKEND-LAN-IP:8000"` and run `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`.

FastAPI defaults to the final `evaluator_test` split. It provides `GET /health`, `GET /trips?split=evaluator_test`, and `GET /score?split=evaluator_test&trip_id=00000015&model=logistic_regression`. Evaluator results include route-alignment Top-1 and candidate coverage; these use the ordered `.route` reference and are not independently verified pointwise ground truth.

## Data

Extract `Student folder.zip` locally under `data/`. The code supports either `data/Student folder/train` or `data/train`. Raw data is ignored by Git.

The final evaluation set is separate at `data/evaluator_test/`. Each numeric trip folder must contain matching `.track`, `.nodes`, `.arcs`, and `.route` files. Generate the all-trip final comparison (without training or student-test data) with:

```powershell
python -m pathsnap.evaluate --data-root data --model-dir results/baseline_models --out results/evaluator_test_metrics.csv
```

The evaluator report contains per-trip and aggregate metrics for both saved models.

Each trip contains whitespace-separated files without headers: `.track` (longitude, latitude, timestamp seconds), `.nodes` (longitude, latitude), `.arcs` (from-node and to-node row indices; row number is `arc_id`), and `.route` (ordered reference arc IDs).

## Pipeline files

- `src/pathsnap/data_loader.py`: reads the four trip files.
- `src/pathsnap/eda.py`: writes per-trip data-quality and inventory summaries.
- `src/pathsnap/candidates.py`: generates nearby road candidates.
- `src/pathsnap/features.py`: adds distance, heading, speed, time-gap, and road-degree features.
- `src/pathsnap/labels.py`: aligns GPS points to ordered route arcs using a forward-only dynamic program.
- `src/pathsnap/prepare_features.py`: writes candidate-level feature/label CSVs.
- `src/pathsnap/train_baselines.py`: trains Logistic Regression and KNN using a trip-grouped holdout.

## Current local outputs

Feature tables and analysis outputs are in `results/`. The original tables are retained; the revised label tables end in `_aligned.csv`. `baseline_metrics.csv` contains the initial group-holdout baseline results. Model files are saved under `results/baseline_models/`.

## Review 2 data-foundation workflow (Windows PowerShell)

Run these commands from the repository root after activating `.venv` (or use the explicit interpreter below). The dataset remains local under `data/` and is not part of the repository.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

To rebuild per-trip inventories from raw trip files:

```powershell
python -m pathsnap.eda --data-root data --split train --out results/eda_inventory.csv
python -m pathsnap.eda --data-root data --split student_test --out results/eda_student_test_inventory.csv
```

To regenerate the final aligned tables from raw trips (these commands replace the existing aligned outputs):

```powershell
python -m pathsnap.prepare_features --data-root data --split train --out results/train_features_aligned.csv
python -m pathsnap.prepare_features --data-root data --split student_test --out results/student_test_features_aligned.csv
```

The existing `_aligned.csv` tables contain forward-only route-alignment labels and are the inputs used for current Review 2 outputs. To tune Logistic Regression regularization and KNN neighbor/weight settings on a nested trip-grouped holdout, then evaluate the selected models on the untouched outer holdout:

```powershell
python -m pathsnap.train_baselines --features results/train_features_aligned.csv --out-dir results/baseline_models
```

The inner tuning results are saved to `results/hyperparameter_tuning.csv`; the outer holdout metrics and selected parameters are saved to `results/baseline_metrics.csv`. KNN tuning and the final saved KNN model use reproducible balanced training samples capped at 25,000 rows; tuning uses up to 250 validation observations by default. Student-test data is not used for tuning.

Validate raw trips and aligned feature tables first. This uses chunked CSV reads and does not rewrite the large feature tables:

```powershell
python -m pathsnap.validate_data --data-root data --train-features results/train_features_aligned.csv --student-features results/student_test_features_aligned.csv
```

After validation, generate the EDA figures, maps, and report with their focused modules:

```powershell
python -m pathsnap.eda --data-root data --split train --out results/eda_inventory.csv --make-figures
python -m pathsnap.maps --data-root data --features results/train_features_aligned.csv --trip-id 00000001 --output-dir results/maps
python -m pathsnap.report --results-dir results
```

Deliverables are written to `results/dataset_validation_report.csv`, `results/feature_validation_report.csv`, `results/feature_table_validation_summary.csv`, `results/candidate_trip_validation_*.csv`, `results/figures/`, `results/maps/`, `results/data_foundation_report.md`, and `results/kushi_review2_executive_summary.md`. The map builder uses Folium when available and falls back to its interactive Leaflet renderer otherwise. Maps use real local trip data; OpenStreetMap tiles require internet access when viewed.

## Label-quality note

The route alignment is inferred from geometry because `.route` does not provide a label per GPS observation. Review `results/label_alignment_quality.csv`: training trips `00000054` and `00000033` have unusually large GPS-to-route distances and should be disclosed as data-quality limitations. Do not describe the labels as verified ground truth.
