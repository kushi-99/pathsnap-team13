# PathSnap Hackathon — Person 4 Work Plan (Kruthik Reddy)

**Role:** ML Engineer — Person 4  
**Core Ownership:** MLP Baseline Implementation, Model Evaluation & Reporting, FastAPI Model-Integration Specification/Interface, and Evaluator-Facing UI Preparation  
**Workspace Root:** `c:\Users\Admin\Downloads\ML HACKATHON`

---

## 1. Existing Project Structure

A complete recursive inspection of the workspace revealed **no source code, notebooks, scripts, trained models, configuration files, or Git repository yet**. Only the official hackathon documentation and raw student dataset partitions are present:

```text
c:\Users\Admin\Downloads\ML HACKATHON\
├── PathSnap_student_materials\
│   ├── Assessment and Plagiarism Policy.pdf
│   ├── Dataset Exploration.pdf
│   ├── Deliverables.pdf
│   ├── Main Guidelines.pdf
│   └── Models and Training Guidelines.pdf
└── Student folder\
    └── Student folder\                  # Note: nested directory from extraction
        ├── train\                       # 80 trip folders (00000001 .. 00000099, subset)
        │   └── <trip_id>\
        │       ├── <trip_id>.track
        │       ├── <trip_id>.nodes
        │       ├── <trip_id>.arcs
        │       └── <trip_id>.route
        └── student_test\                # 10 trip folders
            ├── 00000000\
            ├── 00000007\
            ├── 00000009\
            ├── 00000040\
            ├── 00000052\
            ├── 00000056\
            ├── 00000065\
            ├── 00000076\
            ├── 00000077\
            └── 00000091\
```

*(The 10 reserved `evaluator_test` trips are not present in the student workspace and remain strictly untouched/server-side for final evaluation.)*

---

## 2. Verified Dataset & Environment Findings

### 2.1 Dataset Inventory & Integrity (Verified on Raw Files)

| Partition | Trips | Missing Files | Bad / NaN Rows | `dt <= 0` | Out-of-Bounds Node/Arc IDs | Total GPS Observations | Obs / Trip (min / med / max) | Total Route Arcs | Route Len / Trip (min / med / max) | Nodes / Trip (min / med / max) | Arcs / Trip (min / med / max) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `train` | 80 | 0 | 0 | 0 | 0 | 208,417 | 361 / 1,529 / 19,313 | 31,336 | 59 / 280 / 2,266 | 299 / 9,608 / 155,003 | 649 / 19,527 / 307,392 |
| `student_test` | 10 | 0 | 0 | 0 | 0 | 19,160 | 262 / 1,420 / 3,958 | 3,477 | 95 / 250 / 1,323 | 1,770 / 6,597 / 26,635 | 3,577 / 13,145 / 56,027 |

### 2.2 Critical Domain & Modelling Observations

1. **No 1-to-1 row correspondence between `.track` and `.route`:**
   - `.track` has 3 whitespace-separated headerless columns: `longitude`, `latitude`, `timestamp` (seconds). Nominally 1 Hz (`96.50%` of train steps and `98.62%` of student-test steps have `dt == 1.0 s`).
   - `.nodes` has 2 whitespace-separated headerless columns: `longitude`, `latitude` (0-based row index = `node_id`).
   - `.arcs` has 2 integer columns: `from_node_id`, `to_node_id` (0-based row index = `arc_id`).
   - `.route` has 1 integer column: chronological sequence of traversed `arc_id` values — typically ~5–7× shorter than `.track` because multiple GPS points fall on the same arc (and very short arcs as small as ~2.7 m may be crossed quickly).
2. **Per-trip local `arc_id` namespace:**
   - Each trip has its own road-network subgraph and its own 0-based `arc_id` indexing (`0 .. num_arcs - 1`). Therefore, `arc_id` **cannot** be treated as a global multiclass label across trips.
   - The supervised problem must be formulated as **candidate road-segment scoring (binary classification over `(observation, candidate_arc)` pairs)**: given GPS observation $o_t$ and spatial candidate arc $c \in \mathcal{C}(o_t)$, predict $P(\text{match} \mid \mathbf{x}_{t,c})$. Observation-level Top-1 prediction selects $\arg\max_{c \in \mathcal{C}(o_t)} P(\text{match} \mid \mathbf{x}_{t,c})$, and the per-candidate probabilities directly serve as emission scores for HMM/Viterbi decoding.
3. **High bidirectional arc density (82%–99.7%):**
   - Sample inspection on `train` trips (`00000001`, `00000002`, `00000003`, `00000010`) shows `81.9%–99.7%` of directed arcs `(u, v)` also have their reverse `(v, u)` in `.arcs`.
   - Distance from a GPS point to `(u, v)` and `(v, u)` is identical; therefore **heading/bearing alignment** between vehicle movement and arc orientation is essential to disambiguate travel direction.
4. **Global geographical coverage:**
   - Training coordinates span `lon [-100.42, +137.24]`, `lat [-34.64, +61.70]` (multiple continents/cities).
   - A single fixed UTM zone is invalid. Spatial computations must use **per-trip local tangent-plane projection (equirectangular/azimuthal centered at each trip's centroid)** or geodesic Haversine/great-circle calculations so distances (meters) and bearings (degrees) are physically accurate everywhere.
5. **GPS noise, gaps, and anomalies:**
   - `train` has a maximum time gap of `153,064 s` (~42.5 hours); `student_test` has a maximum gap of `210 s`. Velocity/bearing features must guard against large `dt` jumps.
   - While typical GPS-to-route median distance is ~4.3–5.7 m, Trip `00000002` exhibits severe drift/offset (`p90 = 97.8 m`, `max = 569.5 m`), which will affect Candidate Miss Rate and requires explicit documentation.
6. **High reference route continuity:**
   - `99.96%` of consecutive reference route arcs in `train` (`31,243 / 31,256`) and `99.97%` in `student_test` (`3,466 / 3,467`) satisfy `arc[i].to_node_id == arc[i+1].from_node_id`.

### 2.3 Local Python Environment

- **Active Interpreter:** Python `3.11.8` (`C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe`)
- **Installed Packages:** `numpy 2.3.2`, `pandas 2.3.1`, `scipy 1.17.1`, `scikit-learn 1.9.1`, `matplotlib 3.10.8`, `joblib 1.6.0`, `requests 2.32.5`
- **Not Yet Installed (needed depending on implementation choices):**
  - `torch` (recommended for configurable PyTorch MLP with explicit Dropout, BatchNorm/LayerNorm options, BCEWithLogitsLoss / CrossEntropy, and clean epoch-wise train/val loss & metric logging; `sklearn.neural_network.MLPClassifier` lacks Dropout).
  - `folium` (mandatory per `Deliverables.pdf` Section 6 for interactive geographical maps).
  - `fastapi`, `uvicorn`, `pydantic` (for validating the FastAPI integration contract / mock server testing).
  - `streamlit` + `streamlit-folium` (or HTML/JS + Leaflet/Folium served statically) for the Evaluator UI.

---

## 3. Existing Implementations

**None found in the workspace.**  
Specifically, the following upstream/sibling components are currently absent from this workspace:
- Data-loading, preprocessing, and EDA scripts
- Spatial candidate-generation module (`observation -> candidate arc_ids`)
- Observation-to-reference-route label alignment (`(observation, candidate_arc) -> binary label y`)
- Shared candidate-level feature engineering pipeline
- Other team models (at least 4 other models across distinct algorithm families are required for the team's 5-model comparison)
- HMM / Viterbi sequential decoder
- FastAPI backend implementation
- Frontend / UI scaffold

---

## 4. Person 4 Responsibilities

Per the hackathon role assignment and PES University guidelines (`Models and Training Guidelines.pdf`, `Deliverables.pdf`):

1. **MLP Baseline Model (`ml/mlp/`):**
   - Implement a clean, configurable, reproducible Multi-Layer Perceptron binary candidate classifier (e.g., `128 -> 64 [-> 32]`, ReLU, Dropout, Adam, BCE/Cross-Entropy loss, calibrated/well-behaved probabilities, fixed seeds).
   - Enforce strict **trip-aware / group-aware cross-validation** (`GroupKFold` / `StratifiedGroupKFold` grouped by `trip_id` on the 80 `train` trips) so observations from the same trip never appear in both train and validation folds.
   - Fit all preprocessing/scaling (`StandardScaler`, imputation/clipping) strictly on training folds only.
   - Save reproducible artifacts: trained weights, scaler, feature column list, model/training config, seed, training history, and timing logs.
2. **Rigorous Experimental Evaluation & Reporting (`reports/MLP_EVALUATION_REPORT.md`):**
   - Compute actual metrics on both **80-trip GroupKFold CV** and **10-trip `student_test`** (never touching `evaluator_test`):
     - Candidate-level: Accuracy, Precision, Recall, Macro F1-score, Binary F1 / PR-AUC / ROC-AUC, Log Loss, Confusion Matrix.
     - Observation-level: Top-1 Accuracy, Candidate Miss Rate.
     - Sequence/Route-level: Classifier-only Route Continuity (percentage of consecutive Top-1 predicted arcs that are identical or topologically connected `u->v`), plus hook for HMM/Viterbi route metrics once integrated.
     - Efficiency: Training time, inference time (per trip & per 1,000 candidates).
   - Generate actual plots: epoch-wise train/val loss curves, train/val F1/Top-1 curves, confusion matrix, Folium map comparing raw GPS vs reference route vs MLP Top-1 predictions on representative `student_test` trips, and markdown summary tables.
   - Answer every compulsory model question from `Deliverables.pdf` Section 3.2 (suitability, strengths/limitations, hyperparameters, loss/objective, student-test performance, overfitting/underfitting evidence).
3. **FastAPI Integration Contract & Inference Module (`api_integration/FASTAPI_MLP_INTEGRATION.md`, `ml/mlp/predict.py`):**
   - Provide a standalone, framework-agnostic `MLPPredictor` inference module accepting candidate-level features and returning candidate `arc_id`s, probabilities, observation-level Top-1 predictions, and observation IDs while preserving candidate-to-arc mapping for HMM/Viterbi.
   - Document model loading, required artifacts, JSON request/response schemas (`sample_request.json`, `sample_response.json`), preprocessing, error handling, dependencies, and proposed endpoint (`POST /predict` or team-agreed route).
   - Do **not** build a competing backend if another teammate owns FastAPI; instead make our predictor drop-in ready.
4. **Evaluator-Facing UI Design & Prototype (`ui/evaluator_dashboard/`):**
   - Build a clean, professional two-laptop-ready UI (Laptop 2 client talking to Laptop 1 FastAPI) displaying:
     - PathSnap title & overview
     - Dynamic dropdown/list of the 10 evaluation trips fetched from the API
     - Run Prediction trigger & loading/progress indicator
     - Interactive Folium/Leaflet map showing raw GPS trajectory and predicted road route (never exposing ground-truth labels!)
     - Server-returned evaluation metrics, route continuity, prediction summary, and API connection/error banner
   - Clearly isolate any optional layout-testing mock mode from real API inference (`--mock` / explicit toggle disabled by default), with zero hardcoded evaluation predictions.

---

## 5. Dependencies on Other Team Members & Compatibility Strategy

Because no other team code is currently in this workspace, Person 4 has four key interface dependencies:

| Dependency | Expected Owner | What Is Needed | Person 4 Mitigation / Compatible Interface |
|---|---|---|---|
| **Candidate Generation & Label Alignment** | Person 1 / Data Pipeline | Shared definition of candidate set $\mathcal{C}(o_t)$ (radius/top-$K$) and monotonic/geometric alignment from `.track` observations to `.route` `arc_id`s | Define a clean, modular `CandidateDataset` interface (Parquet/CSV or Python loader) expecting columns `[trip_id, obs_idx, arc_id, <feature_cols>, label]`. If teammates have not pushed their pipeline yet, implement an isolated, documented reference adapter (`ml/mlp/data_adapter.py`) using spatial KD-tree candidate search + monotonic route alignment so MLP experiments run immediately on real data, and swap in the team's shared feature table without changing `model.py`/`train_mlp.py`/`evaluate_mlp.py`. |
| **Shared Feature Schema** | Team-wide (Persons 1–4) | Exact list and semantics of candidate features so all 5+ models are compared fairly on identical features | Drive all feature selection from `feature_columns.json` in `config.py`. Re-running `train_mlp.py` on a new team feature table requires zero architecture code changes. |
| **HMM / Viterbi Emission Interface** | HMM / Sequence Owner | Expected probability format (`P(match \| obs, arc)` vs normalized softmax over candidates per observation, log-prob clamping) | Output both raw sigmoid probability `match_prob` and per-observation normalized `emission_prob` (plus `log_prob`) keyed by `(obs_idx, arc_id)`. |
| **FastAPI Endpoint Schema** | Backend Owner (Laptop 1) | Final URLs (e.g., `GET /trips`, `POST /predict`, `GET /health`) and response JSON payload for the Evaluator UI | Document clear proposed schemas in `FASTAPI_MLP_INTEGRATION.md`, `sample_request.json`, `sample_response.json`, and centralize API endpoint paths & response parsing in one UI config module (`ui/evaluator_dashboard/api_client.py`) for instant alignment. |

---

## 6. Proposed Implementation Sequence (Verifiable Stages)

| Stage | Objective | Key Verification |
|---|---|---|
| **Stage 1: Environment & Data Interface Confirmation** | Confirm with Kruthik whether teammates already have a candidate/feature dataset to drop in (or if we should build our modular reference `data_adapter.py`) and confirm installing `torch`, `folium`, and UI/API packages (`streamlit`/`streamlit-folium` or HTML/JS, `fastapi`, `pydantic`). | Package import check + verified schema agreement. |
| **Stage 2: Data Adapter & Leakage-Free Splits** | Build/connect candidate generation + observation-to-route alignment + physically grounded spatial/kinematic features (local tangent-plane projection, perpendicular distance, bearing difference, speed/dt handling, candidate rank/relative features) and verify Candidate Miss Rate on `train` (80 trips) and `student_test` (10 trips). | Unit checks: 0 trip overlap in `GroupKFold(n_splits=5)`, verified miss rate, zero NaN/Inf features, no `evaluator_test` access. |
| **Stage 3: Configurable MLP Architecture & Training Pipeline** | Implement `config.py`, `model.py`, `train_mlp.py` (PyTorch MLP with configurable hidden layers e.g. `128-64` and `128-64-32`, ReLU, Dropout, Adam, BCEWithLogitsLoss, optional class weighting, early stopping, `StandardScaler` fit per fold). Run 5-fold trip-grouped CV on the 80 train trips, then train final baseline artifact on all 80 train trips. | Reproducible runs with fixed seed; saved weights, scaler, config, feature list, fold logs, and timing metrics. |
| **Stage 4: Evaluation, Plots, Folium Maps & Report** | Run `evaluate_mlp.py` on 5-fold CV out-of-fold predictions and on the 10 `student_test` trips. Generate actual loss/metric curves, confusion matrix, Folium route comparison maps, and write `reports/MLP_EVALUATION_REPORT.md`. | All numbers trace directly to saved JSON/CSV experiment outputs; zero fabricated results. |
| **Stage 5: FastAPI Inference Module & Integration Docs** | Implement `ml/mlp/predict.py` (`MLPPredictor` class), test end-to-end inference against `sample_request.json`, and author `api_integration/FASTAPI_MLP_INTEGRATION.md` + `sample_response.json`. | Live execution test verifying candidate-to-arc ID preservation, latency, error handling, and HMM compatibility. |
| **Stage 6: Evaluator-Facing UI Prototype** | Implement `ui/evaluator_dashboard/` with dynamic trip listing, Run Prediction workflow, loading state, Folium GPS + predicted route visualization, metrics/continuity display, API error handling, and clearly separated layout-only mock mode. | Verified rendering and API client tests without exposing ground truth. |

*(Note: Bayesian hyperparameter optimisation is intentionally deferred until the team completes its 5-model comparison and selects the winning model family.)*

---

## 7. Files to Be Created or Modified

No existing files need modification (only raw data and PDFs exist). Planned new files:

```text
c:\Users\Admin\Downloads\ML HACKATHON\
├── PERSON4_WORK_PLAN.md                         # (Created) This work plan
├── requirements_person4.txt                     # Pinned dependencies for Person 4 components
├── ml\
│   └── mlp\
│       ├── __init__.py
│       ├── config.py                            # Paths, seeds, feature list, MLP & CV hyperparameters
│       ├── data_adapter.py                      # Modular candidate generation, alignment & feature loader (swappable with team pipeline)
│       ├── model.py                             # Configurable PyTorch MLP architecture + probability calibration helpers
│       ├── train_mlp.py                         # Reproducible GroupKFold CV + full 80-trip training & artifact saving
│       ├── evaluate_mlp.py                      # Metrics (Acc/P/R/Macro-F1/Top-1/LogLoss/MissRate/Continuity/Timing), plots & Folium maps
│       ├── predict.py                           # Clean MLPPredictor inference interface for FastAPI & HMM/Viterbi
│       ├── artifacts\                           # Saved model weights (.pt), scaler (.joblib), config & feature schema (.json)
│       ├── results\                             # Actual CV & student_test metrics (.json/.csv) and training histories
│       └── plots\                               # Generated loss curves, metric plots, confusion matrix, Folium HTML maps
├── api_integration\
│   ├── FASTAPI_MLP_INTEGRATION.md               # Complete FastAPI integration specification & backend guide
│   ├── sample_request.json                      # Valid example candidate-feature payload
│   └── sample_response.json                     # Actual predictor output schema example
├── ui\
│   └── evaluator_dashboard\
│       ├── app.py                               # Evaluator-facing dashboard (Streamlit + Folium, or agreed frontend stack)
│       ├── api_client.py                        # Clean HTTP client for Laptop 1 FastAPI (GET /trips, POST /predict, GET /health)
│       ├── map_renderer.py                      # Folium map builder (raw GPS + predicted route, legend, layer controls)
│       ├── mock_data.py                         # Strictly isolated layout-testing mock provider (off by default)
│       └── README.md                            # Two-laptop run instructions
└── reports\
    └── MLP_EVALUATION_REPORT.md                 # Full evaluation report answering Deliverables.pdf Section 3.2 & 3.3
```
