# PathSnap Hackathon — Review 2 Complete Repository Implementation Audit

**Repository:** `https://github.com/kushi-99/pathsnap-team13`  
**Local Git Path:** `c:\Users\Admin\Downloads\ML HACKATHON\pathsnap-team13`  
**Active Branch:** `feat/kruthik-reddy-mlp-baseline` (at commit `bb241a3`)  
**Auditor:** Person 4 — Kruthik Reddy  
**Audit Date:** 2026-11-18

---

## 1. Executive Summary

A full line-by-line inspection of the local repository (`feat/kruthik-reddy-mlp-baseline`) and fetched remote branches (`origin/kushi/data-foundation`, `origin/kritika-hgb-hmm`, `origin/main`) shows that the **MLP baseline pipeline and evaluation artifacts are complete and verified**, and the **shared `src/pathsnap/` package modules exist**, but **several Review 2 integration, backend, multi-model comparison, frontend, documentation, and testing deliverables are currently incomplete or missing**.

---

## 2. Detailed Component Audit

### 2.1 Completed & Verified Components

| Component | Location | Verified Status |
|---|---|---|
| **MLP Pipeline & Training Code** | `ml/mlp/` (`config.py`, `data_adapter.py`, `model.py`, `metrics_utils.py`, `train_mlp.py`, `evaluate_mlp.py`, `predict.py`) | Complete PyTorch MLP (`mlp_128_64_32`, 28 features, `14,529` parameters) with 5-fold trip-grouped `GroupKFold` CV, zero-leakage per-fold `StandardScaler`, and `MLPPredictor` inference wrapper. |
| **Trained MLP Model Artifacts** | `ml/mlp/artifacts/` (`mlp_model.pt`, `scaler.joblib`, `feature_columns.json`, `label_mapping.json`, `model_config.json`, `training_metadata.json`) | Present and valid (`mlp_model.pt` = 67.9 KB). Verified exact match to reported metrics. |
| **Finalized MLP Evaluation Results** | `ml/mlp/results/` (`student_test_metrics.json`, `student_test_per_trip.csv`, `performance_summary_table.csv`, `cv_architecture_comparison.csv/.json`, `cv_selected_summary.json`, `prediction_vs_reference_sample.csv`) | Verified against saved files: Student-Test Candidate Acc = **96.88%** (`0.9688`), Macro F1 = **91.86%** (`0.9186`), Obs Top-1 = **86.41%** (`0.8641`, covered `86.56%`), Log Loss = **0.0779**, Candidate Miss Rate = **0.18%** (`34/19,160`), Obs-Step Continuity = **94.17%**, Compressed Continuity = **79.26%**, Inference = **146.5 ms** (`14.65 ms/trip`). |
| **MLP Plots & Static Folium Maps** | `ml/mlp/plots/` (5 PNG charts + 3 HTML Folium maps: `00000052`, `00000056`, `00000077`) | Present and verified. |
| **Shared Data/Feature Modules** | `src/pathsnap/` (`data_loader.py`, `candidates.py`, `features.py`, `labels.py`, `evaluate.py`, `train_baselines.py`) | Present on branch `feat/kruthik-reddy-mlp-baseline`; aligned to the canonical 28-feature schema (`ml/mlp/config.py`). |
| **Feature & API Specs** | `docs/feature_schema.md`, `api_integration/FASTAPI_MLP_INTEGRATION.md`, `sample_request.json`, `sample_response.json`, `reports/MLP_EVALUATION_REPORT.md` | Present and consistent with `mlp_128_64_32`. |

---

### 2.2 Incomplete, Disconnected, or Broken Components

| Component | Location | Issue Identified |
|---|---|---|
| **Dataset Path Resolution (Broken inside git repo)** | `ml/mlp/config.py` (line 11) | Hardcodes `DATA_ROOT = WORKSPACE_ROOT / "Student folder" / "Student folder"`, which resolves to `pathsnap-team13/Student folder/Student folder` (non-existent; actual dataset sits at `../Student folder/Student folder`). Must support `PATHSNAP_DATA_ROOT` env var + safe relative fallback without modifying raw data. |
| **Baseline Models (`src/pathsnap/train_baselines.py`) Not Executed** | `src/pathsnap/train_baselines.py` | Implements `logistic_regression`, `random_forest`, and `hist_gradient_boosting` on the shared 28-feature schema, **but has never been run on this branch** — no `.joblib` model artifacts, no `student_test` evaluation metrics, no ROC curves, and no multi-model comparison files exist yet. |
| **Streamlit Frontend (`ui/evaluator_dashboard/`)** | `ui/evaluator_dashboard/app.py`, `api_client.py`, `map_renderer.py`, `mock_data.py` | Only a **single-page prototype** exists. Missing the 5 required Review 2 pages (Dashboard, Model Comparison, GPS Map Matching with candidate overlays & optional reference toggle, Evaluation with ROC/PR/downloads, System Information), missing multi-model selection, and disconnected from a live FastAPI backend. |
| **Folium Map Renderer (`map_renderer.py`)** | `ui/evaluator_dashboard/map_renderer.py` | Currently renders only raw GPS + predicted segments. Missing optional reference route layer (with graceful handling when absent), candidate road segments layer, selected candidate highlighting, and trip summary overlay. |
| **Unstaged Typo File Deletion** | `docs/feature_schema,md` | Git status shows unstaged deletion of comma-typo file `docs/feature_schema,md` (already replaced by `docs/feature_schema.md`). Needs clean staging/resolution. |
| **`.gitignore` Incomplete** | `.gitignore` | Only ignores `data/`, `.venv/`, `__pycache__/`, `*.py[cod]`. Missing `.env`, `Student folder/`, `ml/mlp/cache/`, `*.pkl` cache exclusions, and temporary files. |

---

### 2.3 Missing Components (Required for Review 2)

| Component | Expected Path | Status |
|---|---|---|
| **FastAPI Backend Server** | `api/` (`main.py`, `schemas.py`, `model_registry.py`, `trip_service.py`) | **Missing.** No FastAPI application exists anywhere in the repository (`GET /health`, `GET /models`, `GET /trips`, `GET /trips/{trip_id}`, `POST /predict`, `POST /evaluate` must be implemented). |
| **Unified Model Registry & Multi-Model Evaluation Results** | `ml/comparison/` or `artifacts/baselines/` + unified comparison JSON/CSV | **Missing.** Need real, apples-to-apples 28-feature evaluation of available baseline models (`mlp_128_64_32`, `hist_gradient_boosting`, `logistic_regression`, `random_forest`) on the exact same 80-train / 10-student-test splits, plus ROC/PR data. |
| **Two-Laptop Network Setup Guide** | `docs/TWO_LAPTOP_SETUP.md` | **Missing.** |
| **Consolidated Review 2 Technical Report (MD + PDF)** | `reports/PATHSNAP_REVIEW2_TECHNICAL_REPORT.md` & `.pdf` | **Missing.** |
| **Repository `README.md` & `requirements.txt`** | `README.md`, `requirements.txt` | **Empty (0 bytes each).** |
| **Automated Integration & API Tests** | `tests/` | **Missing.** No `pytest` test suite exists to verify model loading, FastAPI endpoints, inference, trip loading, map rendering, and offline API error handling. |

---

### 2.4 Audit of Teammates' Remote Branches (Read-Only Inspection)

Read-only inspection of remote tracking refs (`git fetch origin`, without switching away from `feat/kruthik-reddy-mlp-baseline`) found:

1. **`origin/kushi/data-foundation` (Kushi — Data Foundation & Initial 6-Feature Baselines):**
   - Uses a **different 6-feature schema** (`distance_to_road_m`, `heading_difference_deg`, `speed_mps`, `time_gap_s`, `incoming_degree`, `outgoing_degree`) and a **single 64-train / 16-val holdout split** (not 5-fold CV and not evaluated on `student_test`).
   - Committed `results/baseline_metrics.csv` and `.joblib` models for:
     - `logistic_regression` (6-feat, 64/16 val holdout): Covered Cand Acc `0.8647`, Covered Macro F1 `0.6802`, Log Loss `0.2728`, Top-1 All `0.7285` (Covered `0.7458`), Val Miss Rate `2.32%`.
     - `knn` (6-feat, 64/16 val holdout, 100k subsample): Covered Cand Acc `0.8880`, Covered Macro F1 `0.7057`, Log Loss `1.5288`, Top-1 All `0.6357` (Covered `0.6508`).
   - **Comparability Note:** Because these used 6 features on a 64/16 holdout (and were not evaluated on `student_test`), they must be clearly separated from the 28-feature 5-fold CV / `student_test` comparison table so metrics are never conflated across incompatible setups.
2. **`origin/kritika-hgb-hmm` (Kritika — HistGradientBoosting & Tuning Scripts):**
   - Contains `train_hist_gradient_boosting.py` and `tune_hist_gradient_boosting.py` (targeting the 6-feature schema), **but no committed trained `.joblib` model artifacts, no saved evaluation JSON/CSV metrics, and no HMM/Viterbi decoder implementation yet**.
3. **`origin/main`:** Contains only the initial repository scaffold (`248225b`).

---

## 3. Review 2 Implementation Plan (Inside `pathsnap-team13` on `feat/kruthik-reddy-mlp-baseline`)

1. **Phase 2 — Fix Dataset Path Resolution & Finalize Multi-Model Evaluation:**
   - Update `ml/mlp/config.py` with `PATHSNAP_DATA_ROOT` + automatic fallback (`WORKSPACE_ROOT / "Student folder" / "Student folder"` -> `WORKSPACE_ROOT.parent / "Student folder" / "Student folder"` -> `WORKSPACE_ROOT / "data"`).
   - Execute a reproducible multi-model training & evaluation script (`ml/comparison/run_model_comparison.py`) reusing the exact 28-feature dataset (`80 train` trips for 5-fold GroupKFold CV + final fit, `10 student_test` trips for held-out evaluation) for the repo's existing baseline classifiers (`logistic_regression`, `hist_gradient_boosting`, `random_forest`) alongside the already-trained `mlp_128_64_32`, saving real artifacts, confusion matrices, ROC/PR curves, per-trip metrics, route continuity, and inference times — while transparently noting the status of teammates' separate 6-feature branches (`knn`, `kritika-hgb-hmm`, and pending HMM/Viterbi).
2. **Phase 3 & 4 — Complete 5-Page Streamlit Frontend & Enhanced Folium Visualization:**
   - Upgrade `ui/evaluator_dashboard/map_renderer.py` to render actual GPS trajectory, predicted route, optional reference route (with graceful fallback when absent), candidate road segments, selected candidate highlights, markers, trip info box, legend, and layer controls.
   - Upgrade `ui/evaluator_dashboard/api_client.py` and `app.py` into a full 5-page application (Dashboard, Model Comparison, GPS Map Matching, Evaluation, System Information) powered by live FastAPI calls over configurable `PATHSNAP_API_URL`.
3. **Phase 5 — Production Two-Laptop FastAPI Backend & Network Setup Guide:**
   - Implement `api/` (`main.py`, `schemas.py`, `model_registry.py`, `trip_service.py`) with startup singleton model loading, Pydantic validation, CORS, and all required endpoints (`GET /health`, `GET /models`, `GET /trips`, `GET /trips/{trip_id}`, `POST /predict`, `POST /evaluate`).
   - Create `docs/TWO_LAPTOP_SETUP.md`.
4. **Phase 6 — Consolidated Technical Report (Markdown + PDF):**
   - Generate `reports/PATHSNAP_REVIEW2_TECHNICAL_REPORT.md` and `reports/PATHSNAP_REVIEW2_TECHNICAL_REPORT.pdf` (with embedded real architecture diagram and genuine generated evaluation charts).
5. **Phase 7 — Complete `README.md`, `requirements.txt`, and `.gitignore` Hygiene.**
6. **Phase 8 & 9 — Automated `pytest` Suite, Live Server Verification, and Clean Logical Git Commits (without pushing until confirmed).**
