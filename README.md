# PathSnap Team 13 - data foundation

Kushi's Person 1 work covers trip loading, EDA, candidate-road generation, shared features, route-label alignment, and Logistic Regression/KNN baselines.

## Data

Extract `Student folder.zip` locally under `data/`. The code supports either `data/Student folder/train` or `data/train`. Raw data is ignored by Git.

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

The existing `_aligned.csv` tables contain forward-only route-alignment labels and are the inputs used for current Review 2 outputs. To train the Logistic Regression and KNN baselines and write their artifacts/metrics:

```powershell
python -m pathsnap.train_baselines --features results/train_features_aligned.csv --out-dir results/baseline_models
```

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
