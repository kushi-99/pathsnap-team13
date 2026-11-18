# PathSnap Team 13 - data foundation

Kushi's current work covers trip loading, EDA, candidate-road generation, shared features, provisional route labels, and Logistic Regression/KNN baselines.

## Data

Extract the provided `Student folder.zip` into `data/` locally. The scripts support either `data/Student folder/train` or `data/train`. Raw data is ignored by Git and is not included in this branch.

Each trip contains whitespace-separated files without headers: `.track` (longitude, latitude, timestamp seconds), `.nodes` (longitude, latitude), `.arcs` (from-node and to-node row indices; row number is `arc_id`), and `.route` (ordered reference arc IDs).

## Python setup

Create and activate a Python 3.11 virtual environment, then install `requirements.txt`. Set `PYTHONPATH` to the repository's `src` folder before running the package modules.

## Current pipeline modules

- `data_loader.py`: loads the four trip files.
- `eda.py`: writes per-trip data-quality and inventory summaries.
- `candidates.py`: generates nearby road candidates.
- `features.py`: adds distance, heading, speed, time-gap, and degree features.
- `labels.py`: maps GPS observations to the nearest road in the route sequence.
- `prepare_features.py`: writes candidate-level feature/label CSVs.
- `train_baselines.py`: trains initial Logistic Regression and KNN baselines with a trip-grouped holdout.

The route-to-observation alignment in `labels.py` is provisional and must be validated before its labels or model scores are treated as final. The candidate CSV has one row per GPS observation and candidate arc. The model inputs are listed in `docs/feature_schema.md`; identifiers and labels must not be model inputs.
