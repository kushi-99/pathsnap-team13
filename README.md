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

## Label-quality note

The route alignment is inferred from geometry because `.route` does not provide a label per GPS observation. Review `results/label_alignment_quality.csv`: training trips `00000054` and `00000033` have unusually large GPS-to-route distances and should be disclosed as data-quality limitations. Do not describe the labels as verified ground truth.
