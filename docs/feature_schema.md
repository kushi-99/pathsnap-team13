# PathSnap – Candidate Feature Schema

## 1. Overview

This document describes the candidate-level feature schema used by the PathSnap machine-learning pipeline.

Each row represents a possible road-segment candidate for a particular GPS observation.

The MLP model uses 28 observable numerical features to estimate whether a candidate road segment matches the actual GPS observation.

## 2. Dataset Structure

Each candidate record contains:

| Column | Type | Description |
|---|---|---|
| trip_id | String | Unique trip identifier |
| obs_idx | Integer | GPS observation index within the trip |
| arc_id | Integer | Original zero-based road-segment index |
| label | Integer | Ground-truth match label (0 or 1), when available |
| Feature columns | Float | 28 numerical input features |

The label is included during supervised training and evaluation. It must not be used as an input feature.

## 3. The 28 Model Features

### A. Distance and Geometry Features

| Feature | Description |
|---|---|
| perp_dist_m | Perpendicular distance between GPS observation and candidate road segment, in metres |
| log_perp_dist | Log-transformed perpendicular distance |
| proj_ratio | Relative position of the projected GPS point along the road segment |
| is_interior_proj | Indicates whether the projection lies inside the road segment |
| dist_to_start_m | Distance from the projected point to the segment start |
| dist_to_end_m | Distance from the projected point to the segment end |
| arc_length_m | Length of the candidate road segment in metres |
| log_arc_length | Log-transformed road-segment length |

### B. Heading and Movement Features

| Feature | Description |
|---|---|
| cos_heading_diff | Cosine of the heading difference between GPS movement and road direction |
| sin_heading_diff_abs | Absolute sine of the heading difference |
| heading_diff_norm | Normalized heading difference |
| heading_valid | Indicates whether a valid heading estimate is available |
| speed_mps | Estimated GPS movement speed in metres per second |
| dt_prev_clipped | Clipped time difference from the previous observation |
| is_time_gap | Indicates whether a significant time gap exists |

### C. Observation Continuity Features

| Feature | Description |
|---|---|
| prev_obs_dist_m | Distance from the previous GPS observation |
| next_obs_dist_m | Distance to the next GPS observation |
| progress_along_arc | Progress along the candidate road segment |
| is_prev_connected | Indicates connectivity with the previous candidate |
| is_next_connected | Indicates connectivity with the next candidate |

### D. Road Network Features

| Feature | Description |
|---|---|
| has_reverse_arc | Indicates whether a reverse road connection exists |
| from_node_deg | Degree of the candidate segment's starting node |
| to_node_deg | Degree of the candidate segment's ending node |

### E. Candidate Ranking Features

| Feature | Description |
|---|---|
| dist_rank_norm | Normalized ranking of candidate distance |
| dist_diff_from_min_m | Difference between candidate distance and minimum candidate distance |
| dist_ratio_to_min | Ratio between candidate distance and minimum candidate distance |
| cos_heading_diff_vs_best | Heading similarity relative to the best candidate |
| num_candidates_obs | Number of generated candidates for the observation |

## 4. Feature Matrix

The model input matrix has the following structure:

- Rows: Candidate road-segment records.
- Columns: 28 observable features.
- Data type: Numerical floating-point values.
- Missing or infinite feature values must be checked before model inference.

The feature order must exactly match:

`ml/mlp/artifacts/feature_columns.json`

The same feature ordering must be maintained during training, validation and prediction.

## 5. Target Label

The target variable is `label`.

| Label | Meaning |
|---|---|
| 1 | Candidate road segment matches the reference route |
| 0 | Candidate road segment does not match the reference route |

Labels are generated using the reference route information during supervised training and evaluation.

Labels must not be used as model input features.

## 6. Preprocessing

The MLP pipeline applies StandardScaler to numerical features.

Standardization is performed using training data only.

During cross-validation, the scaler is fitted separately inside each training fold to prevent data leakage.

The fitted scaler is saved as:

`ml/mlp/artifacts/scaler.joblib`

## 7. Model Compatibility

The selected MLP architecture is:

`mlp_128_64_32`

Model configuration:

- Input features: 28
- Hidden layers: 128, 64, 32
- Activation: ReLU
- Batch Normalization: Enabled
- Dropout: 0.20
- Output: One classification logit per candidate

The model produces candidate match probabilities.

These probabilities can be normalized per GPS observation to generate emission probabilities for downstream HMM/Viterbi decoding.

## 8. Dataset Compatibility

The feature extraction pipeline supports:

- Raw PathSnap trip directories.
- Candidate-level CSV datasets.
- Candidate-level Parquet datasets.
- Candidate-level Pickle datasets.

An external candidate dataset must contain the required metadata and all 28 feature columns.

The feature names and ordering must remain consistent with the saved model configuration.

## 9. Important Implementation Files

| File | Responsibility |
|---|---|
| ml/mlp/data_adapter.py | Candidate generation and feature extraction |
| ml/mlp/config.py | Feature and model configuration |
| ml/mlp/model.py | PyTorch MLP architecture |
| ml/mlp/train_mlp.py | Model training and cross-validation |
| ml/mlp/evaluate_mlp.py | Model evaluation |
| ml/mlp/predict.py | Model inference and prediction |
| ml/mlp/artifacts/feature_columns.json | Saved feature-column ordering |

## 10. Summary

The PathSnap MLP uses 28 numerical features describing GPS-to-road geometry, movement direction, observation continuity, road-network connectivity and candidate ranking.

These features help the model estimate the probability that a candidate road segment corresponds to a GPS observation.

The resulting candidate probabilities are designed to support sequential route decoding through HMM/Viterbi integration.
