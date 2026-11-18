# PathSnap Team 13 - Kushi first-review summary

## Completed

- Loaded and inspected 80 training trips and 10 student-test trips.
- Generated EDA inventories for both partitions.
- Generated nearby road candidates and six shared candidate features.
- Replaced pointwise-nearest plus cumulative-max labels with a forward-only dynamic route alignment.
- Regenerated aligned feature tables without changing the original candidate feature tables.
- Trained Logistic Regression and KNN using a trip-grouped 80/20 holdout (64 train trips, 16 validation trips).

## Data quality

Training inventory: 80 trips, 208,417 GPS observations; no missing track cells, invalid coordinates, or invalid arc endpoints. Twenty-two trips have a timestamp gap greater than 60 seconds; the largest gap is 153,064 seconds and needs separate discussion/handling.

The inferred-route candidate miss rate is 11.385% over all training observations and 1.581% over all student-test observations. Most training misses come from two route/GPS outlier trips:

- Trip 00000054: 87.0% candidate miss; median assigned route-arc distance about 140 m.
- Trip 00000033: 51.8% candidate miss; median assigned route-arc distance about 122 m.

Student-test route-arc distances are much closer overall (median of per-trip medians about 4.7 m). These are inferred labels, not independently verified pointwise ground truth; disclose the two training anomalies.

## Shared candidate row

One row is one GPS observation paired with one directed candidate road. Model inputs:

- `distance_to_road_m`
- `heading_difference_deg`
- `speed_mps`
- `time_gap_s`
- `incoming_degree`
- `outgoing_degree`

`trip_id`, `observation_index`, and `candidate_arc_id` are identifiers. `reference_arc_id` and `label` are supervision/evaluation only. The label is 1 when the candidate arc equals the route arc assigned by the forward-only dynamic alignment, otherwise 0.

## Initial baseline results

Metrics below use the same trip holdout. Candidate classification metrics are computed only for observations where candidate generation included the inferred reference road. The all-observation Top-1 metric counts missed candidates as wrong.

| Model | Candidate accuracy | Candidate Macro F1 | Candidate Log Loss | Top-1, covered obs | Top-1, all obs | Fit time | Validation inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8647 | 0.6802 | 0.2728 | 0.7458 | 0.7285 | 8.23 s | 0.84 s |
| KNN | 0.8880 | 0.7057 | 1.5288 | 0.6508 | 0.6357 | 0.32 s | 94.27 s |

Logistic Regression is the stronger initial route-ranking baseline: it has higher all-observation Top-1 accuracy, substantially lower Log Loss, and faster inference. KNN has slightly higher candidate Macro F1, so final model selection remains open until comparison with teammates' models.

## Files produced

- `results/eda_inventory.csv`
- `results/eda_student_test_inventory.csv`
- `results/train_features.csv` and `results/student_test_features.csv` (original labels)
- `results/train_features_aligned.csv` and `results/student_test_features_aligned.csv` (forward-aligned labels)
- `results/label_alignment_audit.csv`
- `results/label_alignment_quality.csv`
- `results/label_alignment_review.png`
- `results/baseline_metrics.csv`
- `results/baseline_models/logistic_regression.joblib`
- `results/baseline_models/knn.joblib`

## Still for later, not first-review claims

This is a single group-held-out baseline comparison, not five-fold cross-validation. Student-test model metrics and final comparison/tuning with the other three models remain future team work. The label alignment still requires the route/GPS outlier caveat above.
