# Review 2 speaking summary — Kushi Niranjana

I validated 80 training trips and 10 student-test trips, with 3,624,101 training and 420,442 student-test candidate rows. The preprocessing pipeline generates nearby roads for each GPS fix and computes six distance, motion, and road-connectivity features. The model pipelines fit median imputation on their training rows to handle missing motion features.

Candidate labels come from inferred alignment to an ordered reference route. Trips 00000054 and 00000033 have unusually high candidate miss rates. I kept both trips in the dataset and documented them in the report and maps.

On the same trip-grouped 80/20 holdout, the models gave these results.

| Model | Selected parameters | Candidate accuracy | Macro F1 | Log loss | Top-1 covered | Top-1 all | Fit (s) | Inference (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | C=0.1 | 0.8646 | 0.6802 | 0.2728 | 0.7458 | 0.7285 | 12.36 | 1.20 |
| knn | k=9, weights=uniform | 0.8787 | 0.6961 | 1.0094 | 0.6697 | 0.6542 | 0.19 | 63.70 |

[Technical report](data_foundation_report.md) · [Full metrics](baseline_metrics.csv) · [Validation summary](feature_table_validation_summary.csv)
