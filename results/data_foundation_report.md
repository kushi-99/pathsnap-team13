# PathSnap Team 13 — Data Foundation and Feature Validation

## Executive summary

The data foundation loads 80 training trips and 10 student-test trips, builds candidate-road feature tables, validates source records and labels, and records Logistic Regression and KNN baselines. Training data contain 208,417 GPS observations and 3,624,101 candidate rows; student-test data contain 19,160 observations and 420,442 candidate rows. Candidate labels are inferred by aligning GPS observations to an ordered reference route. They are not independently verified pointwise ground truth.

## Objective and dataset structure

PathSnap scores likely road segments for noisy GPS observations before route reconstruction. Each trip has whitespace-separated `.track` (longitude, latitude, timestamp seconds), `.nodes` (longitude, latitude), `.arcs` (directed from/to node row indices), and `.route` (ordered arc IDs). Each arc ID is its row index in `.arcs`. The `train` and `student_test` folders are under `data/Student folder/`.

| Split | Trips | GPS observations | Road nodes | Directed arcs | Route arcs | Missing track cells | Invalid GPS / nodes / endpoints | Non-increasing timestamp steps | Largest gap (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 | 0 | 0 / 0 / 0 | 0 | 153,064 |
| Student test | 10 | 19,160 | 90,689 | 185,041 | 3,477 | 0 | 0 / 0 / 0 | 0 | 210 |

Raw checks are recorded per trip in `dataset_validation_report.csv`. Feature-table schemas match between splits: True. The train table has 184,689 positive and 3,439,412 negative candidate rows; the student-test table has 18,857 positive and 401,585 negative rows. Student-test rows were not used to fit or tune models.

## Candidate generation and preprocessing

Coordinates are projected to a local UTM zone. A KD-tree over road-segment midpoints retrieves up to 40 nearby arcs per GPS point. The pipeline retains candidates within 100 m or keeps the closest retrieved segment when all are outside that radius. The feature table has one row per `(trip_id, observation_index, candidate_arc_id)` and includes reference metadata for supervision and evaluation. The models use only the six features below. The fixed midpoint query approximates nearest segment search and can miss the reference road.

The six model inputs are:

| Feature | Meaning | Unit | Train range | Train missing | Student-test missing | Negative values | Limitation |
|---|---|---|---:|---:|---:|---:|---|
| `distance_to_road_m` | Minimum projected distance from the GPS point to the candidate road segment. | meters | 0 to 455.38 | 0 | 0 | 0 / 0 | Candidate geometric fit; candidate search may omit the true road. |
| `heading_difference_deg` | Smallest circular difference between previous-step GPS bearing and directed arc bearing. | degrees | 1.11816e-06 to 180 | 8,590 | 1,002 | 0 / 0 | Missing for the first or near-stationary GPS step; noisy at low speeds. |
| `speed_mps` | Projected distance from the previous GPS point divided by a positive time gap. | meters/second | 0 to 180.54 | 1,989 | 220 | 0 / 0 | GPS noise or long time gaps can produce large values. |
| `time_gap_s` | Current GPS timestamp minus the previous timestamp. | seconds | 1 to 153064 | 1,989 | 220 | 0 / 0 | First observation is missing; timestamp gaps are retained. |
| `incoming_degree` | Number of directed arcs ending at the candidate arc's start node. | count | 0 to 7 | 0 | 0 | 0 / 0 | Directed connectivity only; it does not encode road class. |
| `outgoing_degree` | Number of directed arcs starting at the candidate arc's end node. | count | 0 to 7 | 0 | 0 | 0 / 0 | Directed connectivity only; it does not encode road class. |

Missing values are preserved in the feature tables and handled by median imputation inside each sklearn model pipeline. Imputers are fitted on the training partition only. Training missing counts are `heading_difference_deg` 8,590, `speed_mps` 1,989, `time_gap_s` 1,989. Negative feature counts are zero in both feature tables. The largest recorded time gap is 153,064 seconds in training and 210 seconds in student test; resulting speeds and time gaps are retained rather than silently clipped or dropped. Full minima, maxima, missing counts, and negative checks are in `feature_validation_report.csv`.

## Label alignment and data-quality caveats

The `.route` file provides an ordered road-arc sequence rather than pointwise labels. A forward-only dynamic program aligns each GPS observation to a route position using GPS-to-segment distance, monotonic route progression, and timestamp-aware advancement. Candidate label 1 means its arc equals the inferred reference arc for that observation; all other candidate arcs receive 0. Validation checks that each label agrees with the candidate/reference arc IDs in the row. This alignment is an inference and should not be described as verified ground truth.

Two training trips have unusually high candidate miss rates. The route-distance columns use the existing alignment audit where available.

| Trip | Candidate miss rate | Median GPS-to-route distance (m) | P90 distance (m) |
|---|---:|---:|---:|
| 00000054 | 87.0% | 140.09 | 196.97 |
| 00000033 | 51.8% | 122.41 | 139.68 |

They remain in the data and are documented as source/label-quality limitations. Per-trip candidate coverage is in `candidate_trip_validation_train.csv` and `candidate_trip_validation_student_test.csv`.

## Baseline models and evaluation

Both models use the same six features, median imputation, and standard scaling. Logistic Regression uses class balancing and all eligible training rows. KNN uses 5 neighbors with distance weighting and a reproducible balanced training sample capped at 100,000 candidate rows, so its fit data differ in size from Logistic Regression. The recorded evaluation uses a fixed 80/20 GroupShuffleSplit by trip (random state 42; 64 training trips and 16 validation trips). No trip appears in both partitions. Candidate classification metrics are computed on observations containing at least one inferred reference candidate. Covered-observation Top-1 is reported separately from all-observation Top-1, where missed reference candidates count as incorrect.

| Model | Candidate accuracy | Macro F1 | Log loss | Top-1 covered | Top-1 all | Fit (s) | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8647 | 0.6802 | 0.2728 | 0.7458 | 0.7285 | 8.23 | 0.84 |
| knn | 0.8880 | 0.7057 | 1.5288 | 0.6508 | 0.6357 | 0.32 | 94.27 |

Metrics and exact validation trip IDs are in [baseline_metrics.csv](baseline_metrics.csv). Logistic Regression has higher all-observation Top-1 and lower log loss in this run. KNN has higher candidate Macro F1 and takes longer to score the holdout.

## EDA and maps

The feature distribution panels display speed and time gaps through the 99th percentile of each split's sample. The correlation chart uses a bounded random sample of training candidate rows.

- [GPS observations per trip](figures/gps_observations_per_trip.png)
- [Candidate roads per GPS observation](figures/candidates_per_observation.png)
- [Distributions of all six features](figures/feature_distributions.png)
- [Missing feature values](figures/missing_values.png)
- [Feature correlations](figures/feature_correlation.png)
- [Candidate miss rates by trip](figures/candidate_miss_rate_by_trip.png)
- [Timestamp gaps and trip size](figures/trip_data_quality.png)
- [Logistic Regression and KNN Top-1 accuracy](figures/top1_accuracy.png)

The network display is deterministically thinned to at most 1,500 arcs. All coordinates and road geometries come from the dataset. Green candidate segments indicate inferred positive labels and dashed orange segments indicate negative labels. Base tiles require internet access.

- [GPS trajectory](maps/gps_trajectory.html)
- [Nearby road network](maps/road_network.html)
- [GPS and road-network overlay](maps/gps_network_overlay.html)
- [Candidate roads](maps/candidate_roads.html)
- [Trip 00000054](maps/problematic_trip_00000054.html)
- [Trip 00000033](maps/problematic_trip_00000033.html)

## Limitations and conclusion

The route-to-GPS labels are inferred; candidate generation may omit a reference arc; GPS gaps can create extreme speed values; only 80 labeled training trips are available; and the baseline comparison is one trip-grouped holdout. KNN's balanced 100,000-row cap also means its training sample differs from Logistic Regression's. These limits should accompany performance claims. The validated feature schema, preserved missing values, trip-level checks, measured baselines, figures, and maps provide a reproducible foundation for the team's route-reconstruction and deployment work.
