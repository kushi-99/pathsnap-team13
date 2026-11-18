# Candidate feature table

One row represents one GPS observation paired with one candidate directed road arc. `.route` is used only to assign training/evaluation labels; it is not an input to candidate generation or prediction.

| Column | Type | Role |
|---|---|---|
| `trip_id` | string | Metadata for grouping/splitting; never a model input |
| `observation_index` | integer | Metadata for grouping candidates by GPS point |
| `candidate_arc_id` | integer | Road candidate identity; never a numeric model input |
| `distance_to_road_m` | float | GPS-to-candidate segment distance in metres |
| `heading_difference_deg` | float | Absolute wrapped GPS movement vs directed road heading difference |
| `speed_mps` | float | GPS step distance divided by positive time gap |
| `time_gap_s` | float | Seconds since the previous GPS point |
| `incoming_degree` | integer | Number of arcs entering candidate arc's start node |
| `outgoing_degree` | integer | Number of arcs leaving candidate arc's end node |
| `reference_arc_id` | integer | Inferred reference road for the GPS point; supervision only |
| `label` | integer 0/1 | 1 if candidate arc equals the inferred reference arc, otherwise 0 |

The six model input columns are `distance_to_road_m`, `heading_difference_deg`, `speed_mps`, `time_gap_s`, `incoming_degree`, and `outgoing_degree`. The model pipeline imputes missing feature values. Do not pass identifiers or `reference_arc_id`/`label` into the model.

## Route alignment and limits

The `.route` file is an ordered sequence of `arc_id` values, not a pointwise GPS label file. `labels.py` assigns GPS observations to route positions using point-to-segment distance and a forward-only dynamic program that discourages large jumps. This is an inferred alignment and must not be presented as independently verified ground truth.

`results/label_alignment_quality.csv` records route distance and candidate coverage by trip. Trips `00000054` and `00000033` have median assigned-road distances of roughly 140 m and 122 m, respectively, and are flagged as route/GPS inconsistencies. The student-test partition has much smaller route distances overall. Discuss these anomalies and the candidate miss rate when presenting results.

Candidate miss rate means the fraction of GPS observations for which the generated candidate set does not contain the inferred reference arc. The candidate generator does not read `.route`; labels are attached afterward.
