# Candidate feature table (starter contract)

One row represents one GPS observation paired with one candidate directed road arc.
The `.route` reference is kept separate from prediction features.

| Column | Type | Role |
|---|---|---|
| `trip_id` | string | Metadata/group split only; never a model feature |
| `observation_index` | integer | Metadata for grouping candidate rows by GPS point |
| `candidate_arc_id` | integer | Road candidate being scored; metadata/target identity, not a numeric model feature |
| `distance_to_road_m` | float | Perpendicular distance from GPS point to candidate road segment |
| `heading_difference_deg` | float | Absolute wrapped difference between GPS movement heading and directed arc heading |
| `speed_mps` | float | GPS step distance divided by positive timestamp gap |
| `time_gap_s` | float | Time since preceding GPS observation |
| `reference_arc_id` | integer | Provisional per-observation reference road, created from `.route` geometry |
| `label` | integer 0/1 | 1 if candidate arc equals provisional reference arc, else 0 |

## Provisional label rule: validate before reporting

The supplied `.route` file is an ordered sequence of `arc_id` values, not a pointwise
label file. The starter code assigns each GPS point the closest arc geometry from that
route, then enforces nondecreasing positions in the route sequence. This is a transparent
initial alignment heuristic, not a guarantee of exact ground truth. Inspect mapped
training trips and document limitations before finalizing it.

The candidate generator does not use `.route`; labels are attached afterward. Report
candidate miss rate: the fraction of GPS observations for which no generated candidate
matches the provisional reference arc. Do not use `reference_arc_id`, `label`, `trip_id`,
or row identifiers as numeric model inputs. The starter models use only the four numeric
features listed above.