import pandas as pd

path = "results/train_features_aligned.csv"

features = [
    "distance_to_road_m",
    "heading_difference_deg",
    "speed_mps",
    "time_gap_s",
    "incoming_degree",
    "outgoing_degree"
]

minimum = {f: float("inf") for f in features}
maximum = {f: float("-inf") for f in features}
negative = {f: 0 for f in features}

total_rows = 0

for chunk in pd.read_csv(path, chunksize=100000):

    total_rows += len(chunk)

    for feature in features:
        values = chunk[feature].dropna()

        if len(values) > 0:
            minimum[feature] = min(minimum[feature], values.min())
            maximum[feature] = max(maximum[feature], values.max())
            negative[feature] += (values < 0).sum()

print("Total rows:", total_rows)

print("\nFEATURE VALIDATION")

for feature in features:
    print("\n", feature)
    print("Minimum:", minimum[feature])
    print("Maximum:", maximum[feature])
    print("Negative values:", negative[feature])