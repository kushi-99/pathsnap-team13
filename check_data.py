
import pandas as pd

path = "results/train_features_aligned.csv"

features = [
    "heading_difference_deg",
    "speed_mps",
    "time_gap_s"
]

missing_counts = {feature: 0 for feature in features}
missing_combinations = {}

for chunk in pd.read_csv(path, chunksize=100000):

    missing = chunk[features].isnull()

    for feature in features:
        missing_counts[feature] += missing[feature].sum()

    combinations = (
        missing.astype(int)
        .astype(str)
        .agg("".join, axis=1)
        .value_counts()
    )

    for pattern, count in combinations.items():
        missing_combinations[pattern] = (
            missing_combinations.get(pattern, 0) + count
        )

print("MISSING FEATURE COUNTS")

for feature, count in missing_counts.items():
    print(feature, ":", count)

print("\nMISSING VALUE COMBINATIONS")

for pattern, count in sorted(missing_combinations.items()):
    print(pattern, ":", count)
