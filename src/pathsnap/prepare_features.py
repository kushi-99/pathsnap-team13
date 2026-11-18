import argparse
from pathlib import Path

import pandas as pd

from .candidates import generate_candidates
from .data_loader import list_trip_ids, load_trip
from .features import build_features
from .labels import assign_reference_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--split",
        choices=["train", "student_test"],
        default="train",
    )
    parser.add_argument("--out", default="results/train_features.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="For a quick run, process only the first N trips. 0 means all.",
    )
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--radius-m", type=float, default=100.0)
    args = parser.parse_args()

    trip_ids = list_trip_ids(args.data_root, args.split)
    if args.limit:
        trip_ids = trip_ids[:args.limit]

    all_trip_rows = []

    for number, trip_id in enumerate(trip_ids, start=1):
        print(f"Processing {args.split}/{trip_id} ({number}/{len(trip_ids)})")

        trip = load_trip(args.data_root, trip_id, args.split)

        candidates = generate_candidates(
            trip["track"],
            trip["nodes"],
            trip["arcs"],
            trip_id=trip_id,
            k=args.k,
            radius_m=args.radius_m,
        )

        features = build_features(
            trip["track"],
            trip["nodes"],
            trip["arcs"],
            candidates,
        )

        labelled = assign_reference_labels(
            trip["track"],
            trip["nodes"],
            trip["arcs"],
            trip["route"],
            features,
            trip_id=trip_id,
        )

        all_trip_rows.append(labelled)

    if not all_trip_rows:
        raise RuntimeError("No trips found. Check your data folder and split.")

    result = pd.concat(all_trip_rows, ignore_index=True)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    has_reference_candidate = result.groupby(
        ["trip_id", "observation_index"]
    )["label"].max()

    candidate_miss_rate = (has_reference_candidate == 0).mean()

    print(f"Saved {len(result):,} rows to {output_path}")
    print(f"Candidate miss rate: {candidate_miss_rate:.3%}")


if __name__ == "__main__":
    main()
    