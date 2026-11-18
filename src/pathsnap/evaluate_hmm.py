
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .data_loader import load_trip
from .features import FEATURE_COLUMNS, build_features
from .geo import project_trip
from .hmm import HMM
from .labels import assign_reference_labels
from .road_graph import RoadGraph


VALIDATION_TRIPS = [
    1, 5, 13, 16, 24, 28, 35, 37,
    38, 41, 43, 55, 61, 84, 85, 88
]

BETA_VALUES = [50.0]


def evaluate_trip(trip_id, data_root, hmm, beta_values):

    trip_folder_id = f"{int(trip_id):08d}"

    print(f"\nEvaluating trip {trip_folder_id}...")

    trip = load_trip(
        data_root,
        trip_folder_id,
        split="train"
    )

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    # 1. Generate candidate road segments.
    candidates = generate_candidates(
        track,
        nodes,
        arcs,
        trip_id=trip_folder_id
    )

    # 2. Generate features.
    features = build_features(
        track,
        nodes,
        arcs,
        candidates
    )

    # 3. Assign provisional reference labels.
    labelled = assign_reference_labels(
        track,
        nodes,
        arcs,
        route,
        features,
        trip_id=trip_folder_id
    )

    # 4. Calculate HGB probabilities.
    probabilities = hmm.model.predict_proba(
        labelled[FEATURE_COLUMNS]
    )[:, 1]

    labelled = labelled.copy()
    labelled["hgb_probability"] = probabilities

    grouped = list(
        labelled.groupby("observation_index", sort=True)
    )

    observations = []
    hgb_predictions = []
    reference_arcs = []

    for observation_index, group in grouped:

        best_index = group["hgb_probability"].idxmax()

        hgb_predictions.append(
            int(labelled.loc[best_index, "candidate_arc_id"])
        )

        reference_arcs.append(
            int(group["reference_arc_id"].iloc[0])
        )

        observations.append(group.copy())

    reference_arcs = np.asarray(reference_arcs)
    hgb_predictions = np.asarray(hgb_predictions)

    hgb_accuracy = float(
        (hgb_predictions == reference_arcs).mean()
    )

    # 5. Build road graph.
    road_graph = RoadGraph(nodes, arcs)

    # 6. Calculate GPS distances.
    gps_xy, _ = project_trip(
        track["longitude"],
        track["latitude"]
    )

    gps_distances = np.linalg.norm(
        np.diff(gps_xy, axis=0),
        axis=1
    )

    print(f"GPS observations: {len(reference_arcs):,}")
    print(f"Candidate rows: {len(labelled):,}")
    print(f"HGB-only accuracy: {hgb_accuracy:.4%}")

    results = []

    # 7. Test each beta value.
    for beta in beta_values:

        print(f"\nTesting beta = {beta}...")

        try:

            hmm_predictions = hmm.viterbi(
                observations,
                road_graph,
                gps_distances,
                beta=beta
            )

            hmm_predictions = np.asarray(hmm_predictions)

            hmm_accuracy = float(
                (hmm_predictions == reference_arcs).mean()
            )

            accuracy_difference = (
                hmm_accuracy - hgb_accuracy
            )

            # 8. Reconstruct continuous route.
            reconstructed_route = road_graph.reconstruct_route(
                hmm_predictions
            )

            route_continuous = all(
                road_graph.are_connected(a, b)
                for a, b in zip(
                    reconstructed_route[:-1],
                    reconstructed_route[1:]
                )
            )

            result = {
                "trip_id": trip_folder_id,
                "beta": beta,
                "observations": len(reference_arcs),
                "hgb_accuracy": hgb_accuracy,
                "hmm_accuracy": hmm_accuracy,
                "accuracy_difference": accuracy_difference,
                "reconstructed_arcs": len(reconstructed_route),
                "route_continuous": route_continuous,
                "error": ""
            }

            print(f"HMM accuracy: {hmm_accuracy:.4%}")
            print(
                f"Accuracy difference: "
                f"{accuracy_difference:+.4%}"
            )
            print(
                f"Reconstructed route arcs: "
                f"{len(reconstructed_route)}"
            )
            print(f"Route continuous: {route_continuous}")

        except (ValueError, IndexError) as error:

            print(f"Beta {beta} failed: {error}")

            result = {
                "trip_id": trip_folder_id,
                "beta": beta,
                "observations": len(reference_arcs),
                "hgb_accuracy": hgb_accuracy,
                "hmm_accuracy": np.nan,
                "accuracy_difference": np.nan,
                "reconstructed_arcs": 0,
                "route_continuous": False,
                "error": str(error)
            }

        results.append(result)

    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default=str(Path.home() / "Downloads/Student folder")
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of validation trips. 0 means all 16."
    )

    parser.add_argument(
        "--model",
        default="results/models/hist_gradient_boosting.joblib"
    )

    parser.add_argument(
        "--output",
        default="results/models/beta_comparison.csv"
    )

    args = parser.parse_args()

    trips = VALIDATION_TRIPS

    if args.limit > 0:
        trips = trips[:args.limit]

    print("Loading original HGB model...")
    print(f"Model: {args.model}")
    print(f"Validation trips selected: {len(trips)}")
    print(f"Beta values: {BETA_VALUES}")

    hmm = HMM(args.model)

    all_results = []

    for trip_id in trips:

        trip_results = evaluate_trip(
            trip_id,
            args.data_root,
            hmm,
            BETA_VALUES
        )

        all_results.extend(trip_results)

    results_df = pd.DataFrame(all_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    # 9. Summarise beta performance.
    summary = results_df.groupby("beta").agg(
        average_hgb_accuracy=("hgb_accuracy", "mean"),
        average_hmm_accuracy=("hmm_accuracy", "mean"),
        average_accuracy_difference=("accuracy_difference", "mean"),
        continuous_routes=("route_continuous", "sum"),
        successful_runs=("hmm_accuracy", "count")
    )

    print("\n========== BETA COMPARISON ==========")
    print(summary.to_string())

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
