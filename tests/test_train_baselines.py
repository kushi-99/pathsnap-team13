import unittest

import pandas as pd

from pathsnap.features import FEATURE_COLUMNS
from pathsnap.train_baselines import (
    grouped_train_validation_split,
    tune_hyperparameters,
)
from pathsnap.evaluate import evaluate_predictions


def small_training_frame():
    rows = []
    for trip in range(15):
        for observation in range(5):
            for candidate_arc_id in (0, 1):
                positive = candidate_arc_id == 0
                rows.append({
                    "trip_id": f"{trip:08d}",
                    "observation_index": observation,
                    "candidate_arc_id": candidate_arc_id,
                    "reference_arc_id": 0,
                    "label": int(positive),
                    "distance_to_road_m": 0.0 if positive else 20.0,
                    "heading_difference_deg": 0.0 if positive else 90.0,
                    "speed_mps": float(observation + 1),
                    "time_gap_s": 1.0,
                    "incoming_degree": float(trip % 3),
                    "outgoing_degree": float((trip + 1) % 3),
                })
    return pd.DataFrame(rows, columns=[
        "trip_id", "observation_index", "candidate_arc_id",
        "reference_arc_id", "label", *FEATURE_COLUMNS,
    ])


class GroupedTuningTests(unittest.TestCase):
    def test_grouped_split_keeps_trips_separate(self):
        frame = small_training_frame()
        train_idx, validation_idx = grouped_train_validation_split(frame)
        train_trips = set(frame.iloc[train_idx].trip_id)
        validation_trips = set(frame.iloc[validation_idx].trip_id)
        self.assertTrue(train_trips)
        self.assertTrue(validation_trips)
        self.assertTrue(train_trips.isdisjoint(validation_trips))

    def test_tuning_records_all_candidates_without_using_test_data(self):
        frame = small_training_frame()
        best, results = tune_hyperparameters(
            frame, max_knn_rows=30, max_observations=8
        )
        self.assertEqual(set(best), {"logistic_regression", "knn"})
        self.assertEqual(len(results), 9)
        self.assertEqual(set(results.model), {"logistic_regression", "knn"})
        self.assertTrue(results.top1_accuracy.between(0, 1).all())

    def test_evaluator_metrics_count_missed_candidates_as_incorrect(self):
        trip = {"trip_id": "00000015", "track": pd.DataFrame(index=range(2))}
        labeled = pd.DataFrame({
            "observation_index": [0, 0, 1],
            "candidate_arc_id": [10, 11, 20],
            "reference_arc_id": [10, 10, 21],
            "label": [1, 0, 0],
        })
        result = evaluate_predictions(trip, labeled, [0.8, 0.2, 0.9], "test")
        self.assertEqual(result["candidate_coverage"], 0.5)
        self.assertEqual(result["top1_accuracy_all"], 0.5)
        self.assertEqual(result["top1_accuracy_when_candidate_present"], 1.0)


if __name__ == "__main__":
    unittest.main()
