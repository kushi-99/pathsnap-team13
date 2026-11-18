
import heapq
from collections import OrderedDict

import numpy as np
import joblib

from .features import FEATURE_COLUMNS


class HMM:

    def __init__(self, model_path):
        saved_model = joblib.load(model_path)

        if isinstance(saved_model, dict):
            self.model = saved_model["model"]
            self.feature_columns = saved_model["feature_columns"]
        else:
            self.model = saved_model
            self.feature_columns = FEATURE_COLUMNS

        # Persistent cache for shortest-path distances.
        self.distance_cache = OrderedDict()
        self.max_cache_size = 100000

    def emission_probabilities(self, candidates):
        """Calculate ML-based emission probabilities."""

        if candidates.empty:
            raise ValueError("Candidate dataframe is empty.")

        X = candidates[self.feature_columns]

        scores = self.model.predict_proba(X)[:, 1]
        scores = np.clip(scores, 1e-12, None)

        result = candidates.copy()
        result["emission_probability"] = scores

        group_columns = ["observation_index"]

        if "trip_id" in result.columns:
            group_columns.insert(0, "trip_id")

        totals = result.groupby(
            group_columns
        )["emission_probability"].transform("sum")

        result["emission_probability"] /= totals

        return result

    def distances_to_targets(
        self,
        road_graph,
        source_node,
        target_nodes
    ):
        """Calculate shortest distances to required target nodes."""

        source_node = int(source_node)
        targets = set(int(node) for node in target_nodes)

        result = {}
        missing = set()

        for target in targets:
            key = (source_node, target)

            if key in self.distance_cache:
                self.distance_cache.move_to_end(key)
                result[target] = self.distance_cache[key]
            else:
                missing.add(target)

        if not missing:
            return result

        distances = {source_node: 0.0}
        queue = [(0.0, source_node)]

        if source_node in missing:
            result[source_node] = 0.0
            missing.remove(source_node)

        while queue and missing:

            current_distance, node = heapq.heappop(queue)

            if current_distance > distances.get(node, np.inf):
                continue

            if node in missing:
                result[node] = current_distance
                missing.remove(node)

                if not missing:
                    break

            for arc_id in road_graph.get_outgoing_arcs(node):

                next_node = int(road_graph.arc_to[arc_id])
                weight = float(road_graph.arc_lengths[arc_id])

                new_distance = current_distance + weight

                if new_distance < distances.get(next_node, np.inf):

                    distances[next_node] = new_distance

                    heapq.heappush(
                        queue,
                        (new_distance, next_node)
                    )

        # Store unreachable targets as infinity.
        for target in missing:
            result[target] = np.inf

        # Update persistent cache.
        for target, distance in result.items():

            key = (source_node, target)

            if key not in self.distance_cache:
                self.distance_cache[key] = distance

            self.distance_cache.move_to_end(key)

        while len(self.distance_cache) > self.max_cache_size:
            self.distance_cache.popitem(last=False)

        return result

    def transition_probabilities(
        self,
        previous_candidates,
        current_candidates,
        road_graph,
        gps_distance,
        beta=50.0
    ):
        """Calculate transition probabilities between observations."""

        if beta <= 0:
            raise ValueError("Beta must be greater than zero.")

        previous_arcs = previous_candidates[
            "candidate_arc_id"
        ].to_numpy(dtype=int)

        current_arcs = current_candidates[
            "candidate_arc_id"
        ].to_numpy(dtype=int)

        probabilities = np.zeros(
            (len(previous_arcs), len(current_arcs)),
            dtype=float
        )

        target_nodes = [
            int(road_graph.arc_from[arc])
            for arc in current_arcs
        ]

        source_groups = {}

        for i, arc_id in enumerate(previous_arcs):

            source_node = int(road_graph.arc_to[arc_id])

            source_groups.setdefault(
                source_node, []
            ).append((i, arc_id))

        for source_node, previous_group in source_groups.items():

            distances = self.distances_to_targets(
                road_graph,
                source_node,
                target_nodes
            )

            for i, arc_a in previous_group:

                for j, arc_b in enumerate(current_arcs):

                    if arc_a == arc_b:

                        # Approximate movement along the same road segment.
                        road_distance = min(
                            gps_distance,
                            road_graph.arc_lengths[arc_a]
                        )

                    else:

                        target_node = int(
                            road_graph.arc_from[arc_b]
                        )

                        path_distance = distances[target_node]

                        if not np.isfinite(path_distance):
                            continue

                        road_distance = (
                            road_graph.arc_lengths[arc_a] / 2.0
                            + path_distance
                            + road_graph.arc_lengths[arc_b] / 2.0
                        )

                    difference = abs(
                        road_distance - gps_distance
                    )

                    probabilities[i, j] = np.exp(
                        -difference / beta
                    )

                total = probabilities[i].sum()

                if total > 0:
                    probabilities[i] /= total

        return probabilities

    def viterbi(
        self,
        observations,
        road_graph,
        gps_distances,
        beta=50.0
    ):
        """Decode the most likely sequence of road segments."""

        if not observations:
            return []

        num_steps = len(observations)

        if len(gps_distances) != num_steps - 1:
            raise ValueError(
                "gps_distances must contain one distance "
                "for every consecutive observation pair."
            )

        if beta <= 0:
            raise ValueError("Beta must be greater than zero.")

        # Calculate emission probabilities.
        observations = [
            self.emission_probabilities(group)
            for group in observations
        ]

        first_emissions = observations[0][
            "emission_probability"
        ].to_numpy()

        scores = np.log(
            np.clip(first_emissions, 1e-12, None)
        )

        backpointers = []

        # Process consecutive GPS observations.
        for t in range(1, num_steps):

            transitions = self.transition_probabilities(
                observations[t - 1],
                observations[t],
                road_graph,
                gps_distances[t - 1],
                beta=beta
            )

            # Preserve impossible transitions as negative infinity.
            with np.errstate(divide="ignore"):
                log_transitions = np.log(transitions)

            emissions = observations[t][
                "emission_probability"
            ].to_numpy()

            log_emissions = np.log(
                np.clip(emissions, 1e-12, None)
            )

            combined = scores[:, None] + log_transitions

            best_previous = np.argmax(combined, axis=0)

            best_scores = combined[
                best_previous,
                np.arange(len(emissions))
            ]

            scores = best_scores + log_emissions

            backpointers.append(best_previous)

        # Backtrack to find the best candidate sequence.
        state = int(np.argmax(scores))

        best_path = [state]

        for t in range(num_steps - 2, -1, -1):

            state = int(backpointers[t][state])
            best_path.append(state)

        best_path.reverse()

        predicted_arcs = []

        for t, candidate_index in enumerate(best_path):

            arc_id = int(
                observations[t].iloc[candidate_index][
                    "candidate_arc_id"
                ]
            )

            predicted_arcs.append(arc_id)

        return predicted_arcs
