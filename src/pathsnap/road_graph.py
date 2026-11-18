
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geo import project_trip


class RoadGraph:
    """Directed, weighted road network for one trip."""

    def __init__(self, nodes, arcs):
        self.nodes = nodes
        self.arcs = arcs

        self.arc_from = arcs["from_node_id"].to_numpy(dtype=np.int64)
        self.arc_to = arcs["to_node_id"].to_numpy(dtype=np.int64)
        self.num_arcs = len(arcs)
        self.num_nodes = len(nodes)

        node_xy, self.transformer = project_trip(
            nodes["longitude"], nodes["latitude"]
        )

        self.arc_lengths = np.linalg.norm(
            node_xy[self.arc_to] - node_xy[self.arc_from],
            axis=1
        )

        self.graph = csr_matrix(
            (
                self.arc_lengths,
                (self.arc_from, self.arc_to)
            ),
            shape=(self.num_nodes, self.num_nodes)
        )

        self.outgoing = {}
        self.incoming = {}
        self.pair_to_arc = {}

        for arc_id, (u, v) in enumerate(
            zip(self.arc_from, self.arc_to)
        ):
            self.outgoing.setdefault(u, []).append(arc_id)
            self.incoming.setdefault(v, []).append(arc_id)
            self.pair_to_arc.setdefault((u, v), arc_id)

    def are_connected(self, arc_a, arc_b):
        return self.arc_to[arc_a] == self.arc_from[arc_b]

    def get_outgoing_arcs(self, node_id):
        return self.outgoing.get(int(node_id), [])

    def get_incoming_arcs(self, node_id):
        return self.incoming.get(int(node_id), [])

    def get_next_arcs(self, arc_id):
        return self.get_outgoing_arcs(self.arc_to[arc_id])

    def get_previous_arcs(self, arc_id):
        return self.get_incoming_arcs(self.arc_from[arc_id])

    def shortest_path_distance(self, source_node, target_node):
        if source_node == target_node:
            return 0.0

        distances = self.distances_from_node(source_node)
        return float(distances[int(target_node)])

    def distances_from_node(self, source_node):
        return dijkstra(
            self.graph,
            directed=True,
            indices=int(source_node),
            return_predecessors=False
        )

    def shortest_path_arcs(self, source_node, target_node):
        """Return arc IDs forming the shortest directed path."""

        source_node = int(source_node)
        target_node = int(target_node)

        if source_node == target_node:
            return []

        distances, predecessors = dijkstra(
            self.graph,
            directed=True,
            indices=source_node,
            return_predecessors=True
        )

        if not np.isfinite(distances[target_node]):
            return None

        node_path = [target_node]
        current = target_node

        while current != source_node:
            current = int(predecessors[current])

            if current < 0:
                return None

            node_path.append(current)

        node_path.reverse()

        path_arcs = []

        for u, v in zip(node_path[:-1], node_path[1:]):
            arc_id = self.pair_to_arc.get((u, v))

            if arc_id is None:
                return None

            path_arcs.append(arc_id)

        return path_arcs

    def reconstruct_route(self, predicted_arcs):
        """Build a continuous road route from predicted arc IDs."""

        if len(predicted_arcs) == 0:
            return []

        route = [int(predicted_arcs[0])]

        for next_arc in predicted_arcs[1:]:
            next_arc = int(next_arc)
            previous_arc = route[-1]

            # Staying on the same road segment.
            if previous_arc == next_arc:
                continue

            # Directly connected road segments.
            if self.are_connected(previous_arc, next_arc):
                route.append(next_arc)
                continue

            # Find intermediate road segments.
            bridge = self.shortest_path_arcs(
                self.arc_to[previous_arc],
                self.arc_from[next_arc]
            )

            if bridge is None:
                raise ValueError(
                    f"No road path between arcs "
                    f"{previous_arc} and {next_arc}"
                )

            route.extend(bridge)
            route.append(next_arc)

        return route
