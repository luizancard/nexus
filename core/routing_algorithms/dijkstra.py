"""Least-impedance routing over a condition graph via Dijkstra.

A correct shortest path over the `cost`-weighted graph is all the thesis needs
(SS11: "a correct Dijkstra over the impedance graph is enough"). This is a thin,
audited wrapper over NetworkX's Dijkstra rather than a re-implementation -- the
scientific claim lives in the cost model and the comparison, not in the search.

The one non-trivial detail: the fused graph is a `MultiDiGraph`, so a node path
alone is ambiguous when parallel edges exist (they can carry different
attributes). This graph currently has none (every node pair has a single edge),
but metric #2 scores the *exact* edges traversed, so the wrapper resolves and
returns the concrete edge key per hop -- always the minimum-`cost` parallel edge,
i.e. the one Dijkstra would actually take -- so scoring is never ambiguous even
if a future graph gains parallel edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class Route:
    """One computed path: node sequence, the exact edges used, and total cost."""

    nodes: list[Any]
    edge_keys: list[tuple[Any, Any, int]]
    cost: float

    @property
    def edge_set(self) -> set[tuple[Any, Any, int]]:
        """The traversed edges as a set, for path-difference comparison."""
        return set(self.edge_keys)


def shortest_path(
    graph: nx.MultiDiGraph, origin: Any, destination: Any, weight: str = "cost"
) -> Route | None:
    """Least-`weight` route from `origin` to `destination`, or None if unreachable.

    Returns a `Route` carrying the node path, the concrete (u, v, key) edges
    actually traversed, and the total cost. `None` (rather than an exception) on
    no-path keeps batch OD evaluation simple; the Lourdes graph is a single
    connected component, so this should not occur for valid nodes.
    """
    try:
        nodes = nx.shortest_path(graph, origin, destination, weight=weight)
    except nx.NetworkXNoPath:
        return None

    edge_keys: list[tuple[Any, Any, int]] = []
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        key = _min_weight_key(graph, u, v, weight)
        edge_keys.append((u, v, key))
        total += float(graph[u][v][key][weight])
    return Route(nodes=nodes, edge_keys=edge_keys, cost=total)


def _min_weight_key(graph: nx.MultiDiGraph, u: Any, v: Any, weight: str) -> int:
    """The parallel-edge key between u and v with the smallest `weight`.

    Dijkstra relaxes on the cheapest parallel edge, so recovering the same edge
    here keeps the returned edge sequence consistent with the cost that produced
    the path.
    """
    parallels = graph[u][v]
    return min(parallels, key=lambda k: float(parallels[k][weight]))


__all__ = ["Route", "shortest_path"]
