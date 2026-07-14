# Code for the CausalMan: A Digital-Twin for Large-Scale Causality
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Efficient latent projection and ADMG-to-MAG conversion for CausalMan.

CausalMan uses a NetworkX DAG whose nodes carry a mandatory Boolean
``Observable`` attribute. Projected mixed graphs are represented as
``nx.MultiDiGraph`` objects:

* a directed edge ``A -> B`` is stored once with key/type ``directed``;
* a bidirected edge ``A <-> B`` is stored as two reciprocal edges with
  key/type ``bidirected``.

The multigraph is essential: an ADMG may contain a bow, i.e. both ``A -> B``
and ``A <-> B``. A plain ``nx.DiGraph`` would overwrite one of these edges.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Hashable, Iterable

import networkx as nx


OBSERVABLE_ATTRIBUTE = "Observable"
DIRECTED = "directed"
BIDIRECTED = "bidirected"

Node = Hashable
DirectedEdge = tuple[Node, Node]
BidirectedEdge = frozenset[Node]


def _validate_observability(
    graph: nx.DiGraph,
    observable_attribute: str = OBSERVABLE_ATTRIBUTE,
) -> tuple[list[Node], list[Node]]:
    """Return observed/latent nodes after strict observability validation."""
    observed: list[Node] = []
    latent: list[Node] = []
    for node, attributes in graph.nodes(data=True):
        if observable_attribute not in attributes:
            raise ValueError(
                f"Node {node!r} is missing the required Boolean "
                f"{observable_attribute!r} attribute."
            )
        value = attributes[observable_attribute]
        if type(value) is not bool:
            raise TypeError(
                f"Node {node!r} has non-Boolean {observable_attribute!r}={value!r}."
            )
        (observed if value else latent).append(node)
    return observed, latent


def _extract_mixed_edges(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    strict: bool = True,
) -> tuple[set[DirectedEdge], set[BidirectedEdge]]:
    """Extract logical directed and bidirected edges from a NetworkX graph.

    Bidirected edges must be stored as reciprocal arcs. In strict mode, a
    missing reciprocal half or an unknown edge type raises an exception.
    """
    directed: set[DirectedEdge] = set()
    bidirected_halves: set[DirectedEdge] = set()

    if graph.is_multigraph():
        raw_edges = (
            (u, v, data)
            for u, v, _key, data in graph.edges(keys=True, data=True)
        )
    else:
        raw_edges = graph.edges(data=True)

    for u, v, data in raw_edges:
        if u == v:
            raise ValueError(f"Self-edge {u!r} -> {v!r} is not supported.")
        edge_type = data.get("edge_type")
        if edge_type == DIRECTED:
            directed.add((u, v))
        elif edge_type == BIDIRECTED:
            bidirected_halves.add((u, v))
        elif strict:
            raise ValueError(
                f"Edge ({u!r}, {v!r}) has invalid edge_type={edge_type!r}."
            )

    malformed = {
        (u, v)
        for u, v in bidirected_halves
        if (v, u) not in bidirected_halves
    }
    if malformed and strict:
        raise ValueError(
            "Bidirected edges must be stored as reciprocal arcs; missing "
            f"halves for: {sorted(map(repr, malformed))}."
        )

    bidirected = {
        frozenset((u, v))
        for u, v in bidirected_halves
        if (v, u) in bidirected_halves
    }
    return directed, bidirected


def _new_mixed_graph(
    nodes: Iterable[tuple[Node, dict]],
    *,
    graph_type: str,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    graph.graph["graph_type"] = graph_type
    graph.graph["bidirected_storage"] = "reciprocal"
    return graph


def _add_directed(graph: nx.MultiDiGraph, source: Node, target: Node) -> None:
    graph.add_edge(source, target, key=DIRECTED, edge_type=DIRECTED)


def _add_bidirected(graph: nx.MultiDiGraph, left: Node, right: Node) -> None:
    if left == right:
        raise ValueError("A bidirected self-edge is not valid.")
    graph.add_edge(left, right, key=BIDIRECTED, edge_type=BIDIRECTED)
    graph.add_edge(right, left, key=BIDIRECTED, edge_type=BIDIRECTED)


def _build_mixed_graph(
    nodes: Iterable[tuple[Node, dict]],
    directed: Iterable[DirectedEdge],
    bidirected: Iterable[BidirectedEdge],
    *,
    graph_type: str,
) -> nx.MultiDiGraph:
    graph = _new_mixed_graph(nodes, graph_type=graph_type)
    for source, target in directed:
        _add_directed(graph, source, target)
    for pair in bidirected:
        left, right = tuple(pair)
        _add_bidirected(graph, left, right)
    return graph


def get_latent_projection_single(
    graph: nx.DiGraph,
    *,
    observable_attribute: str = OBSERVABLE_ATTRIBUTE,
) -> nx.MultiDiGraph:
    """Compute the latent projection of a CausalMan DAG.

    The traversal stops whenever it reaches an observed node, so all internal
    nodes on a projected path are latent. The implementation uses reachability
    traversals rather than enumerating paths.

    Returns:
        A bow-safe ``nx.MultiDiGraph`` representing the projected ADMG.
    """
    if graph.is_multigraph():
        raise TypeError("The input causal graph must be a simple nx.DiGraph.")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The input graph must be a DAG.")

    observed, latent = _validate_observability(graph, observable_attribute)
    observed_set = set(observed)

    reachable_cache: dict[Node, set[Node]] = {}

    def latent_reachable(start: Node) -> set[Node]:
        if start in reachable_cache:
            return reachable_cache[start]
        reachable_observed: set[Node] = set()
        visited: set[Node] = {start}
        queue: deque[Node] = deque([start])
        while queue:
            current = queue.popleft()
            for child in graph.successors(current):
                if child in visited:
                    continue
                visited.add(child)
                if child in observed_set:
                    reachable_observed.add(child)
                else:
                    queue.append(child)
        reachable_cache[start] = reachable_observed
        return reachable_observed

    directed: set[DirectedEdge] = set()
    for source in observed:
        for target in latent_reachable(source):
            if source != target:
                directed.add((source, target))

    bidirected: set[BidirectedEdge] = set()
    for latent_node in latent:
        descendants = sorted(latent_reachable(latent_node), key=str)
        for left, right in combinations(descendants, 2):
            bidirected.add(frozenset((left, right)))

    observed_nodes = [(node, dict(graph.nodes[node])) for node in observed]
    return _build_mixed_graph(
        observed_nodes,
        directed,
        bidirected,
        graph_type="ADMG",
    )


def _ancestry(
    nodes: Iterable[Node],
    directed: set[DirectedEdge],
) -> tuple[nx.DiGraph, dict[Node, set[Node]]]:
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(nodes)
    directed_graph.add_edges_from(directed)
    if not nx.is_directed_acyclic_graph(directed_graph):
        raise ValueError("The directed component of the mixed graph is cyclic.")
    ancestors = {
        node: nx.ancestors(directed_graph, node) | {node}
        for node in directed_graph.nodes
    }
    return directed_graph, ancestors


def _adjacencies(
    directed: set[DirectedEdge],
    bidirected: set[BidirectedEdge],
) -> set[BidirectedEdge]:
    return {frozenset(edge) for edge in directed} | set(bidirected)


def _inducing_pairs(
    nodes: list[Node],
    directed: set[DirectedEdge],
    bidirected: set[BidirectedEdge],
) -> list[tuple[Node, Node]]:
    """Find non-adjacent endpoints connected by an inducing path.

    In an ADMG/MAG, adjacent internal colliders on an inducing path must be
    joined by bidirected edges and must be ancestors of at least one endpoint.
    This yields a polynomial connectivity test without path enumeration.
    """
    _directed_graph, ancestors = _ancestry(nodes, directed)
    adjacent = _adjacencies(directed, bidirected)

    bidirected_graph = nx.Graph()
    bidirected_graph.add_nodes_from(nodes)
    bidirected_graph.add_edges_from(tuple(pair) for pair in bidirected)

    inducing: list[tuple[Node, Node]] = []
    for left, right in combinations(nodes, 2):
        if frozenset((left, right)) in adjacent:
            continue

        valid_colliders = (ancestors[left] | ancestors[right]) - {left, right}
        if not valid_colliders:
            continue

        # An endpoint contributes an arrowhead at the neighboring collider via
        # endpoint -> collider or endpoint <-> collider.
        from_left = {
            node
            for node in valid_colliders
            if (left, node) in directed
            or frozenset((left, node)) in bidirected
        }
        from_right = {
            node
            for node in valid_colliders
            if (right, node) in directed
            or frozenset((right, node)) in bidirected
        }
        if not from_left or not from_right:
            continue

        collider_subgraph = bidirected_graph.subgraph(valid_colliders)
        if any(
            not from_left.isdisjoint(component)
            and not from_right.isdisjoint(component)
            for component in nx.connected_components(collider_subgraph)
        ):
            inducing.append((left, right))

    return inducing


def _orient_mag_adjacency(
    left: Node,
    right: Node,
    ancestors: dict[Node, set[Node]],
) -> tuple[str, DirectedEdge | BidirectedEdge]:
    if left in ancestors[right]:
        return DIRECTED, (left, right)
    if right in ancestors[left]:
        return DIRECTED, (right, left)
    return BIDIRECTED, frozenset((left, right))


def admg2mag(admg: nx.DiGraph | nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Convert an ADMG into a maximal ancestral graph.

    Existing bows are resolved according to directed ancestry, and missing
    adjacencies implied by inducing paths are added iteratively. The result is
    validated as ancestral and maximal.
    """
    nodes = list(admg.nodes())
    directed_admg, bidirected_admg = _extract_mixed_edges(admg, strict=True)
    _directed_graph, ancestors = _ancestry(nodes, directed_admg)

    # Inducing paths must first be detected in the original ADMG. Resolving a
    # bow to its ancestral MAG orientation can otherwise remove an arrowhead
    # that participates in such a path before the required adjacency is found.
    required_pairs = _adjacencies(directed_admg, bidirected_admg)
    required_pairs.update(
        frozenset(pair)
        for pair in _inducing_pairs(nodes, directed_admg, bidirected_admg)
    )

    mag_directed: set[DirectedEdge] = set()
    mag_bidirected: set[BidirectedEdge] = set()
    for pair in required_pairs:
        left, right = tuple(pair)
        edge_type, edge = _orient_mag_adjacency(left, right, ancestors)
        if edge_type == DIRECTED:
            mag_directed.add(edge)  # type: ignore[arg-type]
        else:
            mag_bidirected.add(edge)  # type: ignore[arg-type]

    # Add every adjacency required by an inducing path. Repeating to a fixed
    # point makes maximality explicit while still avoiding path enumeration.
    max_passes = max(1, len(nodes) * (len(nodes) - 1) // 2)
    for _ in range(max_passes):
        missing_pairs = _inducing_pairs(nodes, mag_directed, mag_bidirected)
        if not missing_pairs:
            break
        for left, right in missing_pairs:
            edge_type, edge = _orient_mag_adjacency(left, right, ancestors)
            if edge_type == DIRECTED:
                mag_directed.add(edge)  # type: ignore[arg-type]
            else:
                mag_bidirected.add(edge)  # type: ignore[arg-type]
    else:
        raise RuntimeError("MAG maximalization did not converge.")

    mag_nodes = [(node, dict(admg.nodes[node])) for node in nodes]
    mag = _build_mixed_graph(
        mag_nodes,
        mag_directed,
        mag_bidirected,
        graph_type="MAG",
    )
    validate_mag(mag)
    return mag


def validate_mag(mag: nx.DiGraph | nx.MultiDiGraph) -> None:
    """Raise if ``mag`` is cyclic, non-ancestral, or non-maximal."""
    nodes = list(mag.nodes())
    directed, bidirected = _extract_mixed_edges(mag, strict=True)
    _directed_graph, ancestors = _ancestry(nodes, directed)

    bows = {frozenset(edge) for edge in directed} & bidirected
    if bows:
        raise ValueError(f"A MAG cannot contain bows: {sorted(map(repr, bows))}.")

    for pair in bidirected:
        left, right = tuple(pair)
        if left in ancestors[right] or right in ancestors[left]:
            raise ValueError(
                f"Bidirected edge {left!r} <-> {right!r} violates ancestry."
            )

    inducing = _inducing_pairs(nodes, directed, bidirected)
    if inducing:
        raise ValueError(
            "The graph is not maximal; inducing paths remain between "
            f"non-adjacent pairs: {inducing[:10]!r}."
        )


def count_edge_types(graph: nx.DiGraph | nx.MultiDiGraph) -> tuple[int, int]:
    """Return logical ``(directed, bidirected)`` edge counts."""
    directed, bidirected = _extract_mixed_edges(graph, strict=True)
    return len(directed), len(bidirected)


_GRAPHML_TYPES = (bool, int, float, str)


def write_mixed_graph_graphml(
    graph: nx.DiGraph | nx.MultiDiGraph,
    path: str | Path,
) -> None:
    """Validate and write a mixed graph using a matching GraphML format."""
    _extract_mixed_edges(graph, strict=True)
    # GraphML only supports bool/int/float/str — export a copy with non-serializable
    # node attributes stripped so sympy objects (from the original DAG) don't break writes.
    export = graph.copy()
    for _, data in export.nodes(data=True):
        for key in list(data.keys()):
            if not isinstance(data[key], _GRAPHML_TYPES):
                del data[key]
    nx.write_graphml(export, Path(path))


def create_node_dict(nodes: Iterable[Node], observed_list: Iterable[Node]):
    """Create nodes carrying the correctly spelled observability attribute."""
    observed = set(observed_list)
    return [(node, {OBSERVABLE_ATTRIBUTE: node in observed}) for node in nodes]


def test_mediation_chain() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(create_node_dict(["A", "U", "B"], ["A", "B"]))
    graph.add_edges_from([("A", "U"), ("U", "B")])
    admg = get_latent_projection_single(graph)
    directed, bidirected = _extract_mixed_edges(admg)
    assert directed == {("A", "B")}
    assert not bidirected


def test_latent_confounder_fork() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(create_node_dict(["A", "U", "B"], ["A", "B"]))
    graph.add_edges_from([("U", "A"), ("U", "B")])
    admg = get_latent_projection_single(graph)
    directed, bidirected = _extract_mixed_edges(admg)
    assert not directed
    assert bidirected == {frozenset(("A", "B"))}


def test_latent_collider() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(create_node_dict(["A", "U", "B"], ["A", "B"]))
    graph.add_edges_from([("A", "U"), ("B", "U")])
    admg = get_latent_projection_single(graph)
    assert count_edge_types(admg) == (0, 0)


def test_bow_graph() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(create_node_dict(["A", "U", "B"], ["A", "B"]))
    graph.add_edges_from([("A", "B"), ("U", "A"), ("U", "B")])
    admg = get_latent_projection_single(graph)
    directed, bidirected = _extract_mixed_edges(admg)
    assert directed == {("A", "B")}
    assert bidirected == {frozenset(("A", "B"))}
    assert count_edge_types(admg) == (1, 1)

    mag = admg2mag(admg)
    mag_directed, mag_bidirected = _extract_mixed_edges(mag)
    assert mag_directed == {("A", "B")}
    assert not mag_bidirected


def test_long_latent_path() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(
        create_node_dict(["A", "U1", "U2", "B"], ["A", "B"])
    )
    graph.add_edges_from([("A", "U1"), ("U1", "U2"), ("U2", "B")])
    admg = get_latent_projection_single(graph)
    assert _extract_mixed_edges(admg)[0] == {("A", "B")}


def test_missing_observable_attribute() -> None:
    graph = nx.DiGraph([("A", "B")])
    try:
        get_latent_projection_single(graph)
    except ValueError as error:
        assert "Observable" in str(error)
    else:
        raise AssertionError("Missing observability metadata should raise.")


if __name__ == "__main__":
    test_mediation_chain()
    test_latent_confounder_fork()
    test_latent_collider()
    test_bow_graph()
    test_long_latent_path()
    test_missing_observable_attribute()
    print("All graph projection tests passed.")
