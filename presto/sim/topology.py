"""Topology creation and stable link indexing."""

import networkx as nx
import numpy as np

NSFNET_EDGES = [
    (0, 1),
    (0, 2),
    (0, 7),
    (1, 2),
    (1, 3),
    (2, 5),
    (3, 4),
    (3, 10),
    (4, 5),
    (4, 6),
    (5, 9),
    (5, 12),
    (6, 7),
    (7, 8),
    (8, 9),
    (8, 11),
    (8, 13),
    (10, 11),
    (10, 13),
    (11, 12),
    (12, 13),
]


def _canonical_edges(graph: nx.Graph) -> list[tuple[int, int]]:
    return sorted((min(u, v), max(u, v)) for u, v in graph.edges)


def _assign_link_attributes(
    graph: nx.Graph, cfg: dict, rng: np.random.Generator, rgg: bool
) -> None:
    topology_cfg = cfg["topology"]
    delay_scale = topology_cfg.get("delay_per_unit_dist_ms")
    if rgg and delay_scale is None:
        raise ValueError("RGG topology requires topology.delay_per_unit_dist_ms")
    low, high = topology_cfg["base_delay_ms"] if not rgg else (0.0, 0.0)
    for link_id, (u, v) in enumerate(_canonical_edges(graph)):
        delay = (
            float(
                np.linalg.norm(
                    np.asarray(graph.nodes[u]["pos"]) - np.asarray(graph.nodes[v]["pos"])
                )
            )
            * delay_scale
            if rgg
            else float(rng.uniform(low, high))
        )
        graph.edges[u, v].update(
            capacity=float(topology_cfg["capacity"]),
            base_delay=float(delay),
            link_id=link_id,
        )


def load_topology(cfg: dict, rng: np.random.Generator) -> nx.Graph:
    """Build NSFNET or a connected random geometric graph with stable link ids."""
    topology_cfg = cfg["topology"]
    name = topology_cfg["name"]
    if name == "nsfnet":
        graph = nx.Graph()
        graph.add_nodes_from(range(14))
        graph.add_edges_from(NSFNET_EDGES)
        _assign_link_attributes(graph, cfg, rng, rgg=False)
        return graph
    if name != "rgg":
        raise ValueError(f"Unsupported topology: {name}")

    n_nodes = int(topology_cfg["n_nodes"])
    radius = float(topology_cfg["radius"])
    for _ in range(10_000):
        graph_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        graph = nx.random_geometric_graph(n_nodes, radius, seed=graph_seed)
        if nx.is_connected(graph) and min(dict(graph.degree()).values()) >= 2:
            _assign_link_attributes(graph, cfg, rng, rgg=True)
            return graph
    raise RuntimeError(
        f"Could not generate a connected RGG with minimum degree 2 "
        f"after 10000 attempts (n={n_nodes}, radius={radius})"
    )


def link_index(graph: nx.Graph) -> dict[tuple[int, int], int]:
    """Map canonical undirected edges to their stable link ids."""
    return {(min(u, v), max(u, v)): int(data["link_id"]) for u, v, data in graph.edges(data=True)}
