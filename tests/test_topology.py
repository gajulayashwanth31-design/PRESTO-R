import networkx as nx

from presto.config import load_config
from presto.seeding import make_rng
from presto.sim.topology import link_index, load_topology


def test_nsfnet_shape_and_stable_link_ids():
    cfg = load_config("configs/scenario/mixed.yaml")

    first = load_topology(cfg, make_rng(5, "topology"))
    second = load_topology(cfg, make_rng(5, "topology"))

    assert first.number_of_nodes() == 14
    assert first.number_of_edges() == 21
    assert link_index(first) == link_index(second)
    indexed_edges = [
        (min(u, v), max(u, v))
        for u, v, _ in sorted(first.edges(data=True), key=lambda edge: edge[2]["link_id"])
    ]
    assert indexed_edges == sorted(indexed_edges)


def test_rgg_is_connected_with_minimum_degree_two():
    cfg = load_config("configs/scenario/mixed.yaml", ["topology.name=rgg"])
    cfg["topology"].update(n_nodes=24, radius=0.32, delay_per_unit_dist_ms=30)

    graph = load_topology(cfg, make_rng(11, "topology"))
    repeated = load_topology(cfg, make_rng(11, "topology"))

    assert nx.is_connected(graph)
    assert min(dict(graph.degree()).values()) >= 2
    assert link_index(graph) == link_index(repeated)
    assert all(
        "base_delay" in data and data["base_delay"] > 0 for *_, data in graph.edges(data=True)
    )
