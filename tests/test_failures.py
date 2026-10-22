import networkx as nx
import numpy as np

from presto.config import load_config
from presto.seeding import make_rng
from presto.sim.failures import Event, ScenarioGenerator, build_schedules
from presto.sim.topology import load_topology


def test_precursor_degrades_before_failure_and_recovers():
    cfg = load_config("configs/scenario/mixed.yaml")
    event = Event("link_failure", [0], start=8, precursor=4, duration=3)

    health, up, _ = build_schedules([event], 1, 15, cfg, make_rng(3, "failures"))

    assert np.all(np.diff(health[4:8, 0]) <= 0)
    assert health[4, 0] == 1.0
    assert np.all(~up[8:11, 0])
    assert np.all(up[11:, 0])
    assert health[11, 0] == 1.0


def test_sudden_failure_has_no_precursor():
    cfg = load_config("configs/scenario/mixed.yaml")
    event = Event("link_failure", [0], start=8, precursor=0, duration=2)

    health, up, _ = build_schedules([event], 1, 12, cfg, make_rng(3, "failures"))

    np.testing.assert_array_equal(health[:8, 0], 1.0)
    assert not up[8, 0]


def test_node_failures_never_select_protected_nodes():
    cfg = load_config("configs/scenario/mixed.yaml")
    cfg["scenario"]["events_per_episode"] = {
        "congestion": 0,
        "link_failure": 0,
        "node_failure": 30,
        "fluctuation": 0,
    }
    graph = load_topology(cfg, make_rng(4, "topology"))
    generator = ScenarioGenerator(cfg, make_rng(8, "failures"))

    events = generator.generate(graph, 100, set(range(10)), {0, 13})

    node_events = [event for event in events if event.kind == "node_failure"]
    assert node_events
    assert all(event.params["node"] not in {0, 13} for event in node_events)
    assert all(len(event.links) == graph.degree[event.params["node"]] for event in node_events)


def test_rgg_generation_uses_a_connected_graph():
    cfg = load_config("configs/scenario/mixed.yaml", ["topology.name=rgg"])
    cfg["topology"].update(n_nodes=24, radius=0.32, delay_per_unit_dist_ms=30)
    graph = load_topology(cfg, make_rng(9, "topology"))

    assert nx.is_connected(graph)
