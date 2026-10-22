import numpy as np

from presto.config import load_config
from presto.seeding import make_rng
from presto.sim.topology import link_index, load_topology
from presto.state import build_observation, candidate_paths, choose_flow


def test_observation_dimensions_and_bounds():
    latest = np.array(
        [[2.0, 0.01, 0.2, 0.5, 1.0], [3.0, 0.02, 0.4, 1.0, 1.0]],
        dtype=np.float64,
    )
    risk = np.array([0.3, 0.8])

    with_risk = build_observation(latest, risk, [[0], [1]], 0, 2, 5.0, 2, True, 4)
    without_risk = build_observation(latest, risk, [[0], [1]], 0, 2, 5.0, 2, False, 4)

    assert with_risk.shape == (37,)
    assert without_risk.shape == (29,)
    assert np.all((with_risk >= 0.0) & (with_risk <= 1.0))
    assert np.all((without_risk >= 0.0) & (without_risk <= 1.0))
    assert with_risk[7] == 1.0


def test_candidate_paths_are_stable_and_flow_has_alternatives():
    cfg = load_config("configs/scenario/mixed.yaml")
    graph = load_topology(cfg, make_rng(3, "topology"))
    lid = link_index(graph)

    src, dst, paths = choose_flow(graph, make_rng(17, "flow"), 4, lid)

    assert src != dst
    assert len(paths) == 4
    assert paths == candidate_paths(graph, src, dst, 4, lid)
