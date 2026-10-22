import numpy as np

from presto.config import load_config
from presto.sim.failures import Event, build_schedules
from presto.sim.simulator import NetworkSimulator


def _simulator(seed=4, managed_rate=0.0):
    cfg = load_config("configs/scenario/steady.yaml")
    cfg["traffic"]["n_background"] = 0
    cfg["traffic"]["managed_rate"] = managed_rate
    return NetworkSimulator(cfg, seed)


def test_idle_links_have_no_queue_and_base_delay():
    simulator = _simulator()
    simulator.reset(0, 1, set())

    stats = simulator.step([])

    np.testing.assert_allclose(stats.queue, 0.0)
    np.testing.assert_allclose(stats.delay, simulator.base_delay)


def test_overload_fills_queue_and_causes_loss():
    simulator = _simulator(managed_rate=30.0)
    link = simulator.edge_ids[(0, 1)]
    simulator.reset(0, 1, {link})

    simulator.step([link])
    stats = simulator.step([link])

    assert stats.queue[link] == simulator.qmax
    assert stats.loss[link] > 0.0


def test_same_seed_produces_identical_stats():
    first = _simulator(seed=91, managed_rate=2.0)
    second = _simulator(seed=91, managed_rate=2.0)
    first.reset(0, 1, set())
    second.reset(0, 1, set())

    for _ in range(5):
        a, b = first.step([]), second.step([])
        for field in ("delay", "loss", "queue", "util", "up"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_down_link_has_full_loss():
    simulator = _simulator()
    link = simulator.edge_ids[(0, 1)]
    simulator.reset(0, 1, {link})
    simulator.events = [Event("link_failure", [link], start=0, precursor=0, duration=2)]
    health, up, mult = build_schedules(
        simulator.events,
        simulator.n_links,
        simulator.total_ticks,
        simulator.cfg,
        simulator.rngs["failures"],
    )
    simulator._schedules = {"health": health, "up": up, "mult": mult}

    stats = simulator.step([link])

    assert stats.loss[link] == 1.0
    assert not stats.up[link]
    assert simulator.path_qos(stats, [link])["delivered"] == 0.0
