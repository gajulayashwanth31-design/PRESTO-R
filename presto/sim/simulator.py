"""Deterministic fluid-network simulator, independent of Gymnasium."""

from dataclasses import dataclass
from math import prod

import numpy as np

from presto.seeding import make_rng
from presto.sim.failures import Event, ScenarioGenerator, build_schedules
from presto.sim.topology import link_index, load_topology
from presto.sim.traffic import BackgroundTraffic


@dataclass
class LinkStats:
    delay: np.ndarray
    loss: np.ndarray
    queue: np.ndarray
    util: np.ndarray
    up: np.ndarray


class NetworkSimulator:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.rngs = {
            name: make_rng(seed, name) for name in ("topology", "traffic", "failures", "noise")
        }
        self.graph = load_topology(cfg, self.rngs["topology"])
        self.edge_ids = link_index(self.graph)
        self.n_links = len(self.edge_ids)
        self.capacity = np.empty(self.n_links, dtype=np.float64)
        self.base_delay = np.empty(self.n_links, dtype=np.float64)
        for _, _, data in self.graph.edges(data=True):
            link_id = int(data["link_id"])
            self.capacity[link_id] = float(data["capacity"])
            self.base_delay[link_id] = float(data["base_delay"])
        self.qmax = float(cfg["topology"]["qmax"])
        self.tick_ms = float(cfg["sim"]["tick_ms"])
        self.timeout_ms = float(cfg["link"]["timeout_ms"])
        self.base_loss = float(cfg["link"]["base_loss"])
        self.degr_loss = float(cfg["link"]["degr_loss"])
        self.managed_rate = float(cfg["traffic"]["managed_rate"])
        self.total_ticks = int(cfg["sim"]["warmup"]) + int(cfg["sim"]["episode_len"])
        self._t = 0
        self._schedules: dict[str, np.ndarray] = {}
        self.events: list[Event] = []
        self._queues = np.zeros(self.n_links, dtype=np.float64)
        self._last_up = np.ones(self.n_links, dtype=bool)
        self.traffic: BackgroundTraffic | None = None
        self.managed_src: int | None = None
        self.managed_dst: int | None = None

    def reset(self, managed_src: int, managed_dst: int, candidate_links: set[int]) -> None:
        """Create episode-specific traffic and event schedules using isolated streams."""
        if (
            managed_src not in self.graph
            or managed_dst not in self.graph
            or managed_src == managed_dst
        ):
            raise ValueError("Managed flow endpoints must be distinct nodes in the topology")
        self.managed_src = managed_src
        self.managed_dst = managed_dst
        self._t = 0
        self._queues.fill(0.0)
        self._last_up.fill(True)
        self.traffic = BackgroundTraffic(self.graph, self.cfg, self.rngs["traffic"])
        generator = ScenarioGenerator(self.cfg, self.rngs["failures"])
        protected = {managed_src, managed_dst}
        self.events = generator.generate(
            self.graph,
            int(self.cfg["sim"]["episode_len"]),
            candidate_links,
            protected,
        )
        health, up, mult = build_schedules(
            self.events,
            self.n_links,
            self.total_ticks,
            self.cfg,
            self.rngs["failures"],
        )
        self._schedules = {"health": health, "up": up, "mult": mult}

    def step(self, managed_path_links: list[int]) -> LinkStats:
        """Advance one tick on the selected path and return true per-link metrics."""
        if self.traffic is None or not self._schedules:
            raise RuntimeError("Call reset before stepping the simulator")
        if self._t >= self.total_ticks:
            raise RuntimeError("Episode is complete; reset the simulator before stepping again")
        if any(link < 0 or link >= self.n_links for link in managed_path_links):
            raise ValueError("Managed path contains an invalid link id")

        health = self._schedules["health"][self._t]
        up = self._schedules["up"][self._t]
        multipliers = self._schedules["mult"][self._t]
        if not np.array_equal(up, self._last_up):
            self.traffic.reroute(up)
            self._last_up = up.copy()

        load = self.traffic.link_load(self._t, multipliers)
        if managed_path_links:
            load[managed_path_links] += self.managed_rate

        effective_capacity = self.capacity * health
        effective_capacity = np.where(up, effective_capacity, 0.0)
        raw_queue = self._queues + load - effective_capacity
        overflow = np.maximum(0.0, raw_queue - self.qmax)
        self._queues = np.where(up, np.clip(raw_queue, 0.0, self.qmax), 0.0)
        loss = np.clip(
            overflow / np.maximum(load, np.finfo(np.float64).eps)
            + self.base_loss
            + (1.0 - health) * self.degr_loss,
            0.0,
            1.0,
        )
        delay = (
            self.base_delay
            + (self._queues / np.maximum(effective_capacity, np.finfo(np.float64).eps))
            * self.tick_ms
        )
        delay = np.where(up, delay, self.base_delay + self.timeout_ms)
        util = load / self.capacity
        loss = np.where(up, loss, 1.0)
        stats = LinkStats(delay, loss, self._queues.copy(), util, up.copy())
        self._t += 1
        return stats

    def observe(self, stats: LinkStats) -> LinkStats:
        """Add configured measurement noise without exposing hidden link health."""
        measurement = self.cfg.get("measurement", {})
        noise = self.rngs["noise"].normal(size=(4, self.n_links))
        delay = stats.delay + noise[0] * np.maximum(stats.delay, self.base_delay) * measurement.get(
            "delay_noise_rel", 0.0
        )
        loss = stats.loss + noise[1] * measurement.get("loss_noise_abs", 0.0)
        queue = stats.queue + noise[2] * self.qmax * measurement.get("queue_noise_rel", 0.0)
        util = stats.util + noise[3] * self.cfg["traffic"].get("noise_std", 0.0)
        return LinkStats(
            delay=np.maximum(delay, self.base_delay),
            loss=np.clip(loss, 0.0, 1.0),
            queue=np.clip(queue, 0.0, self.qmax),
            util=np.maximum(util, 0.0),
            up=stats.up.copy(),
        )

    def path_qos(self, stats: LinkStats, path_links: list[int]) -> dict:
        """Compute true path delay, loss, delivery, hop count, and availability."""
        if not path_links:
            raise ValueError("A path must contain at least one link")
        if any(link < 0 or link >= self.n_links for link in path_links):
            raise ValueError("Path contains an invalid link id")
        valid = all(bool(stats.up[link]) for link in path_links)
        path_loss = float(1.0 - prod(1.0 - float(stats.loss[link]) for link in path_links))
        if not valid:
            path_loss = 1.0
        return {
            "delay": sum(float(stats.delay[link]) for link in path_links),
            "loss": path_loss,
            "delivered": self.managed_rate * (1.0 - path_loss),
            "hops": len(path_links),
            "valid": valid,
        }

    @property
    def t(self) -> int:
        return self._t

    @property
    def schedules(self) -> dict[str, np.ndarray]:
        return self._schedules
