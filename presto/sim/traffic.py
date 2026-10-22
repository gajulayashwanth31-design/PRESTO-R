"""Time-varying background traffic routed over the network."""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from presto.sim.topology import link_index


@dataclass
class BgFlow:
    src: int
    dst: int
    base_rate: float
    phase: float
    path_links: list[int]


class BackgroundTraffic:
    def __init__(self, graph: nx.Graph, cfg: dict, rng: np.random.Generator):
        self.graph = graph
        self.cfg = cfg["traffic"]
        self.rng = rng
        self.edge_ids = link_index(graph)
        nodes = sorted(graph.nodes)
        self.flows: list[BgFlow] = []
        mean, sigma = self.cfg["bg_rate_lognormal"]
        for _ in range(int(self.cfg["n_background"])):
            src, dst = rng.choice(nodes, size=2, replace=False)
            self.flows.append(
                BgFlow(
                    src=int(src),
                    dst=int(dst),
                    base_rate=float(rng.lognormal(mean, sigma)),
                    phase=float(rng.uniform(0.0, 2.0 * np.pi)),
                    path_links=[],
                )
            )
        self._rates_base = np.asarray([flow.base_rate for flow in self.flows])
        self._phases = np.asarray([flow.phase for flow in self.flows])
        self._incidence = np.zeros((len(self.flows), len(self.edge_ids)), dtype=np.float64)
        self._has_link = self._incidence.astype(bool)
        self._flow_indices, self._link_indices = np.nonzero(self._incidence)
        self.reroute(np.ones(len(self.edge_ids), dtype=bool))

    def rates(self, t: int, multipliers: np.ndarray) -> np.ndarray:
        """Return one congestion-modulated noisy rate per background flow."""
        if multipliers.shape != (len(self.edge_ids),):
            raise ValueError("Congestion multiplier vector has the wrong shape")
        if not self.flows:
            return np.zeros(0, dtype=np.float64)
        amplitude = float(self.cfg["diurnal_amp"])
        period = float(self.cfg["diurnal_period"])
        noise_std = float(self.cfg.get("noise_std", 0.0))
        waves = 1.0 + amplitude * np.sin(2.0 * np.pi * t / period + self._phases)
        congestion = np.max(np.where(self._has_link, multipliers[None, :], 1.0), axis=1)
        noise = self.rng.normal(0.0, noise_std, len(self.flows))
        return np.maximum(0.0, self._rates_base * waves * congestion + noise)

    def reroute(self, up: np.ndarray) -> None:
        """Route each flow by base delay using only currently live links."""
        if up.shape != (len(self.edge_ids),):
            raise ValueError("Link-up vector has the wrong shape")
        live = nx.Graph()
        live.add_nodes_from(self.graph.nodes)
        for u, v, data in self.graph.edges(data=True):
            if up[int(data["link_id"])]:
                live.add_edge(u, v, base_delay=data["base_delay"])
        for flow in self.flows:
            try:
                nodes = nx.shortest_path(
                    live, flow.src, flow.dst, weight="base_delay", method="dijkstra"
                )
            except nx.NetworkXNoPath:
                flow.path_links = []
            else:
                flow.path_links = [
                    self.edge_ids[(min(u, v), max(u, v))] for u, v in zip(nodes, nodes[1:])
                ]
        self._incidence.fill(0.0)
        for index, flow in enumerate(self.flows):
            self._incidence[index, flow.path_links] = 1.0
        self._has_link = self._incidence.astype(bool)
        self._flow_indices, self._link_indices = np.nonzero(self._incidence)

    def link_load(self, t: int, multipliers: np.ndarray) -> np.ndarray:
        """Aggregate background rates per link."""
        if multipliers.shape != (len(self.edge_ids),):
            raise ValueError("Congestion multiplier vector has the wrong shape")
        if not self.flows:
            return np.zeros(len(self.edge_ids), dtype=np.float64)
        rates = self.rates(t, multipliers)
        return np.bincount(
            self._link_indices,
            weights=rates[self._flow_indices],
            minlength=len(self.edge_ids),
        )
