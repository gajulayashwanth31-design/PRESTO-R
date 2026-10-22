"""Scenario event generation and ground-truth link schedules."""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Event:
    kind: str
    links: list[int]
    start: int
    precursor: int
    duration: int
    params: dict = field(default_factory=dict)


class ScenarioGenerator:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg["scenario"]
        self.warmup = int(cfg["sim"]["warmup"])
        self.rng = rng

    def _int_range(self, key: str) -> int:
        low, high = self.cfg[key]
        return int(self.rng.integers(int(low), int(high) + 1))

    def _target_link(self, n_links: int, candidate_links: set[int]) -> int:
        candidates = sorted(candidate_links)
        if candidates and self.rng.random() < self.cfg["target_on_candidate_prob"]:
            return int(self.rng.choice(candidates))
        return int(self.rng.integers(0, n_links))

    def generate(
        self,
        graph: nx.Graph,
        episode_len: int,
        candidate_links: set[int],
        protected_nodes: set[int],
    ) -> list[Event]:
        """Sample event counts and targets from the configured scenario mix."""
        edge_by_id = {int(data["link_id"]): (u, v) for u, v, data in graph.edges(data=True)}
        n_links = len(edge_by_id)
        earliest = self.warmup + 15
        latest = self.warmup + episode_len - 10
        if latest < earliest:
            raise ValueError("Episode is too short for the configured event-start window")
        events: list[Event] = []
        for kind, mean in self.cfg["events_per_episode"].items():
            for _ in range(int(self.rng.poisson(float(mean)))):
                start = int(self.rng.integers(earliest, latest + 1))
                if kind == "congestion":
                    link = self._target_link(n_links, candidate_links)
                    events.append(
                        Event(
                            kind,
                            [link],
                            start,
                            0,
                            self._int_range("congestion_duration"),
                            {
                                "multiplier": float(
                                    self.rng.uniform(*self.cfg["congestion_multiplier"])
                                )
                            },
                        )
                    )
                elif kind == "fluctuation":
                    link = self._target_link(n_links, candidate_links)
                    ends_in_failure = bool(self.rng.random() < 0.3)
                    params = {
                        "period": self._int_range("fluct_period"),
                        "depth": float(self.rng.uniform(*self.cfg["fluct_depth"])),
                        "ends_in_failure": ends_in_failure,
                    }
                    if ends_in_failure:
                        params["failure_duration"] = self._int_range("down_duration")
                    events.append(
                        Event(
                            kind, [link], start, 0, self._int_range("congestion_duration"), params
                        )
                    )
                elif kind == "link_failure":
                    link = self._target_link(n_links, candidate_links)
                    precursor = (
                        0
                        if self.rng.random() < self.cfg["no_precursor_prob"]
                        else self._int_range("precursor_len")
                    )
                    events.append(
                        Event(
                            kind,
                            [link],
                            start,
                            precursor,
                            self._int_range("down_duration"),
                        )
                    )
                elif kind == "node_failure":
                    possible_nodes = sorted(set(graph.nodes) - set(protected_nodes))
                    if not possible_nodes:
                        continue
                    if candidate_links and self.rng.random() < self.cfg["target_on_candidate_prob"]:
                        candidate_nodes = sorted(
                            {
                                node
                                for link_id in candidate_links
                                for node in edge_by_id[link_id]
                                if node not in protected_nodes
                            }
                        )
                    else:
                        candidate_nodes = []
                    node = int(self.rng.choice(candidate_nodes or possible_nodes))
                    incident = sorted(
                        int(data["link_id"]) for _, _, data in graph.edges(node, data=True)
                    )
                    if incident:
                        precursor = (
                            0
                            if self.rng.random() < self.cfg["no_precursor_prob"]
                            else self._int_range("precursor_len")
                        )
                        events.append(
                            Event(
                                kind,
                                incident,
                                start,
                                precursor,
                                self._int_range("down_duration"),
                                {"node": node},
                            )
                        )
                else:
                    raise ValueError(f"Unsupported event kind: {kind}")
        return sorted(events, key=lambda event: (event.start, event.kind, event.links))


def build_schedules(
    events: list[Event],
    E: int,
    T: int,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand events into health, availability, and congestion arrays for T ticks."""
    health = np.ones((T, E), dtype=np.float64)
    up = np.ones((T, E), dtype=bool)
    congestion = np.ones((T, E), dtype=np.float64)
    scenario_cfg = cfg["scenario"]
    minimum_health = float(scenario_cfg["h_min_precursor"])
    down_low, down_high = scenario_cfg["down_duration"]

    def degrade_and_fail(links: list[int], start: int, precursor: int, duration: int) -> None:
        precursor_start = max(0, start - precursor)
        if precursor > 0 and precursor_start < T:
            ticks = np.arange(precursor_start, min(start, T))
            if ticks.size:
                levels = np.linspace(1.0, minimum_health, precursor + 1)[:-1]
                levels = levels[: ticks.size]
                noise = rng.normal(0.0, 0.005, size=ticks.size)
                levels = np.minimum.accumulate(np.clip(levels + noise, minimum_health, 1.0))
                health[np.ix_(ticks, links)] = np.minimum(
                    health[np.ix_(ticks, links)], levels[:, None]
                )
        end = min(T, start + duration)
        if start < T and end > 0:
            up[max(0, start) : end, links] = False
            health[max(0, start) : end, links] = 0.0

    for event in events:
        links = sorted(set(event.links))
        if any(link < 0 or link >= E for link in links):
            raise ValueError(f"Event contains a link outside 0..{E - 1}")
        if event.kind in {"link_failure", "node_failure"}:
            degrade_and_fail(links, event.start, event.precursor, event.duration)
        elif event.kind == "congestion":
            multiplier = float(event.params["multiplier"])
            stop = min(T, event.start + event.duration)
            ticks = np.arange(max(0, event.start), stop)
            if ticks.size:
                ramp = np.minimum(1.0, (ticks - event.start + 1) / 3.0)
                congestion[np.ix_(ticks, links)] = np.maximum(
                    congestion[np.ix_(ticks, links)], 1.0 + (multiplier - 1.0) * ramp[:, None]
                )
        elif event.kind == "fluctuation":
            stop = min(T, event.start + event.duration)
            ticks = np.arange(max(0, event.start), stop)
            if ticks.size:
                phase = 2.0 * np.pi * (ticks - event.start) / float(event.params["period"])
                fluctuating = 1.0 - float(event.params["depth"]) * (0.5 + 0.5 * np.sin(phase))
                fluctuating += rng.normal(0.0, 0.01, size=ticks.size)
                health[np.ix_(ticks, links)] = np.minimum(
                    health[np.ix_(ticks, links)], np.clip(fluctuating[:, None], 0.0, 1.0)
                )
            if event.params.get("ends_in_failure", False):
                if "failure_duration" in event.params:
                    duration = int(event.params["failure_duration"])
                else:
                    duration = int(rng.integers(int(down_low), int(down_high) + 1))
                degrade_and_fail(links, event.start + event.duration, 0, duration)
        else:
            raise ValueError(f"Unsupported event kind: {event.kind}")

    return health, up, congestion
