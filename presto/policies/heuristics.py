"""Deterministic and random heuristic routing baselines."""

import numpy as np

from presto.policies.base import Policy


def _valid_paths(info: dict) -> list[int]:
    valid = np.asarray(info["valid"], dtype=bool)
    if valid.ndim != 1 or valid.size != len(info["paths"]):
        raise ValueError("valid must contain one flag per candidate path")
    return np.flatnonzero(valid).tolist()


class SPFailover(Policy):
    """Stay on the shortest-base-delay path until it becomes unavailable."""

    name = "sp_failover"

    def act(self, obs: np.ndarray, info: dict) -> int:
        current = int(info["current"])
        if bool(info["valid"][current]):
            return current
        candidates = _valid_paths(info)
        return candidates[0] if candidates else current


class GreedyDelay(Policy):
    """Choose the currently valid path with lowest observed end-to-end delay."""

    name = "greedy_delay"

    def act(self, obs: np.ndarray, info: dict) -> int:
        candidates = _valid_paths(info)
        if not candidates:
            return int(info["current"])
        links = info["paths"]
        delay = np.asarray(info["obs_link"])[:, 0]
        return min(candidates, key=lambda i: (sum(float(delay[link]) for link in links[i]), i))


class RiskThreshold(Policy):
    """Leave the current path when any link's predicted risk exceeds tau."""

    name = "risk_threshold"

    def __init__(self, tau: float = 0.5):
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        self.tau = float(tau)

    def act(self, obs: np.ndarray, info: dict) -> int:
        current = int(info["current"])
        paths = info["paths"]
        risk = np.asarray(info["risk"], dtype=np.float64)
        if risk.ndim != 1 or np.any(~np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
            raise ValueError("risk must be a finite per-link probability vector")
        path_risk = [max((float(risk[link]) for link in path), default=0.0) for path in paths]
        if bool(info["valid"][current]) and path_risk[current] <= self.tau:
            return current
        candidates = _valid_paths(info)
        return min(candidates, key=lambda i: (path_risk[i], i)) if candidates else current


class RandomPolicy(Policy):
    """Uniform valid-path baseline with deterministic per-episode randomness."""

    name = "random"

    def reset(self, info: dict) -> None:
        self.rng = np.random.default_rng(int(info["scenario_seed"]))

    def act(self, obs: np.ndarray, info: dict) -> int:
        candidates = _valid_paths(info)
        if not candidates:
            return int(info["current"])
        return int(self.rng.choice(candidates))
