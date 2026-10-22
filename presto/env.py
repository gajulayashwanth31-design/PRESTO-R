"""Gymnasium routing environment over the reusable fluid network simulator."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from presto.config import load_config
from presto.reward import RewardWeights, compute_reward
from presto.risk import RiskProvider, ZeroRisk
from presto.seeding import make_rng
from presto.sim.simulator import LinkStats, NetworkSimulator
from presto.state import build_observation, choose_flow
from presto.telemetry import Telemetry


class RoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict,
        risk: RiskProvider | None = None,
        seed_pool: Any = None,
    ):
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.risk_provider = risk or ZeroRisk()
        self.seed_pool = seed_pool
        reward_cfg = self.cfg.get("reward", {})
        self.reward_weights = RewardWeights(
            **{
                key: reward_cfg[key]
                for key in ("alpha", "beta", "gamma", "lam", "mu")
                if key in reward_cfg
            }
        )
        self.use_risk = bool(self.cfg.get("agent", {}).get("use_risk", True))
        self.k = int(self.cfg["paths"]["k"])
        if self.k < 1:
            raise ValueError("paths.k must be positive")
        obs_dim = self.k * (8 if self.use_risk else 6) + self.k + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.k)
        self._seed_iterator = (
            iter(seed_pool) if seed_pool is not None and not callable(seed_pool) else None
        )
        self.sim: NetworkSimulator | None = None
        self.telemetry: Telemetry | None = None
        self.paths: list[list[int]] = []
        self.current = 0
        self.steps_since_switch = 0
        self.ref_delay = 1.0
        self.max_hops = 1
        self.scenario_seed = 0
        self.scenario_name = str(self.cfg["scenario"]["name"])
        self._latest_observed: LinkStats | None = None
        self._last_true_stats: LinkStats | None = None

    def _pick_episode(self, options: dict | None) -> tuple[int, str]:
        if options is not None and ("scenario_seed" in options or "scenario" in options):
            maximum = int(self.cfg.get("seeds", {}).get("train_max", 5000))
            episode_seed = int(options.get("scenario_seed", self.np_random.integers(0, maximum)))
            scenario = str(options.get("scenario", self.cfg["scenario"]["name"]))
        elif self._seed_iterator is not None:
            try:
                episode_seed, scenario = next(self._seed_iterator)
            except StopIteration as exc:
                raise RuntimeError("The evaluation seed_pool is exhausted") from exc
            episode_seed, scenario = int(episode_seed), str(scenario)
        elif callable(self.seed_pool):
            episode_seed, scenario = self.seed_pool(self.np_random)
            episode_seed, scenario = int(episode_seed), str(scenario)
        else:
            maximum = int(self.cfg.get("seeds", {}).get("train_max", 5000))
            episode_seed = int(self.np_random.integers(0, maximum))
            scenario = str(self.cfg["scenario"]["name"])
        return episode_seed, scenario

    def _episode_cfg(self, scenario: str) -> dict:
        episode_cfg = deepcopy(self.cfg)
        scenario_path = Path("configs") / "scenario" / f"{scenario}.yaml"
        if scenario_path.is_file():
            scenario_cfg = load_config(scenario_path)
            episode_cfg["scenario"] = scenario_cfg["scenario"]
        else:
            known = self.cfg["scenario"].get("name")
            if scenario != known:
                raise ValueError(f"No configuration is available for scenario {scenario!r}")
        return episode_cfg

    def _latest_matrix(self) -> np.ndarray:
        if self._latest_observed is None or self.telemetry is None:
            raise RuntimeError("Environment has no observations; call reset first")
        return np.column_stack(
            (
                self._latest_observed.delay,
                self._latest_observed.loss,
                self._latest_observed.queue / self.sim.qmax,
                self._latest_observed.util,
                self._latest_observed.up.astype(np.float64),
            )
        )

    def _risk(self) -> np.ndarray:
        if self.telemetry is None or self.sim is None:
            raise RuntimeError("Environment has not been reset")
        risk = np.asarray(self.risk_provider(self.telemetry, self.sim.t - 1), dtype=np.float64)
        if risk.shape != (self.sim.n_links,):
            raise ValueError("Risk provider must return one value per link")
        if np.any(~np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
            raise ValueError("Risk provider values must be finite and within [0, 1]")
        return risk

    def _observation(self, risk: np.ndarray, latest: np.ndarray | None = None) -> np.ndarray:
        return build_observation(
            latest=self._latest_matrix() if latest is None else latest,
            risk=risk,
            paths=self.paths,
            current=self.current,
            steps_since_switch=self.steps_since_switch,
            ref_delay=self.ref_delay,
            max_hops=self.max_hops,
            use_risk=self.use_risk,
            k=self.k,
        )

    def _info(self, risk: np.ndarray, latest: np.ndarray | None = None, **extra: Any) -> dict:
        latest = self._latest_matrix() if latest is None else latest
        valid = np.asarray(
            [bool(path) and bool(np.all(latest[path, 4] > 0.5)) for path in self.paths],
            dtype=bool,
        )
        info = {
            "t": self.sim.t - 1,
            "action": self.current,
            "switched": False,
            "qos": {},
            "reward_terms": {},
            "path_links": list(self.paths[self.current]),
            "scenario_seed": self.scenario_seed,
            "scenario": self.scenario_name,
            "demand": float(self.sim.managed_rate),
            "paths": [list(path) for path in self.paths],
            "current": self.current,
            "valid": valid,
            "obs_link": latest,
            "risk": risk,
            "events": self.sim.events,
        }
        info.update(extra)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.scenario_seed, self.scenario_name = self._pick_episode(options)
        episode_cfg = self._episode_cfg(self.scenario_name)
        self.sim = NetworkSimulator(episode_cfg, self.scenario_seed)
        flow_rng = make_rng(self.scenario_seed, "flow")
        src, dst, self.paths = choose_flow(self.sim.graph, flow_rng, self.k, self.sim.edge_ids)
        candidate_links = {link for path in self.paths for link in path}
        self.sim.reset(src, dst, candidate_links)
        self.risk_provider.reset(self.sim)
        self.telemetry = Telemetry(
            self.sim.n_links,
            int(episode_cfg["sim"]["warmup"]),
            self.sim.base_delay,
            self.sim.qmax,
        )
        self.telemetry.reset()
        self._latest_observed = None
        self._last_true_stats = None
        self.current = 0
        self.steps_since_switch = int(episode_cfg["sim"]["warmup"])
        self.ref_delay = float(sum(self.sim.base_delay[link] for link in self.paths[0]))
        self.max_hops = max(map(len, self.paths))

        for _ in range(int(episode_cfg["sim"]["warmup"])):
            stats = self.sim.step(self.paths[0])
            self._last_true_stats = stats
            self._latest_observed = self.sim.observe(stats)
            self.telemetry.push(self._latest_observed)

        risk = self._risk()
        latest = self._latest_matrix()
        return self._observation(risk, latest), self._info(risk, latest)

    def step(self, action: int):
        if self.sim is None or self.telemetry is None:
            raise RuntimeError("Call reset before step")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space")
        previous = self.current
        switched = action != previous
        self.current = action
        stats = self.sim.step(self.paths[action])
        self._last_true_stats = stats
        qos = self.sim.path_qos(stats, self.paths[action])
        transient_loss = float(self.cfg.get("reward", {}).get("switch_transient_loss", 0.0))
        if switched and transient_loss:
            qos["loss"] = min(1.0, qos["loss"] + transient_loss)
            qos["delivered"] = self.sim.managed_rate * (1.0 - qos["loss"])
        self._latest_observed = self.sim.observe(stats)
        self.telemetry.push(self._latest_observed)

        if switched:
            self.steps_since_switch = 0
        else:
            self.steps_since_switch += 1
        reward, reward_terms = compute_reward(
            qos,
            switched,
            self.ref_delay,
            self.max_hops,
            self.sim.managed_rate,
            self.reward_weights,
        )
        risk = self._risk()
        latest = self._latest_matrix()
        info = self._info(
            risk,
            latest=latest,
            action=action,
            switched=switched,
            qos=qos,
            reward_terms=reward_terms,
            path_links=list(self.paths[action]),
        )
        truncated = self.sim.t >= int(self.cfg["sim"]["warmup"]) + int(
            self.cfg["sim"]["episode_len"]
        )
        return self._observation(risk, latest), reward, False, truncated, info

    def close(self) -> None:
        self.sim = None
        self.telemetry = None
        self._last_true_stats = None
