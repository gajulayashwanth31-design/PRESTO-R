"""Adapter exposing a Stable-Baselines3 model through the policy interface."""

import numpy as np
from stable_baselines3 import DQN, PPO

from presto.policies.base import Policy


class SB3Policy(Policy):
    def __init__(self, model_path: str, algo: str = "dqn", name: str = "dqn"):
        algorithms = {"dqn": DQN, "ppo": PPO}
        try:
            model_class = algorithms[algo.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported SB3 algorithm: {algo}") from exc
        self.model = model_class.load(model_path, device="cpu")
        self.name = name

    def act(self, obs: np.ndarray, info: dict) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
