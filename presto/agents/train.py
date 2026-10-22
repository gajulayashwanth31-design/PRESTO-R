"""Train Stable-Baselines3 routing agents with fixed-seed validation."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from presto.config import load_config, save_config
from presto.env import RoutingEnv
from presto.risk import LearnedRisk, LogRegRisk, OracleRisk
from presto.seeding import seed_everything


class TrainSeedPool:
    """Sample reproducible fresh training seeds and scenario types per episode."""

    def __init__(self, seed_max: int, scenarios: list[str]):
        if seed_max < 1 or not scenarios:
            raise ValueError("Training requires a positive seed range and scenarios")
        self.seed_max = int(seed_max)
        self.scenarios = tuple(scenarios)
        self.available = np.arange(self.seed_max, dtype=np.int32)
        self.remaining = self.seed_max

    def __call__(self, rng: np.random.Generator) -> tuple[int, str]:
        if self.remaining == 0:
            self.available = np.arange(self.seed_max, dtype=np.int32)
            self.remaining = self.seed_max
        index = int(rng.integers(self.remaining))
        seed = int(self.available[index])
        self.remaining -= 1
        self.available[index], self.available[self.remaining] = (
            self.available[self.remaining],
            self.available[index],
        )
        return seed, str(rng.choice(self.scenarios))


class FixedSeedPool:
    """Cycle through fixed validation seeds and scenario labels."""

    def __init__(self, low: int, high: int, scenarios: list[str]):
        if low >= high or not scenarios:
            raise ValueError("Validation seed range and scenario list must be non-empty")
        self.seeds = tuple(range(int(low), int(high)))
        self.scenarios = tuple(scenarios)
        self.index = 0

    def __call__(self, rng: np.random.Generator) -> tuple[int, str]:
        del rng
        index = self.index
        self.index += 1
        return self.seeds[index % len(self.seeds)], self.scenarios[index % len(self.scenarios)]


class SwitchLogger(BaseCallback):
    """Write per-episode switching and delivery summaries during training."""

    def __init__(self, output: Path):
        super().__init__(verbose=0)
        self.output = output
        self.switches: list[int] = []
        self.delivered: list[float] = []
        self.demand: list[float] = []
        self.episode = 0

    def _on_training_start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                ["timesteps", "episode", "switches", "pdr", "scenario_seed", "scenario"]
            )
        count = int(self.training_env.num_envs)
        self.switches = [0] * count
        self.delivered = [0.0] * count
        self.demand = [0.0] * count

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for index, info in enumerate(infos):
            self.switches[index] += int(bool(info.get("switched", False)))
            qos = info.get("qos", {})
            self.delivered[index] += float(qos.get("delivered", 0.0))
            self.demand[index] += float(info.get("demand", 0.0))
            if index < len(dones) and dones[index]:
                pdr = (
                    self.delivered[index] / self.demand[index]
                    if self.demand[index]
                    else float("nan")
                )
                with self.output.open("a", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerow(
                        [
                            self.num_timesteps,
                            self.episode,
                            self.switches[index],
                            pdr,
                            info.get("scenario_seed", ""),
                            info.get("scenario", ""),
                        ]
                    )
                self.episode += 1
                self.switches[index] = 0
                self.delivered[index] = 0.0
                self.demand[index] = 0.0
        return True


def build_risk(cfg: dict):
    agent_cfg = cfg.get("agent", {})
    if not bool(agent_cfg.get("use_risk", False)):
        return None
    risk_model = agent_cfg.get("risk_model")
    if isinstance(risk_model, dict):
        model_type = str(risk_model.get("type", "gru"))
        model_path = risk_model.get("path")
        horizon = int(risk_model.get("horizon", cfg.get("predictor", {}).get("horizon", 5)))
    else:
        model_path = risk_model
        model_type = (
            str(model_path)
            if str(model_path) == "oracle"
            else ("logreg" if str(model_path or "").endswith(".joblib") else "gru")
        )
        horizon = int(cfg.get("predictor", {}).get("horizon", 5))
    if model_type == "oracle":
        return OracleRisk(horizon)
    if model_path is None:
        predictor_type = model_type
        extension = "joblib" if predictor_type == "logreg" else "pt"
        model_path = (
            Path(cfg["paths"]["models"])
            / "predictor"
            / f"{predictor_type}_W{cfg['predictor']['window']}_"
            f"H{cfg['predictor']['horizon']}.{extension}"
        )
    if model_type == "gru":
        return LearnedRisk(str(model_path))
    if model_type == "logreg":
        return LogRegRisk(str(model_path))
    raise ValueError(f"Unsupported risk model type: {model_type}")


def train_agent(cfg: dict, seed: int) -> Path:
    agent_cfg = cfg["agent"]
    algo = str(agent_cfg.get("algo", "dqn")).lower()
    algorithms = {"dqn": DQN, "ppo": PPO}
    if algo not in algorithms:
        raise ValueError(f"Unsupported training algorithm: {algo}")
    algorithm_cfg_key = "dqn" if algo == "dqn" else "ppo"
    if algorithm_cfg_key not in agent_cfg:
        raise ValueError(f"Missing agent.{algorithm_cfg_key} training settings")
    seed_everything(seed)

    name = str(agent_cfg["name"])
    run_dir = Path(cfg["paths"]["runs"]) / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    risk = build_risk(cfg)
    train_env = Monitor(
        RoutingEnv(
            cfg,
            risk=risk,
            seed_pool=TrainSeedPool(
                int(cfg["seeds"]["train_max"]), list(agent_cfg["train_scenarios"])
            ),
        )
    )
    val_scenarios = list(agent_cfg.get("eval_scenarios", [cfg["scenario"]["name"]]))
    validation_seeds = cfg["seeds"]["val"]
    eval_env = Monitor(
        RoutingEnv(
            cfg,
            risk=build_risk(cfg),
            seed_pool=FixedSeedPool(
                int(validation_seeds[0]), int(validation_seeds[1]), val_scenarios
            ),
        )
    )
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=int(agent_cfg["eval_episodes"]),
        eval_freq=max(1, int(agent_cfg["eval_every"])),
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        deterministic=True,
        verbose=0,
    )
    switch_logger = SwitchLogger(run_dir / "log.csv")
    model = algorithms[algo](
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=0,
        policy_kwargs=agent_cfg.get("policy_kwargs", {}),
        **agent_cfg[algorithm_cfg_key],
    )
    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=int(agent_cfg["total_steps"]),
            callback=CallbackList([eval_callback, switch_logger]),
        )
        best_model = run_dir / "best_model.zip"
        if not best_model.is_file():
            model.save(best_model)
        save_config(cfg, run_dir / "config.yaml")
        metadata = {
            "seed": seed,
            "method": name,
            "training_steps": int(agent_cfg["total_steps"]),
            "training_seconds": time.perf_counter() - started,
            "best_validation_model": best_model.name,
        }
        (run_dir / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        train_env.close()
        eval_env.close()
    return run_dir / "best_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent/reactive_dqn.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    output = train_agent(cfg, args.seed)
    print(f"Saved best validation model to {output}")


if __name__ == "__main__":
    main()
