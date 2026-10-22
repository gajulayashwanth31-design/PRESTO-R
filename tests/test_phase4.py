from pathlib import Path

import pandas as pd
import pytest

from presto.agents.train import build_risk, train_agent
from presto.config import load_config
from presto.eval.select_lambda import _choose_lambda
from presto.risk import LearnedRisk, OracleRisk


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/agent/reactive_dqn.yaml",
        "configs/agent/reactive_dqn_pen.yaml",
        "configs/agent/predictive_dqn.yaml",
        "configs/agent/presto_dqn.yaml",
        "configs/agent/presto_oracle.yaml",
        "configs/agent/presto_ppo.yaml",
    ],
)
def test_agent_configs_load_and_keep_expected_ablation_settings(config_path):
    cfg = load_config(config_path)

    assert cfg["agent"]["algo"] in {"dqn", "ppo"}
    if cfg["agent"]["name"] == "reactive_dqn":
        assert cfg["agent"]["use_risk"] is False
    if cfg["agent"]["algo"] == "dqn":
        assert cfg["agent"]["dqn"] == load_config("configs/agent/reactive_dqn.yaml")["agent"]["dqn"]
    if cfg["agent"]["name"] == "predictive_dqn":
        assert cfg["reward"]["lam"] == 0.0
        assert cfg["agent"]["use_risk"] is True
    if cfg["agent"]["name"] == "presto_ppo":
        assert cfg["agent"]["ppo"]["n_steps"] == 2048


def test_build_risk_supports_learned_and_oracle_providers():
    learned_cfg = load_config("configs/agent/presto_dqn.yaml")
    oracle_cfg = load_config("configs/agent/presto_oracle.yaml")

    assert isinstance(build_risk(learned_cfg), LearnedRisk)
    oracle = build_risk(oracle_cfg)
    assert isinstance(oracle, OracleRisk)
    assert oracle.horizon == 5


def test_lambda_selection_uses_largest_candidate_within_one_pdr_point():
    summary = pd.DataFrame(
        {
            "lambda": [0.0, 0.05, 0.1, 0.5],
            "pdr_mean": [0.97, 0.965, 0.96, 0.959],
        }
    )

    assert _choose_lambda(summary) == 0.1


@pytest.mark.parametrize(
    "config_path", ["configs/agent/presto_dqn.yaml", "configs/agent/presto_ppo.yaml"]
)
def test_dqn_and_ppo_train_smoke(config_path, tmp_path):
    cfg = load_config(
        config_path,
        [
            "agent.total_steps=2048",
            "agent.eval_every=2048",
            "agent.eval_episodes=2",
            "agent.dqn.learning_starts=16",
            "agent.dqn.buffer_size=1000",
            "agent.dqn.batch_size=8",
            "agent.dqn.target_update_interval=128",
            "agent.ppo.n_steps=64",
            "agent.ppo.batch_size=32",
            "agent.ppo.n_epochs=1",
            "sim.episode_len=30",
        ],
    )
    cfg["paths"]["runs"] = str(tmp_path)

    model_path = train_agent(cfg, seed=71)

    assert model_path.is_file()
    assert (model_path.parent / "training.json").is_file()
    assert Path(model_path.parent / "best_model.zip").stat().st_size > 0
