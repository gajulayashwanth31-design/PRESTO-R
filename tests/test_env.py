import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from presto.config import load_config
from presto.env import RoutingEnv


@pytest.mark.parametrize("use_risk", [False, True])
def test_gymnasium_checker(use_risk):
    cfg = load_config("configs/scenario/steady.yaml", [f"agent.use_risk={str(use_risk).lower()}"])

    check_env(RoutingEnv(cfg), skip_render_check=True)


def test_episode_truncates_after_configured_steps():
    cfg = load_config("configs/scenario/steady.yaml")
    cfg["sim"]["episode_len"] = 40
    env = RoutingEnv(cfg)
    env.reset(seed=1, options={"scenario_seed": 2, "scenario": "steady"})

    for tick in range(40):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert truncated == (tick == 39)
    env.close()


def test_same_seed_and_actions_are_deterministic():
    cfg = load_config("configs/scenario/mixed.yaml", ["agent.use_risk=false"])

    def run():
        env = RoutingEnv(cfg)
        env.reset(seed=5, options={"scenario_seed": 21, "scenario": "mixed"})
        result = []
        for action in [0, 1, 1, 2, 0]:
            obs, reward, _, _, info = env.step(action)
            result.append((obs.copy(), reward, info["qos"]["delay"], info["qos"]["loss"]))
        env.close()
        return result

    first, second = run(), run()
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]


def test_risk_provider_is_included_and_validated():
    cfg = load_config("configs/scenario/steady.yaml", ["agent.use_risk=true"])
    env = RoutingEnv(cfg)
    _, info = env.reset(seed=0, options={"scenario_seed": 4, "scenario": "steady"})

    assert info["risk"].shape == (env.sim.n_links,)
    assert np.all(info["risk"] == 0)
    with pytest.raises(ValueError, match="outside the action space"):
        env.step(env.k)
    env.close()


def test_scenario_can_be_selected_without_explicit_seed():
    cfg = load_config("configs/scenario/mixed.yaml")
    env = RoutingEnv(cfg)

    _, info = env.reset(seed=3, options={"scenario": "steady"})

    assert info["scenario"] == "steady"
    assert info["scenario_seed"] >= 0
    assert not env.sim.events
    env.close()


def test_rgg_topology_runs_in_the_environment():
    cfg = load_config("configs/scenario/mixed.yaml", ["agent.use_risk=false"])
    cfg["topology"] = load_config("configs/topo/rgg.yaml")["topology"]
    env = RoutingEnv(cfg)

    observation, info = env.reset(seed=13, options={"scenario_seed": 13, "scenario": "steady"})
    next_observation, _, _, _, _ = env.step(0)

    assert env.sim.graph.number_of_nodes() == 24
    assert observation.shape == next_observation.shape == env.observation_space.shape
    assert info["obs_link"].shape == (env.sim.n_links, 5)
    env.close()
