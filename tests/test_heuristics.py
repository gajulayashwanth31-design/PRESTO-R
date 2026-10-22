import numpy as np
import pytest

from presto.policies import GreedyDelay, RandomPolicy, RiskThreshold, SPFailover


@pytest.fixture
def policy_info():
    return {
        "current": 0,
        "paths": [[0, 1], [2], [3, 4]],
        "valid": np.asarray([True, True, False]),
        "obs_link": np.asarray(
            [
                [2.0, 0.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "risk": np.asarray([0.1, 0.2, 0.4, 0.8, 0.9]),
        "scenario_seed": 12,
    }


def test_sp_failover_stays_on_valid_shortest_path_and_moves_to_first_valid(policy_info):
    policy = SPFailover()

    assert policy.act(np.zeros(1), policy_info) == 0
    policy_info["valid"] = np.asarray([False, True, True])
    assert policy.act(np.zeros(1), policy_info) == 1


def test_greedy_delay_picks_minimum_observed_delay_among_valid_paths(policy_info):
    assert GreedyDelay().act(np.zeros(1), policy_info) == 1


def test_risk_threshold_stays_below_threshold_and_leaves_above(policy_info):
    policy = RiskThreshold(tau=0.3)
    assert policy.act(np.zeros(1), policy_info) == 0
    policy_info["risk"] = np.asarray([0.8, 0.2, 0.4, 0.8, 0.9])
    assert policy.act(np.zeros(1), policy_info) == 1


def test_risk_threshold_validates_tau():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RiskThreshold(1.1)


def test_random_policy_is_reproducible_per_episode(policy_info):
    policy = RandomPolicy()
    policy.reset(policy_info)
    first = [policy.act(np.zeros(1), policy_info) for _ in range(10)]
    policy.reset(policy_info)
    second = [policy.act(np.zeros(1), policy_info) for _ in range(10)]

    assert first == second
