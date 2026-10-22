import pytest

from presto.reward import RewardWeights, compute_reward


def test_switch_penalty_is_applied_only_on_switch():
    qos = {"delay": 10.0, "loss": 0.1, "delivered": 1.8, "hops": 2}
    weights = RewardWeights(lam=0.25)
    no_switch, terms = compute_reward(qos, False, 10.0, 4, 2.0, weights)
    switched, switched_terms = compute_reward(qos, True, 10.0, 4, 2.0, weights)

    assert switched == pytest.approx(no_switch - 0.25)
    assert terms["S"] == 0.0
    assert switched_terms["S"] == -0.25


def test_lambda_zero_removes_switch_penalty():
    qos = {"delay": 10.0, "loss": 0.1, "delivered": 1.8, "hops": 2}
    no_switch, _ = compute_reward(qos, False, 10.0, 4, 2.0, RewardWeights(lam=0.0))
    switched, _ = compute_reward(qos, True, 10.0, 4, 2.0, RewardWeights(lam=0.0))

    assert switched == pytest.approx(no_switch)


def test_reward_terms_sum_to_reward():
    qos = {"delay": 12.0, "loss": 0.2, "delivered": 1.6, "hops": 3}

    reward, terms = compute_reward(qos, True, 10.0, 4, 2.0, RewardWeights())

    assert sum(terms.values()) == pytest.approx(reward)
