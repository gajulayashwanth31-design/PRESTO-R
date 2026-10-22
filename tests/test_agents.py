import numpy as np

from presto.agents.train import FixedSeedPool, TrainSeedPool


def test_train_seed_pool_does_not_repeat_until_exhausted():
    pool = TrainSeedPool(5, ["congestion", "mixed"])
    rng = np.random.default_rng(17)

    episodes = [pool(rng) for _ in range(5)]

    assert len({seed for seed, _ in episodes}) == 5
    assert all(scenario in {"congestion", "mixed"} for _, scenario in episodes)


def test_fixed_seed_pool_cycles_validation_seeds_and_scenarios():
    pool = FixedSeedPool(5000, 5002, ["congestion", "mixed"])
    rng = np.random.default_rng(4)

    episodes = [pool(rng) for _ in range(5)]

    assert episodes == [
        (5000, "congestion"),
        (5001, "mixed"),
        (5000, "congestion"),
        (5001, "mixed"),
        (5000, "congestion"),
    ]
