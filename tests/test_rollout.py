import json

import pandas as pd
import pytest

from presto.config import load_config
from presto.env import RoutingEnv
from presto.eval.metrics import rollout_metrics
from presto.eval.rollout import rollout
from presto.policies import SPFailover


def test_rollout_is_deterministic_and_writes_parquet(tmp_path):
    cfg = load_config("configs/scenario/steady.yaml")
    cfg["sim"]["episode_len"] = 30
    output = tmp_path / "fixed_seeds"
    steps_a, events_a = rollout(SPFailover(), cfg, [5000], out_path=output)
    steps_b, events_b = rollout(SPFailover(), cfg, [5000])

    pd.testing.assert_frame_equal(steps_a, steps_b)
    pd.testing.assert_frame_equal(events_a, events_b)
    assert output.with_suffix(".steps.parquet").is_file()
    assert output.with_suffix(".events.parquet").is_file()
    assert len(rollout_metrics(steps_a, events_a)) == 1


def test_rollout_event_records_match_simulator_generator():
    cfg = load_config("configs/scenario/mixed.yaml")
    cfg["sim"]["episode_len"] = 40
    env = RoutingEnv(cfg)
    try:
        env.reset(seed=5000, options={"scenario_seed": 5000, "scenario": "mixed"})
        expected = [
            (event.kind, sorted(set(event.links)), event.start, event.precursor, event.duration)
            for event in env.sim.events
        ]
    finally:
        env.close()

    _, events = rollout(SPFailover(), cfg, [5000], scenario="mixed")
    actual = [
        (row.kind, sorted(json.loads(row.links)), row.start, row.precursor, row.duration)
        for row in events.itertuples(index=False)
    ]

    assert actual == expected


def test_rollout_rejects_training_seeds_unless_explicitly_allowed():
    cfg = load_config("configs/scenario/steady.yaml")
    with pytest.raises(ValueError, match="validation or test"):
        rollout(SPFailover(), cfg, [7])
