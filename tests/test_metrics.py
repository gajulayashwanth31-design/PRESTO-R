import json

import numpy as np
import pandas as pd

from presto.eval.metrics import episode_metrics


def make_steps(ticks, paths, switched=None, losses=None):
    switched = switched or {}
    losses = losses or {}
    rows = []
    previous = [0]
    for tick, path in zip(ticks, paths, strict=True):
        rows.append(
            {
                "scenario_seed": 5000,
                "t": tick,
                "switched": tick in switched,
                "path_links": json.dumps(path),
                "previous_path_links": json.dumps(previous),
                "path_loss": losses.get(tick, 0.0),
                "path_delay": 2.0,
                "delivered": 1.0 - losses.get(tick, 0.0),
                "demand": 1.0,
                "hops": len(path),
                "reward": -0.1,
                "valid_chosen": losses.get(tick, 0.0) < 1.0,
                "n_links": 2,
                "true_bad_links": "[]",
                "down_links": "[]",
            }
        )
        previous = path
    return pd.DataFrame(rows)


def test_proactive_switch_away_from_failing_path_is_necessary_and_zero_recovery():
    steps = make_steps([10, 11, 12, 13], [[1], [1], [1], [1]], switched={10})
    up = np.ones((14, 2), dtype=bool)
    up[12:, 0] = False

    metrics = episode_metrics(steps, pd.DataFrame(), up, H=2)

    assert metrics["unnecessary_switch_rate"] == 0.0
    assert metrics["proactive_rate"] == 1.0
    assert metrics["recovery_time"] == 0.0


def test_switch_away_from_a_healthy_path_is_unnecessary():
    steps = make_steps([10, 11, 12], [[1], [1], [1]], switched={10})
    up = np.ones((13, 2), dtype=bool)

    metrics = episode_metrics(steps, pd.DataFrame(), up, H=2)

    assert metrics["unnecessary_switch_rate"] == 1.0
    assert np.isnan(metrics["recovery_time"])


def test_recovery_time_is_four_ticks_after_failure():
    ticks = list(range(10, 19))
    paths = [[0]] * 6 + [[1]] * 3
    steps = make_steps(ticks, paths, switched={16}, losses={12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0})
    up = np.ones((19, 2), dtype=bool)
    up[12:, 0] = False

    metrics = episode_metrics(steps, pd.DataFrame(), up, H=2)

    assert metrics["proactive_rate"] == 0.0
    assert metrics["recovery_time"] == 4.0


def test_no_switches_reports_nan_unnecessary_switch_rate():
    steps = make_steps([10, 11], [[0], [0]])

    metrics = episode_metrics(steps, pd.DataFrame(), H=2)

    assert np.isnan(metrics["unnecessary_switch_rate"])
