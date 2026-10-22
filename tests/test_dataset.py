import json

import numpy as np

from presto.config import load_config
from presto.predictor.dataset import build_dataset
from presto.predictor.features import (
    bad_link_mask,
    build_future_labels,
    sample_ticks,
)
from presto.sim.simulator import LinkStats
from presto.telemetry import Telemetry


def test_history_and_future_label_ticks_do_not_overlap():
    window = 4
    horizon = 3
    tick = 8
    history_ticks = np.arange(tick - window + 1, tick + 1)
    future_ticks = np.arange(tick + 1, tick + horizon + 1)

    assert history_ticks.tolist() == [5, 6, 7, 8]
    assert future_ticks.tolist() == [9, 10, 11]
    assert set(history_ticks).isdisjoint(future_ticks)

    telemetry = Telemetry(1, window, np.array([1.0]), 10.0)
    for source_tick in range(9):
        telemetry.push(
            LinkStats(
                delay=np.array([source_tick + 1.0]),
                loss=np.zeros(1),
                queue=np.zeros(1),
                util=np.zeros(1),
                up=np.ones(1, dtype=bool),
            )
        )
    bad = np.zeros((12, 1), dtype=bool)
    bad[future_ticks, 0] = True
    labels, onset = build_future_labels(bad, np.array([tick]), horizon)

    np.testing.assert_array_equal(telemetry.window()[0, :, 0], [6.0, 7.0, 8.0, 9.0])
    assert labels[0, 0]
    assert onset[0, 0]


def test_future_labels_use_open_left_closed_right_interval():
    bad = np.zeros((12, 2), dtype=bool)
    bad[5, 0] = True
    bad[8, 1] = True
    ticks = np.array([2, 3, 4, 5, 6], dtype=np.int32)

    labels, onset = build_future_labels(bad, ticks, horizon=3)

    np.testing.assert_array_equal(labels[:, 0], [True, True, True, False, False])
    np.testing.assert_array_equal(labels[:, 1], [False, False, False, True, True])
    np.testing.assert_array_equal(onset[:, 0], [True, True, True, False, True])


def test_bad_mask_uses_true_loss_delay_and_availability():
    up = np.array([[True, True, False]])
    loss = np.array([[0.01, 0.06, 0.0]])
    delay = np.array([[5.0, 4.0, 2.0]])

    bad = bad_link_mask(up, loss, delay, np.array([2.0, 2.0, 2.0]), 0.05, 3.0)

    np.testing.assert_array_equal(bad, [[False, True, True]])


def test_sample_ticks_keep_full_label_horizon():
    ticks = sample_ticks(total_ticks=15, warmup=5, horizon=3)

    assert ticks.tolist() == list(range(5, 12))
    assert ticks[-1] + 3 == 14


def test_dataset_splits_by_episode_and_subsamples_train_negatives(tmp_path):
    cfg = load_config(
        "configs/predictor/gru.yaml",
        [
            "predictor.data.scenario_mix=[steady]",
            "predictor.data.topologies=[nsfnet]",
            "predictor.data.episodes_per_combo=6",
            "predictor.data.neg_subsample=0.2",
            "predictor.data.split_by_episode=[0.5,0.25,0.25]",
        ],
    )
    cfg["sim"]["episode_len"] = 30
    path = build_dataset(cfg, tmp_path / "sample.npz", seed=17)
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

    with np.load(path, allow_pickle=False) as data:
        train_seeds = set(data["train_meta"][:, 0])
        val_seeds = set(data["val_meta"][:, 0])
        test_seeds = set(data["test_meta"][:, 0])
        assert train_seeds.isdisjoint(val_seeds)
        assert train_seeds.isdisjoint(test_seeds)
        assert val_seeds.isdisjoint(test_seeds)
        train_raw = report["raw_class_balance"]["train"]
        val_raw = report["raw_class_balance"]["val"]
        test_raw = report["raw_class_balance"]["test"]
        assert np.count_nonzero(data["train_labels"]) == train_raw["positive"]
        assert (
            data["train_labels"].size - np.count_nonzero(data["train_labels"])
            < train_raw["negative"]
        )
        assert data["val_labels"].size == report["raw_class_balance"]["val"]["samples"]
        assert data["test_labels"].size == report["raw_class_balance"]["test"]["samples"]
        assert np.count_nonzero(data["val_labels"]) == val_raw["positive"]
        assert np.count_nonzero(data["test_labels"]) == test_raw["positive"]
        assert data["train_windows"].shape[1:] == (10, 5)
        assert data["train_summaries"].shape[1] == 21
