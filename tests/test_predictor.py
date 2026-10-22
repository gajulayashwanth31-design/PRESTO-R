import numpy as np
import pandas as pd
import torch

from presto.config import load_config
from presto.env import RoutingEnv
from presto.predictor.evaluate import _median_lead_time, evaluate_dataset
from presto.predictor.gru import LinkGRU
from presto.predictor.trace import _validation_episodes
from presto.predictor.train import train_gru, train_logreg
from presto.risk import LearnedRisk, LogRegRisk
from presto.sim.simulator import LinkStats
from presto.telemetry import Telemetry


def _synthetic_dataset(path, seed=3):
    rng = np.random.default_rng(seed)
    archive = {}
    for split, count in (("train", 64), ("val", 32), ("test", 32)):
        windows = rng.normal(size=(count, 10, 5)).astype(np.float32)
        summaries = rng.normal(size=(count, 21)).astype(np.float32)
        labels = (windows[:, -1, 0] > 0).astype(np.int8)
        if labels.min() == labels.max():
            labels[0] = 0
            labels[-1] = 1
        archive[f"{split}_windows"] = windows
        archive[f"{split}_summaries"] = summaries
        archive[f"{split}_labels"] = labels
        archive[f"{split}_onset"] = np.ones(count, dtype=bool)
        archive[f"{split}_meta"] = np.column_stack(
            (
                np.arange(count, dtype=np.int32) + {"train": 0, "val": 100, "test": 200}[split],
                np.arange(count, dtype=np.int32),
                np.zeros(count, dtype=np.int32),
            )
        )
        archive[f"{split}_scenario"] = np.full(count, "mixed", dtype="U32")
        archive[f"{split}_topology"] = np.full(count, "nsfnet", dtype="U16")
        archive[f"{split}_failure_type"] = np.where(labels, 1, -1).astype(np.int8)
        archive[f"{split}_bad_now"] = np.zeros(count, dtype=bool)
    np.savez_compressed(path, **archive)


def test_gru_forward_returns_one_logit_per_sample():
    model = LinkGRU(n_features=5, hidden=8, layers=1, dropout=0.0)

    assert model(torch.zeros(6, 10, 5)).shape == (6,)


def test_trace_uses_unique_episodes_from_validation_split(tmp_path):
    dataset = tmp_path / "validation.npz"
    np.savez_compressed(
        dataset,
        val_meta=np.asarray(
            [[250, 10, 0], [250, 11, 0], [251, 10, 0], [252, 10, 0]],
            dtype=np.int32,
        ),
        val_scenario=np.asarray(["mixed", "mixed", "mixed", "congestion"]),
        val_topology=np.asarray(["nsfnet", "nsfnet", "rgg", "nsfnet"]),
    )

    assert _validation_episodes(dataset, "mixed", 2) == [
        (250, "nsfnet"),
        (251, "rgg"),
    ]


def test_median_lead_time_uses_aligned_episode_rows():
    meta = np.asarray([[7, tick, 0] for tick in range(5)], dtype=np.int32)
    probabilities = np.asarray([0.1, 0.8, 0.1, 0.9, 0.9])
    bad_now = np.asarray([False, False, False, True, True])

    lead_time = _median_lead_time(probabilities, 0.5, meta, bad_now, horizon=3)

    assert lead_time == 2.0


def test_train_roundtrip_risk_providers_and_evaluation(tmp_path):
    cfg = load_config(
        "configs/predictor/gru.yaml",
        [
            "predictor.model.hidden=8",
            "predictor.model.dropout=0.0",
            "predictor.train.epochs=1",
            "predictor.train.patience=1",
            "predictor.train.batch=16",
        ],
    )
    dataset = tmp_path / "tiny.npz"
    _synthetic_dataset(dataset)
    gru_path = train_gru(cfg, dataset, tmp_path / "gru.pt", seed=9)
    gru_risk = LearnedRisk(str(gru_path))
    telemetry = Telemetry(3, 10, np.ones(3), 10.0)
    for _ in range(10):
        telemetry.push(
            LinkStats(
                delay=np.ones(3),
                loss=np.zeros(3),
                queue=np.zeros(3),
                util=np.zeros(3),
                up=np.ones(3, dtype=bool),
            )
        )
    first_risk = gru_risk(telemetry, 10)
    second_risk = LearnedRisk(str(gru_path))(telemetry, 10)
    np.testing.assert_array_equal(first_risk, second_risk)
    assert first_risk.shape == (3,)
    assert np.all((first_risk >= 0.0) & (first_risk <= 1.0))

    env_cfg = load_config("configs/scenario/steady.yaml", ["agent.use_risk=true"])
    env = RoutingEnv(env_cfg, risk=LearnedRisk(str(gru_path)))
    observation, info = env.reset(seed=4, options={"scenario_seed": 4, "scenario": "steady"})
    assert observation.shape == env.observation_space.shape
    assert info["risk"].shape == (env.sim.n_links,)
    env.close()

    cfg["predictor"]["type"] = "logreg"
    cfg["predictor"]["model"] = {"C": 1.0, "class_weight": "balanced"}
    logreg_path = train_logreg(cfg, dataset, tmp_path / "logreg.joblib", seed=9)
    logreg_risk = LogRegRisk(str(logreg_path))(telemetry, 10)
    assert logreg_risk.shape == (3,)
    assert np.all((logreg_risk >= 0.0) & (logreg_risk <= 1.0))

    cfg["predictor"]["type"] = "gru"
    metrics = evaluate_dataset(cfg, dataset, gru_path, tmp_path / "evaluation")
    assert metrics.is_file()
    assert (tmp_path / "evaluation" / "calibration.png").is_file()
    cfg["predictor"]["type"] = "logreg"
    evaluate_dataset(cfg, dataset, logreg_path, tmp_path / "evaluation")
    metric_models = set(pd.read_csv(metrics)["model"])
    assert metric_models == {"gru", "logreg"}
    assert (tmp_path / "evaluation" / "calibration_gru.png").is_file()
    assert (tmp_path / "evaluation" / "calibration_logreg.png").is_file()
