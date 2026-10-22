import json

import numpy as np
import pandas as pd
import pytest

from presto.eval.metrics import compute_all
from presto.eval.run_grid import _apply_evaluation_overrides, build_tasks
from presto.eval.stats import (
    horizon_sweep_summary,
    lambda_sweep_summary,
    paired_tests,
    robustness_summary,
    summary_table,
)


def _steps(seed, scenario="mixed"):
    rows = []
    for tick in range(3):
        rows.append(
            {
                "scenario_seed": seed,
                "scenario": scenario,
                "t": tick,
                "action": 0,
                "switched": tick == 1,
                "valid_chosen": True,
                "path_delay": 2.0,
                "path_loss": 0.0,
                "delivered": 1.0,
                "demand": 1.0,
                "hops": 2,
                "reward": -0.1,
                "path_links": json.dumps([0]),
                "previous_path_links": json.dumps([0]),
                "max_true_health": 1.0,
                "max_risk": 0.0,
                "true_bad_links": "[]",
                "down_links": "[]",
                "n_links": 1,
            }
        )
    return pd.DataFrame(rows)


def test_compute_all_traces_metrics_to_raw_parquet(tmp_path):
    raw = tmp_path / "raw" / "presto_dqn" / "nsfnet" / "mixed"
    raw.mkdir(parents=True)
    _steps(10000).to_parquet(raw / "s0.steps.parquet", index=False)
    pd.DataFrame(
        columns=["scenario_seed", "scenario", "kind", "links", "start", "precursor", "duration"]
    ).to_parquet(raw / "s0.events.parquet", index=False)
    (raw / "s0.config.yaml").write_text(
        "reward:\n  lam: 0.5\npredictor:\n  horizon: 3\n", encoding="utf-8"
    )

    result = compute_all(tmp_path / "raw", tmp_path / "metrics.parquet")

    assert len(result) == 1
    assert result.iloc[0]["method"] == "presto_dqn"
    assert result.iloc[0]["topology"] == "nsfnet"
    assert result.iloc[0]["scenario_seed"] == 10000
    assert result.iloc[0]["raw_steps"].endswith("s0.steps.parquet")
    assert result.iloc[0]["lambda"] == 0.5
    assert result.iloc[0]["horizon"] == 3
    assert (tmp_path / "metrics.parquet").is_file()


def test_summary_table_uses_training_seed_variance_and_ci(tmp_path):
    rows = []
    for training_seed, pdr in enumerate([0.9, 0.92, 0.94]):
        for scenario_seed in range(10000, 10004):
            rows.append(
                {
                    "method": "presto_dqn",
                    "topology": "nsfnet",
                    "scenario": "mixed",
                    "training_seed": str(training_seed),
                    "scenario_seed": scenario_seed,
                    "pdr": pdr,
                    "switches": 4.0,
                    "recovery_time": 2.0,
                }
            )

    tables = summary_table(pd.DataFrame(rows), tmp_path)

    table = tables["nsfnet"]
    assert table.loc[0, "summary_pdr_mixed"].startswith("0.92 ±")
    assert np.isfinite(table.loc[0, "ci95_lower_pdr_mixed"])
    assert np.isfinite(table.loc[0, "ci95_upper_pdr_mixed"])
    assert (tmp_path / "summary_nsfnet.csv").is_file()


def test_paired_tests_compare_shared_test_episodes_and_holm_correct(tmp_path):
    rows = []
    for method, adjustment in [
        ("presto_dqn", 0.05),
        ("reactive_dqn", 0.0),
        ("predictive_dqn", 0.0),
    ]:
        for training_seed in range(2):
            for scenario_seed in range(10000, 10010):
                rows.append(
                    {
                        "method": method,
                        "topology": "nsfnet",
                        "scenario": "mixed",
                        "training_seed": str(training_seed),
                        "scenario_seed": scenario_seed,
                        "pdr": 0.9 + adjustment,
                        "switches": 10.0 - adjustment * 10,
                        "recovery_time": 4.0,
                    }
                )

    results = paired_tests(pd.DataFrame(rows), output=tmp_path / "paired.csv")

    assert len(results) == 6
    assert set(results["n_episodes"]) == {10}
    assert np.all(results["p_holm"] >= results["p_raw"])
    assert (tmp_path / "paired.csv").is_file()


def test_lambda_sweep_summary_keeps_test_lambda_and_metrics(tmp_path):
    rows = []
    for method, lam in [("presto_lam0", 0.0), ("presto_lam05", 0.5)]:
        for seed in range(2):
            rows.append(
                {
                    "method": method,
                    "scenario": "mixed",
                    "lambda": lam,
                    "pdr": 0.95 + lam / 100,
                    "switches": 10 - lam,
                    "unnecessary_switch_rate": 0.2,
                    "recovery_time": 4.0,
                    "proactive_rate": 0.3,
                }
            )

    result = lambda_sweep_summary(pd.DataFrame(rows), tmp_path / "lambda.csv")

    assert set(result["lambda"]) == {0.0, 0.5}
    assert "switches_mean" in result.columns
    assert (tmp_path / "lambda.csv").is_file()


def test_horizon_sweep_summary_joins_predictor_onset_f1(tmp_path):
    rows = []
    for horizon in (3, 10):
        for seed in range(2):
            rows.append(
                {
                    "method": f"presto_h{horizon}",
                    "scenario": "mixed",
                    "training_seed": str(seed),
                    "horizon": horizon,
                    "pdr": 0.95,
                    "switches": 5.0,
                    "proactive_rate": 0.3,
                    "recovery_time": 4.0,
                }
            )
    predictor_files = {}
    for horizon, f1 in ((3, 0.6), (10, 0.7)):
        path = tmp_path / f"predictor_h{horizon}.csv"
        pd.DataFrame([{"model": "gru", "subset": "onset", "group": "all", "f1": f1}]).to_csv(
            path, index=False
        )
        predictor_files[horizon] = path

    result = horizon_sweep_summary(pd.DataFrame(rows), predictor_files, tmp_path / "horizon.csv")

    assert result.set_index("horizon")["predictor_onset_f1"].to_dict() == {
        3: 0.6,
        10: 0.7,
    }
    assert (tmp_path / "horizon.csv").is_file()


def test_robustness_summary_aggregates_seed_level_policy_metrics(tmp_path):
    rows = []
    for method in (
        "reactive_sudden",
        "presto_sudden",
        "reactive_noise2x",
        "presto_noise2x",
        "reactive_load150",
        "presto_load150",
    ):
        for seed in range(3):
            rows.append(
                {
                    "method": method,
                    "training_seed": str(seed),
                    "scenario": "mixed",
                    "pdr": 0.95,
                    "delay_mean": 2.0,
                    "switches": 4.0,
                    "unnecessary_switch_rate": 0.1,
                    "recovery_time": 3.0,
                    "proactive_rate": 0.2,
                }
            )

    result = robustness_summary(pd.DataFrame(rows), tmp_path / "robustness.csv")

    assert set(result["case"]) == {"sudden", "noise2x", "load150"}
    assert set(result["training_seeds"]) == {3}
    assert (tmp_path / "robustness.csv").is_file()


def test_evaluation_overrides_are_applied_only_to_existing_config_keys():
    config = {"measurement": {"delay_noise_rel": 0.05}}

    _apply_evaluation_overrides(config, {"measurement.delay_noise_rel": 0.1})

    assert config["measurement"]["delay_noise_rel"] == 0.1


def test_grid_requires_test_seeds_inside_held_out_range():
    grid = {
        "methods": {"sp_failover": {"type": "heuristic", "cls": "SPFailover"}},
        "scenarios": ["steady"],
        "topologies": ["nsfnet"],
        "training_seeds": [],
        "test_seeds": [5000, 5002],
    }

    with pytest.raises(ValueError, match="held-out range"):
        build_tasks(grid)
