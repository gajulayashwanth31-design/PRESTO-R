import pandas as pd

from presto.eval.report import _ablation_table, _markdown_table, _summary_table


def test_markdown_table_escapes_cells():
    table = _markdown_table(pd.DataFrame([{"name": "risk | policy", "value": 1}]))

    assert r"risk \| policy" in table
    assert "| name | value |" in table


def test_summary_table_uses_seed_means():
    episodes = pd.DataFrame(
        [
            {
                "method": "reactive_dqn",
                "topology": "nsfnet",
                "scenario": "mixed",
                "training_seed": seed,
                "pdr": value,
                "delay_mean": 10.0,
                "recovery_time": 1.0,
                "switches": 2.0,
                "proactive_rate": 0.2,
                "unnecessary_switch_rate": 0.1,
            }
            for seed, value in ((0, 0.8), (1, 1.0))
        ]
    )

    summary = _summary_table(episodes, "nsfnet")
    row = summary[(summary["Method"] == "reactive_dqn") & (summary["Scenario"] == "mixed")]

    assert row.iloc[0]["pdr"] == "0.9000 ± 0.1414"


def test_ablation_marks_unrun_factorial_cell():
    episodes = pd.DataFrame(
        [
            {
                "method": "reactive_dqn",
                "topology": "nsfnet",
                "scenario": "mixed",
                "training_seed": 0,
                "pdr": 0.9,
                "switches": 3,
            }
        ]
    )

    ablation = _ablation_table(episodes)
    missing_cell = ablation[ablation["Method"] == "reactive_dqn_pen"].iloc[0]

    assert missing_cell["PDR (mean over seeds)"] == "not run in the completed P0 grid"
