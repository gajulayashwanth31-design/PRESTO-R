"""Render Phase 5 result artifacts into a reproducible Markdown report."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from presto.config import load_config

REPORT_PATH = Path("results/report.md")
SCENARIOS = ("steady", "congestion", "node_failure", "fluctuation", "mixed")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = [str(column) for column in frame.columns]

    def cell(value) -> str:
        if pd.isna(value):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _parameter_table() -> pd.DataFrame:
    base = load_config("configs/base.yaml")
    predictor = load_config("configs/predictor/gru.yaml")
    agent = load_config("configs/agent/presto_dqn.yaml")
    topology = load_config("configs/topo/nsfnet.yaml")
    return pd.DataFrame(
        [
            ("Primary topology", "NSFNET"),
            (
                "Episode length (ticks)",
                base["sim"]["episode_len"],
            ),
            (
                "Link capacity / base-delay range (ms)",
                f"{topology['topology']['capacity']} / {topology['topology']['base_delay_ms']}",
            ),
            ("Candidate paths", agent["paths"]["k"]),
            (
                "Predictor window W / horizon H",
                f"{predictor['predictor']['window']} / {predictor['predictor']['horizon']}",
            ),
            ("Switch penalty λ", agent["reward"]["lam"]),
            ("Agent training steps", agent["agent"]["total_steps"]),
            ("Training seeds", "0–4 (NSFNET); 0–2 (RGG)"),
            ("Held-out test seeds", f"{base['seeds']['test'][0]}–{base['seeds']['test'][1] - 1}"),
            ("Scenarios", ", ".join(SCENARIOS)),
        ],
        columns=["Parameter", "Value"],
    )


def _summary_table(episodes: pd.DataFrame, topology: str) -> pd.DataFrame:
    methods = ("sp_failover", "reactive_dqn", "predictive_dqn", "presto_dqn")
    metrics = (
        "pdr",
        "delay_mean",
        "recovery_time",
        "switches",
        "proactive_rate",
        "unnecessary_switch_rate",
    )
    selected = episodes[
        (episodes["topology"] == topology)
        & episodes["method"].isin(methods)
        & episodes["scenario"].isin(SCENARIOS)
    ]
    seed_means = selected.groupby(["method", "scenario", "training_seed"], as_index=False)[
        list(metrics)
    ].mean()
    records = []
    for method in methods:
        for scenario in SCENARIOS:
            group = seed_means[
                (seed_means["method"] == method) & (seed_means["scenario"] == scenario)
            ]
            record = {"Method": method, "Scenario": scenario}
            for metric in metrics:
                values = group[metric].dropna()
                mean = values.mean() if not values.empty else np.nan
                std = values.std() if len(values) > 1 else np.nan
                record[metric] = (
                    f"{mean:.4f} ± {std:.4f}"
                    if np.isfinite(mean) and np.isfinite(std)
                    else (f"{mean:.4f} ± NA" if np.isfinite(mean) else "not available")
                )
            records.append(record)
    return pd.DataFrame(records)


def _predictor_table(path: str | Path = "results/predictor/metrics.csv") -> pd.DataFrame:
    metrics = pd.read_csv(path)
    selected = metrics[
        metrics["subset"].isin(["all", "onset"])
        & metrics["group"].isin(["all", "failure:precursor", "failure:sudden"])
    ]
    columns = [
        column
        for column in (
            "model",
            "subset",
            "group",
            "f1",
            "pr_auc",
            "brier",
            "median_lead_time",
            "samples",
        )
        if column in selected.columns
    ]
    return selected[columns].copy()


def _paired_table(path: str | Path = "results/tables/paired_tests.csv") -> pd.DataFrame:
    paired = pd.read_csv(path)
    selected = paired[
        (paired["topology"] == "nsfnet")
        & (paired["target"] == "presto_dqn")
        & paired["baseline"].isin(["reactive_dqn", "predictive_dqn"])
        & paired["metric"].isin(["pdr", "recovery_time", "switches"])
    ]
    columns = [
        "scenario",
        "metric",
        "baseline",
        "n_episodes",
        "median_difference",
        "p_holm",
        "effect_size_r",
    ]
    selected = selected[columns].sort_values(["scenario", "metric", "baseline"]).copy()
    selected["median_difference"] = selected["median_difference"].map(lambda value: f"{value:.5g}")
    selected["p_holm"] = selected["p_holm"].map(lambda value: f"{value:.3g}")
    selected["effect_size_r"] = selected["effect_size_r"].map(lambda value: f"{value:.3f}")
    return selected


def _ablation_table(episodes: pd.DataFrame) -> pd.DataFrame:
    cells = [
        ("No prediction; no penalty", "reactive_dqn"),
        ("Prediction; no penalty", "presto_lam0"),
        ("Prediction; selected penalty", "presto_dqn"),
    ]
    mixed = episodes[(episodes["topology"] == "nsfnet") & (episodes["scenario"] == "mixed")]
    rows = []
    for label, method in cells:
        selected = mixed[mixed["method"] == method]
        seed_means = selected.groupby("training_seed")[["pdr", "switches"]].mean()
        rows.append(
            {
                "Ablation": label,
                "Method": method,
                "PDR (mean over seeds)": (
                    f"{seed_means['pdr'].mean():.4f}" if not seed_means.empty else "not available"
                ),
                "Switches / episode (mean over seeds)": (
                    f"{seed_means['switches'].mean():.3f}"
                    if not seed_means.empty
                    else "not available"
                ),
            }
        )
    rows.append(
        {
            "Ablation": "No prediction; selected penalty",
            "Method": "reactive_dqn_pen",
            "PDR (mean over seeds)": "not run in the completed P0 grid",
            "Switches / episode (mean over seeds)": "not run in the completed P0 grid",
        }
    )
    return pd.DataFrame(rows)


def _headline(episodes: pd.DataFrame) -> str:
    mixed = episodes[
        (episodes["topology"] == "nsfnet")
        & (episodes["scenario"] == "mixed")
        & episodes["method"].isin(["presto_dqn", "predictive_dqn"])
    ]
    seed_means = mixed.groupby(["method", "training_seed"])[["pdr", "switches"]].mean()
    methods = set(seed_means.index.get_level_values("method"))
    if not {"presto_dqn", "predictive_dqn"}.issubset(methods):
        return "The mixed-scenario headline comparison is unavailable in the supplied metrics."
    presto = seed_means.loc["presto_dqn"].mean()
    predictive = seed_means.loc["predictive_dqn"].mean()
    switch_change = (
        (predictive["switches"] - presto["switches"]) / predictive["switches"] * 100.0
        if predictive["switches"]
        else np.nan
    )
    pdr_difference = presto["pdr"] - predictive["pdr"]
    return (
        f"On NSFNET mixed test episodes, PRESTO-R averaged {presto['switches']:.2f} "
        f"switches per episode versus {predictive['switches']:.2f} for Predictive DQN "
        f"({switch_change:.1f}% fewer) and its PDR was {presto['pdr']:.4f} versus "
        f"{predictive['pdr']:.4f} (difference {pdr_difference:+.4f}). These values average "
        "episode metrics within each training seed before averaging across training seeds."
    )


def _robustness_table(path: str | Path = "results/tables/robustness.csv") -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[
        frame["metric"].isin(["pdr", "switches", "recovery_time", "proactive_rate"])
    ].copy()


def _success_table(path: str | Path = "results/success_criteria.md") -> str:
    content = Path(path).read_text(encoding="utf-8")
    lines = content.splitlines()
    try:
        start = lines.index("| Check | Result | Details |")
    except ValueError as error:
        raise ValueError(f"Success criteria table not found in {path}") from error
    return "\n".join(lines[start:])


def render_report(
    episodes_path: str | Path = "results/episode_metrics.parquet",
    output_path: str | Path = REPORT_PATH,
) -> Path:
    """Generate the report and all data tables from Phase 5 artifacts."""
    episodes = pd.read_parquet(episodes_path)
    required = {"method", "topology", "scenario", "training_seed", "pdr", "switches"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"Episode metrics are missing required fields: {sorted(missing)}")

    nsfnet = _summary_table(episodes, "nsfnet")
    rgg = _summary_table(episodes, "rgg")
    predictor = _predictor_table()
    paired = _paired_table()
    ablation = _ablation_table(episodes)
    robustness = _robustness_table()
    horizon_path = Path("results/tables/horizon_sweep.csv")
    horizon = pd.read_csv(horizon_path)
    lambda_sweep = pd.read_csv("results/tables/lambda_sweep_test.csv")
    success = _success_table()

    text = f"""# PRESTO-R: Predictive, Stability-Aware Reinforcement Learning Routing

## Abstract

This report evaluates risk-aware routing with a GRU link-degradation predictor and a switch-penalized DQN policy. Results are from the repository's simulated network environment; they are not evidence of deployment performance on production networks. All reported metrics and tables below are generated from saved test episodes and resolved configurations.

## 1. Introduction

Routing policies must balance delivery and recovery against the instability caused by frequent path changes. PRESTO-R augments per-path observations with predicted link risk and adds a penalty for switching. The project evaluates whether this combination improves routing behavior over a reactive DQN, a predictive policy without the selected switch penalty, and a shortest-path failover baseline.

## 2. Related work and scope

The experimental comparisons are internal baselines implemented in this repository. The project does not claim a reproduction of an external RQSMR implementation. Its predictor is trained on simulator-derived telemetry, and failures are synthetic. Claims in this report are limited to the tested simulator and seed ranges.

## 3. System design

![PRESTO-R architecture](figures/F1_architecture.png)

The simulator supplies noisy telemetry and route outcomes. A GRU estimates link degradation risk over horizon H; the DQN policy observes path features and risk features. Its reward includes the selected switch penalty λ. Heuristic and learned policies use a common deterministic rollout and metric pipeline.

## 4. Experimental setup

The main held-out grid uses five scenarios and two topologies. NSFNET learned agents use five training seeds; RGG agents use three. The common test-seed range and configuration-derived parameters are shown below. PDR is delivered traffic divided by offered demand; recovery time, proactive rate, and route switches are computed per episode. Across-seed standard deviations describe training-seed variation; paired tests use matched held-out scenario seeds with Holm correction.

### T1. Simulation and training parameters

{_markdown_table(_parameter_table())}

## 5. Results

### 5.1 Predictor quality

![Predictor precision-recall and calibration](figures/F2_predictor_quality.png)

The precision-recall panel is restricted to onset samples. Calibration uses the held-out predictor test split. The table distinguishes onset from all samples and includes predictor subgroup rows when available.

### T2. Held-out predictor metrics

{_markdown_table(predictor.round(4))}

![Example learned and oracle risk trace](figures/F3_risk_trace.png)

The risk trace is a deterministic illustrative mixed-scenario episode, not an aggregate score.

### 5.2 Main comparison

![NSFNET results](figures/F4_nsfnet_results.png)

{_headline(episodes)}

Bars show the mean across training-seed episode averages; error bars are standard deviations across training seeds. The deterministic heuristic has no between-training-seed variation.

#### T3. NSFNET test results (mean ± standard deviation over training seeds)

{_markdown_table(nsfnet)}

#### T4. Paired PRESTO-R comparisons on NSFNET

{_markdown_table(paired)}

### 5.3 Prediction and switch-penalty ablation

![Switch penalty trade-off](figures/F5_lambda_tradeoff.png)

The λ curve uses mixed-scenario test episodes for the validation-screened penalty variants. The full 2×2 factorial is not available because the no-prediction/penalty (`reactive_dqn_pen`) cell was not part of the completed P0 test grid; it is explicitly marked rather than inferred.

#### T5. Available mixed-scenario ablation cells

{_markdown_table(ablation)}

### 5.4 Behavior around failures

![Held-out mixed-scenario event trace](figures/F6_event_trace.png)

The event trace shows the first suitable held-out test episode found among the first 20 test seeds. It is illustrative and should be interpreted alongside the per-episode statistics, not as a representative estimate.

### 5.5 Sensitivity to horizon and switch penalty

![Prediction horizon sensitivity](figures/F7_horizon_sweep.png)

Horizon-specific agents are evaluated on mixed and node-failure test scenarios; predictor F1 is measured on the corresponding held-out predictor data. The λ sweep is shown in F5.

#### Horizon-sweep results

{_markdown_table(horizon.round(4))}

### 5.6 Second topology and robustness

![RGG results](figures/F9_rgg_results.png)

#### RGG test results

{_markdown_table(rgg)}

#### T6. Evaluation-only robustness checks

{_markdown_table(robustness.round(4))}

Sudden failures remove precursor warning; noise and background-load changes are applied only at evaluation. No robustness condition is used for retraining or test-based tuning.

### Mixed-scenario λ sensitivity data

{_markdown_table(lambda_sweep.round(4))}

## 6. Discussion and limitations

Results describe a fluid simulator rather than packet-level network behavior. The setup controls one managed flow and uses synthetic failures with designed precursor behavior. The predictor is trained on telemetry generated under the simulator's failover policy, which can limit transfer to other policies or networks. The sudden-failure check is especially important because such events cannot be anticipated from precursor signals. The no-prediction/penalty factorial ablation is missing, and the report does not treat the available partial ablation as a complete causal decomposition.

## 7. Conclusion and future work

PRESTO-R combines risk features with a switching cost; the outcomes and failed criteria are reported without suppressing negative comparisons. Further validation should add multiple managed flows, multipath splitting, packet-level or real-network traces, and online predictor updates. NS-3 validation remains optional Phase 7 work and is not included here.

## 8. Success criteria

### T7. Predeclared criteria

{success}

## Reproducibility

From the repository root, install the pinned dependencies, then run `make quick` for a smoke validation. With the completed raw Phase 5 results present, `make tables`, `make figures`, and `make report` regenerate the derived outputs. All full-grid runs can be resumed with the Phase 5 commands in the README. Test outputs must not be used to retune models or thresholds.
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/episode_metrics.parquet")
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()
    path = render_report(args.metrics, args.out)
    print(f"Saved report to {path}")


if __name__ == "__main__":
    main()
