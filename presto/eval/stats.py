"""Seed-aware summaries and paired statistical comparisons for test results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t, wilcoxon

METRICS = ("pdr", "switches", "recovery_time")


def _seed_level(df: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "topology", "scenario", "training_seed", "scenario_seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Episode metrics are missing columns: {', '.join(sorted(missing))}")
    metric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column
        in {
            "pdr",
            "delay_mean",
            "throughput",
            "switches",
            "unnecessary_switch_rate",
            "recovery_time",
            "proactive_rate",
            "energy",
            "reward",
            "failure_events",
            "scheduled_events",
        }
    ]
    if not metric_columns:
        raise ValueError("Episode metrics contain no numeric metrics")
    return df.groupby(["method", "topology", "scenario", "training_seed"], as_index=False)[
        metric_columns
    ].mean()


def summary_table(
    df: pd.DataFrame,
    output_dir: str | Path = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Write one method summary per topology using training-seed uncertainty."""
    seed_means = _seed_level(df)
    outputs = {}
    for topology, topology_df in seed_means.groupby("topology", sort=True):
        metrics = [
            column
            for column in topology_df.select_dtypes(include="number").columns
            if column
            in {
                "pdr",
                "delay_mean",
                "throughput",
                "switches",
                "unnecessary_switch_rate",
                "recovery_time",
                "proactive_rate",
                "energy",
                "reward",
                "failure_events",
                "scheduled_events",
            }
        ]
        long = topology_df.melt(
            id_vars=["method", "scenario", "training_seed"],
            value_vars=metrics,
            var_name="metric",
            value_name="seed_mean",
        )
        stats = long.groupby(["method", "scenario", "metric"])["seed_mean"].agg(
            mean="mean", std="std", count="count"
        )
        standard_error = stats["std"] / np.sqrt(stats["count"])
        critical = stats["count"].map(
            lambda count: float(t.ppf(0.975, count - 1)) if count > 1 else np.nan
        )
        stats["ci95_lower"] = stats["mean"] - critical * standard_error
        stats["ci95_upper"] = stats["mean"] + critical * standard_error

        def format_cell(row: pd.Series) -> str:
            if not np.isfinite(row["mean"]):
                return "nan"
            std = row["std"]
            return (
                f"{row['mean']:.6g} ± {std:.6g}" if np.isfinite(std) else f"{row['mean']:.6g} ± NA"
            )

        stats["summary"] = stats.apply(format_cell, axis=1)
        table = stats[["summary", "ci95_lower", "ci95_upper"]].unstack(["metric", "scenario"])
        table.columns = [
            f"{column}_{metric}_{scenario}" for column, metric, scenario in table.columns
        ]
        outputs[str(topology)] = table.reset_index()
        output_path = Path(output_dir) / f"summary_{topology}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs[str(topology)].to_csv(output_path, index=False)
    return outputs


def paired_tests(
    df: pd.DataFrame,
    target: str = "presto_dqn",
    metrics: tuple[str, ...] = METRICS,
    output: str | Path | None = "results/tables/paired_tests.csv",
) -> pd.DataFrame:
    """Run paired episode-level Wilcoxon tests and apply Holm correction globally."""
    required = {"method", "topology", "scenario", "training_seed", "scenario_seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Episode metrics are missing columns: {', '.join(sorted(missing))}")
    missing_metrics = set(metrics) - set(df.columns)
    if missing_metrics:
        raise ValueError(f"Requested metrics are missing: {', '.join(sorted(missing_metrics))}")

    rows = []
    for (topology, scenario), cell in df.groupby(["topology", "scenario"], sort=True):
        if target not in set(cell["method"]):
            continue
        target_values = (
            cell[cell["method"] == target].groupby("scenario_seed")[list(metrics)].mean()
        )
        for baseline in sorted(set(cell["method"]) - {target}):
            baseline_values = (
                cell[cell["method"] == baseline].groupby("scenario_seed")[list(metrics)].mean()
            )
            common_seeds = target_values.index.intersection(baseline_values.index)
            for metric in metrics:
                paired = pd.concat(
                    [
                        target_values.loc[common_seeds, metric].rename("target"),
                        baseline_values.loc[common_seeds, metric].rename("baseline"),
                    ],
                    axis=1,
                ).dropna()
                if paired.empty:
                    continue
                difference = paired["target"].to_numpy() - paired["baseline"].to_numpy()
                if np.allclose(difference, 0):
                    statistic, p_value, z_score = 0.0, 1.0, 0.0
                else:
                    result = wilcoxon(
                        paired["target"],
                        paired["baseline"],
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                    z_score = float(norm.isf(p_value / 2.0)) * float(np.sign(np.median(difference)))
                rows.append(
                    {
                        "topology": topology,
                        "scenario": scenario,
                        "metric": metric,
                        "target": target,
                        "baseline": baseline,
                        "n_episodes": int(len(paired)),
                        "median_difference": float(np.median(difference)),
                        "wilcoxon_statistic": statistic,
                        "p_raw": p_value,
                        "effect_size_r": z_score / np.sqrt(len(paired)),
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        order = result["p_raw"].sort_values(kind="stable").index
        m = len(order)
        adjusted = pd.Series(index=result.index, dtype=float)
        running_max = 0.0
        for rank, index in enumerate(order):
            corrected = min(1.0, float(result.loc[index, "p_raw"]) * (m - rank))
            running_max = max(running_max, corrected)
            adjusted.loc[index] = running_max
        result["p_holm"] = adjusted
    else:
        result["p_holm"] = pd.Series(dtype=float)
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def lambda_sweep_summary(
    df: pd.DataFrame,
    output: str | Path = "results/tables/lambda_sweep_test.csv",
) -> pd.DataFrame:
    """Summarize mixed-scenario test outcomes for validation-screened lambda agents."""
    required = {"method", "scenario", "lambda"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Episode metrics are missing columns: {', '.join(sorted(missing))}")
    selected = df[df["method"].str.startswith("presto_lam") & (df["scenario"] == "mixed")]
    if selected.empty:
        raise ValueError("No lambda-screen test episodes are present")
    metrics = [
        metric
        for metric in (
            "pdr",
            "switches",
            "unnecessary_switch_rate",
            "recovery_time",
            "proactive_rate",
        )
        if metric in selected.columns
    ]
    result = selected.groupby(["lambda", "method"], as_index=False)[metrics].agg(["mean", "std"])
    result.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
        for column in result.columns
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def horizon_sweep_summary(
    df: pd.DataFrame,
    predictor_metrics: dict[int, str | Path],
    output: str | Path = "results/tables/horizon_sweep.csv",
) -> pd.DataFrame:
    """Summarize H-specific test agents with predictor onset F1 by horizon."""
    required = {"method", "scenario", "training_seed", "horizon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Episode metrics are missing columns: {', '.join(sorted(missing))}")
    selected = df[df["method"].isin({"presto_h3", "presto_h10"})]
    if selected.empty:
        raise ValueError("No horizon-sweep agent episodes are present")
    metrics = [
        metric
        for metric in ("pdr", "switches", "proactive_rate", "recovery_time")
        if metric in selected.columns
    ]
    seed_means = selected.groupby(
        ["horizon", "scenario", "method", "training_seed"], as_index=False
    )[metrics].mean()
    summary = seed_means.groupby(["horizon", "scenario"], as_index=False)[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    onset_scores = []
    for horizon, path in sorted(predictor_metrics.items()):
        predictor_df = pd.read_csv(path)
        rows = predictor_df[
            (predictor_df["model"] == "gru")
            & (predictor_df["subset"] == "onset")
            & (predictor_df["group"] == "all")
        ]
        if rows.empty:
            raise ValueError(f"Missing GRU onset metrics for horizon {horizon}: {path}")
        onset_scores.append({"horizon": horizon, "predictor_onset_f1": float(rows.iloc[-1]["f1"])})
    summary = summary.merge(
        pd.DataFrame(onset_scores), on="horizon", how="left", validate="many_to_one"
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def robustness_summary(
    df: pd.DataFrame,
    output: str | Path = "results/tables/robustness.csv",
) -> pd.DataFrame:
    """Report P0 policy metrics under evaluation-only robustness perturbations."""
    cases = {
        "sudden": ("reactive_sudden", "presto_sudden"),
        "noise2x": ("reactive_noise2x", "presto_noise2x"),
        "load150": ("reactive_load150", "presto_load150"),
    }
    selected = df[df["method"].isin({name for pair in cases.values() for name in pair})]
    if selected.empty:
        raise ValueError("No robustness test episodes are present")
    metrics = [
        metric
        for metric in (
            "pdr",
            "delay_mean",
            "switches",
            "unnecessary_switch_rate",
            "recovery_time",
            "proactive_rate",
        )
        if metric in selected.columns
    ]
    seed_means = selected.groupby(["method", "training_seed"], as_index=False)[metrics].mean()
    records = []
    for case, (reactive_method, presto_method) in cases.items():
        for method in (reactive_method, presto_method):
            rows = seed_means[seed_means["method"] == method]
            if rows.empty:
                raise ValueError(f"Missing robustness results for {method}")
            for metric in metrics:
                records.append(
                    {
                        "case": case,
                        "method": method,
                        "metric": metric,
                        "mean_over_training_seeds": float(rows[metric].mean()),
                        "std_over_training_seeds": float(rows[metric].std()),
                        "training_seeds": int(rows[metric].count()),
                    }
                )
    result = pd.DataFrame(records)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/episode_metrics.parquet")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument("--horizon-predictor", action="append", default=[], metavar="H=CSV")
    args = parser.parse_args()
    episodes = pd.read_parquet(args.metrics)
    summary_table(episodes, args.out_dir)
    tests = paired_tests(episodes, output=Path(args.out_dir) / "paired_tests.csv")
    lambda_sweep_summary(episodes, Path(args.out_dir) / "lambda_sweep_test.csv")
    if episodes["method"].astype(str).str.contains("_sudden|_noise2x|_load150", regex=True).any():
        robustness_summary(episodes, Path(args.out_dir) / "robustness.csv")
    if args.horizon_predictor:
        predictors = {}
        for item in args.horizon_predictor:
            if "=" not in item:
                parser.error(f"Horizon predictor metrics must use H=CSV syntax: {item}")
            horizon, path = item.split("=", 1)
            if not horizon or not path:
                parser.error(f"Horizon predictor metrics must use H=CSV syntax: {item}")
            predictors[int(horizon)] = path
        horizon_sweep_summary(episodes, predictors, Path(args.out_dir) / "horizon_sweep.csv")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
