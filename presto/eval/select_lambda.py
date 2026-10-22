"""Select the largest validation lambda within one PDR point of the best."""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from presto.agents.train import build_risk
from presto.config import load_config
from presto.eval.metrics import rollout_metrics
from presto.eval.rollout import rollout
from presto.policies.sb3_policy import SB3Policy


def _choose_lambda(overall: pd.DataFrame) -> float:
    """Choose largest lambda whose mean validation PDR is within one point of best."""
    required = {"lambda", "pdr_mean"}
    if required - set(overall.columns) or overall.empty:
        raise ValueError("Lambda summary must include lambda and pdr_mean")
    finite = overall[np.isfinite(overall["pdr_mean"].to_numpy(dtype=float))]
    if finite.empty:
        raise ValueError("No finite validation PDR scores are available for lambda selection")
    best_pdr = float(finite["pdr_mean"].max())
    candidates = finite[finite["pdr_mean"] >= best_pdr - 0.01]
    return float(candidates["lambda"].max())


def select_lambda(run_patterns: list[str], output: str | Path) -> tuple[pd.DataFrame, float]:
    """Evaluate trained lambda-screening runs on configured validation seeds only."""
    run_dirs = sorted(
        {
            Path(match)
            for pattern in run_patterns
            for match in glob.glob(pattern)
            if Path(match).is_dir()
        }
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched: {run_patterns}")

    run_rows = []
    low_high = None
    for run_dir in run_dirs:
        config_path = run_dir / "config.yaml"
        model_path = run_dir / "best_model.zip"
        if not config_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"Incomplete lambda run {run_dir}; expected config.yaml and best_model.zip"
            )
        cfg = load_config(config_path)
        low, high = (int(value) for value in cfg["seeds"]["val"])
        if low_high is not None and low_high != (low, high):
            raise ValueError("Lambda runs must use the same validation seed range")
        low_high = low, high
        seeds = list(range(low, high))
        lambda_value = float(cfg["reward"]["lam"])
        method = str(cfg["agent"]["name"])
        policy = SB3Policy(str(model_path), algo=str(cfg["agent"]["algo"]), name=method)
        risk = build_risk(cfg)
        scenarios = list(cfg["agent"].get("eval_scenarios", cfg["agent"]["train_scenarios"]))
        for scenario in scenarios:
            steps, events = rollout(
                policy,
                cfg,
                seeds,
                risk,
                scenario=scenario,
            )
            metrics = rollout_metrics(
                steps,
                events,
                H=int(cfg.get("predictor", {}).get("horizon", 5)),
            )
            metrics["lambda"] = lambda_value
            metrics["method"] = method
            metrics["training_seed"] = f"{run_dir.parent.name}/{run_dir.name}"
            run_rows.append(metrics)

    if not run_rows:
        raise FileNotFoundError("No complete runs with config.yaml and best_model.zip were found")
    per_episode = pd.concat(run_rows, ignore_index=True)
    summary = per_episode.groupby(["lambda", "scenario"], as_index=False).agg(
        pdr_mean=("pdr", "mean"),
        pdr_std=("pdr", "std"),
        switches_mean=("switches", "mean"),
        switches_std=("switches", "std"),
        unnecessary_switch_rate_mean=("unnecessary_switch_rate", "mean"),
        reward_mean=("reward", "mean"),
        recovery_time_mean=("recovery_time", "mean"),
        proactive_rate_mean=("proactive_rate", "mean"),
        training_seeds=("training_seed", "nunique"),
        episodes=("scenario_seed", "count"),
    )
    overall = per_episode.groupby("lambda", as_index=False).agg(
        pdr_mean=("pdr", "mean"),
        pdr_std=("pdr", "std"),
        switches_mean=("switches", "mean"),
        switches_std=("switches", "std"),
        unnecessary_switch_rate_mean=("unnecessary_switch_rate", "mean"),
        reward_mean=("reward", "mean"),
        recovery_time_mean=("recovery_time", "mean"),
        proactive_rate_mean=("proactive_rate", "mean"),
        training_seeds=("training_seed", "nunique"),
        episodes=("scenario_seed", "count"),
    )
    selected = _choose_lambda(overall)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    overall.to_csv(output_path.with_name(f"{output_path.stem}_overall.csv"), index=False)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter(overall["switches_mean"], overall["pdr_mean"], c=overall["lambda"])
    for _, row in overall.iterrows():
        axis.annotate(
            f"λ={row['lambda']:g}",
            (row["switches_mean"], row["pdr_mean"]),
        )
    axis.set(xlabel="Mean switches per episode", ylabel="Mean validation PDR")
    figure.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(figure)
    output_path.with_suffix(".selected.txt").write_text(f"{selected:g}\n", encoding="utf-8")
    return summary, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="results/tuning/lambda.csv")
    args = parser.parse_args()
    summary, selected = select_lambda(args.runs, args.out)
    print(summary.to_string(index=False))
    print(f"Selected lambda={selected:g} using validation PDR criterion")


if __name__ == "__main__":
    main()
