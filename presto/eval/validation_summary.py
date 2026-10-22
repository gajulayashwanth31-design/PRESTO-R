"""Aggregate validation metrics for trained routing-method run directories."""

import argparse
import json
from pathlib import Path

import pandas as pd

from presto.agents.train import build_risk
from presto.config import load_config
from presto.eval.metrics import rollout_metrics
from presto.eval.rollout import rollout
from presto.policies.sb3_policy import SB3Policy


def validation_summary(
    run_directories: list[str | Path],
    output: str | Path,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate every best model on validation seeds and aggregate training seeds."""
    records = []
    for run_directory in map(Path, run_directories):
        model_path = run_directory / "best_model.zip"
        config_path = run_directory / "config.yaml"
        if not model_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Incomplete agent run directory: {run_directory}")
        cfg = load_config(config_path)
        low, high = map(int, cfg["seeds"]["val"])
        validation_seeds = list(range(low, high))
        selected_scenarios = scenarios or list(
            cfg["agent"].get(
                "eval_scenarios",
                ["congestion", "node_failure", "fluctuation", "mixed"],
            )
        )
        policy = SB3Policy(
            str(model_path),
            algo=str(cfg["agent"].get("algo", "dqn")),
            name=str(cfg["agent"]["name"]),
        )
        risk = build_risk(cfg)
        for scenario in selected_scenarios:
            steps, events = rollout(
                policy,
                cfg,
                validation_seeds,
                risk,
                scenario=scenario,
            )
            metrics = rollout_metrics(
                steps,
                events,
                H=int(cfg.get("predictor", {}).get("horizon", 5)),
            )
            for metric in (
                "pdr",
                "delay_mean",
                "throughput",
                "switches",
                "unnecessary_switch_rate",
                "recovery_time",
                "proactive_rate",
                "energy",
                "reward",
            ):
                records.append(
                    {
                        "method": str(cfg["agent"]["name"]),
                        "training_seed": int(run_directory.name.removeprefix("seed_")),
                        "scenario": scenario,
                        "metric": metric,
                        "mean": float(metrics[metric].mean()),
                        "std": float(metrics[metric].std()),
                        "episodes": int(metrics[metric].count()),
                    }
                )
    result = pd.DataFrame(records)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "validation_seeds_only": True,
                "runs": [str(Path(path)) for path in run_directories],
                "scenarios": scenarios,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["congestion", "node_failure", "fluctuation", "mixed"],
    )
    parser.add_argument("--out", default="results/tuning/validation_summary.csv")
    args = parser.parse_args()
    result = validation_summary(args.runs, args.out, args.scenarios)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
