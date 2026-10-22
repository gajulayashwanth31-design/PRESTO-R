"""Choose a risk-threshold heuristic on validation seeds only."""

import argparse
import json
from pathlib import Path

import pandas as pd

from presto.config import load_config
from presto.eval.metrics import rollout_metrics
from presto.eval.rollout import rollout
from presto.policies.heuristics import RiskThreshold
from presto.risk import LearnedRisk, LogRegRisk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/predictor/gru.yaml")
    parser.add_argument("--model")
    parser.add_argument("--risk-type", choices=("gru", "logreg"), default="gru")
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--out", default="results/predictor/risk_threshold_tuning")
    args = parser.parse_args()
    cfg = load_config(args.config)
    extension = "joblib" if args.risk_type == "logreg" else "pt"
    model = args.model or str(
        Path(cfg["paths"]["models"])
        / "predictor"
        / (
            f"{args.risk_type}_W{cfg['predictor']['window']}_"
            f"H{cfg['predictor']['horizon']}.{extension}"
        )
    )
    risk = LogRegRisk(model) if args.risk_type == "logreg" else LearnedRisk(model)
    low, high = cfg["seeds"]["val"]
    seeds = list(range(int(low), int(high)))
    rows = []
    for threshold in args.thresholds:
        steps, events = rollout(
            RiskThreshold(threshold),
            cfg,
            seeds,
            risk,
            scenario=args.scenario,
        )
        metrics = rollout_metrics(steps, events, H=int(cfg["predictor"]["horizon"]))
        row = metrics.mean(numeric_only=True).to_dict()
        row["tau"] = threshold
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("reward", ascending=False)
    best = float(result.iloc[0]["tau"])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "selected_tau": best,
                "selection_metric": "mean validation reward",
                "scenario": args.scenario,
                "validation_seeds": [int(low), int(high)],
                "risk_model": str(model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result[["tau", "reward", "pdr", "switches"]].to_string(index=False))
    print(f"Selected tau={best:.2f}; saved validation tuning results to {output}")


if __name__ == "__main__":
    main()
