"""Evaluate the predeclared PRD success criteria against held-out test results."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from presto.eval.run_grid import _run_cell, build_tasks


def _result(condition: bool, detail: str) -> dict[str, str]:
    return {"status": "PASS" if condition else "FAIL", "detail": detail}


def _reproducibility_result(
    grid_path: str | Path,
    raw_dir: str | Path,
    method: str,
    topology: str,
    scenario: str,
    training_seed: str,
) -> dict[str, str]:
    grid = (
        json.loads(Path(grid_path).read_text(encoding="utf-8"))
        if str(grid_path).endswith(".json")
        else None
    )
    if grid is None:
        import yaml

        grid = yaml.safe_load(Path(grid_path).read_text(encoding="utf-8"))
    candidates = build_tasks(grid)
    chosen = None
    for task in candidates:
        if (
            task["method"] == method
            and task["topology"] == topology
            and task["scenario"] == scenario
            and str(task.get("run_directory", "")).endswith(f"seed_{training_seed}")
        ):
            chosen = task
            break
        if (
            task["method"] == method
            and task["topology"] == topology
            and task["scenario"] == scenario
            and training_seed == "deterministic"
            and "run_directory" not in task
        ):
            chosen = task
            break
    if chosen is None:
        raise ValueError("Requested reproducibility cell does not exist in the grid")

    original_stem = Path(raw_dir) / method / topology / scenario / f"s{training_seed}"
    original_steps = original_stem.with_suffix(".steps.parquet")
    original_events = original_stem.with_suffix(".events.parquet")
    if not original_steps.is_file() or not original_events.is_file():
        raise FileNotFoundError(f"Missing raw logs for reproducibility cell {original_stem}")
    with tempfile.TemporaryDirectory(prefix="presto-repro-") as temporary:
        chosen["out_dir"] = temporary
        _run_cell(chosen)
        reproduced_stem = Path(temporary) / method / topology / scenario / f"s{training_seed}"
        checks = (
            (original_steps, reproduced_stem.with_suffix(".steps.parquet")),
            (original_events, reproduced_stem.with_suffix(".events.parquet")),
        )
        matches = all(
            hashlib.sha256(original.read_bytes()).digest()
            == hashlib.sha256(reproduced.read_bytes()).digest()
            for original, reproduced in checks
        )
    return _result(matches, "Step and event parquet hashes match on a fresh rerun")


def check_success(
    metrics_path: str | Path = "results/episode_metrics.parquet",
    paired_path: str | Path = "results/tables/paired_tests.csv",
    predictor_path: str | Path = "results/predictor/metrics.csv",
) -> pd.DataFrame:
    """Return PASS/FAIL records for scientific success checks 1–4."""
    episodes = pd.read_parquet(metrics_path)
    paired = pd.read_csv(paired_path)
    predictor = pd.read_csv(predictor_path)
    nsfnet = episodes[episodes["topology"] == "nsfnet"]
    summaries = nsfnet.groupby(["method", "scenario"]).mean(numeric_only=True)
    results = []
    checks = []
    for scenario in ("node_failure", "fluctuation"):
        presto = summaries.loc[("presto_dqn", scenario)]
        reactive = summaries.loc[("reactive_dqn", scenario)]
        significant = paired[
            (paired["topology"] == "nsfnet")
            & (paired["scenario"] == scenario)
            & (paired["target"] == "presto_dqn")
            & (paired["baseline"] == "reactive_dqn")
            & (paired["metric"].isin(["pdr", "recovery_time"]))
        ]
        p_values = dict(zip(significant["metric"], significant["p_holm"], strict=True))
        passed = (
            presto["pdr"] > reactive["pdr"]
            and presto["recovery_time"] < reactive["recovery_time"]
            and all(float(p_values.get(metric, 1.0)) < 0.05 for metric in ("pdr", "recovery_time"))
        )
        checks.append(
            (
                f"PRESTO-R vs reactive in {scenario}",
                _result(
                    passed,
                    f"PDR {presto['pdr']:.4f} vs {reactive['pdr']:.4f}; "
                    f"recovery {presto['recovery_time']:.3f} vs "
                    f"{reactive['recovery_time']:.3f}; Holm p "
                    f"PDR={p_values.get('pdr', float('nan')):.4g}, "
                    f"recovery={p_values.get('recovery_time', float('nan')):.4g}",
                ),
            )
        )
    for scenario in ("mixed", "fluctuation"):
        presto = summaries.loc[("presto_dqn", scenario)]
        predictive = summaries.loc[("predictive_dqn", scenario)]
        passed = (
            presto["switches"] < predictive["switches"]
            and presto["pdr"] >= predictive["pdr"] - 0.02
        )
        checks.append(
            (
                f"PRESTO-R vs predictive in {scenario}",
                _result(
                    passed,
                    f"switches {presto['switches']:.3f} vs {predictive['switches']:.3f}; "
                    f"PDR {presto['pdr']:.4f} vs {predictive['pdr']:.4f}",
                ),
            )
        )
    presto_steady = summaries.loc[("presto_dqn", "steady")]
    reactive_steady = summaries.loc[("reactive_dqn", "steady")]
    max_delay = 1.10 * reactive_steady["delay_mean"]
    checks.append(
        (
            "Steady-state delay",
            _result(
                presto_steady["delay_mean"] <= max_delay,
                f"PRESTO-R delay {presto_steady['delay_mean']:.4f}; "
                f"limit {max_delay:.4f} (reactive × 1.10)",
            ),
        )
    )
    onset_rows = predictor[
        (predictor["model"] == "gru")
        & (predictor["subset"] == "onset")
        & (predictor["group"] == "all")
    ]
    if onset_rows.empty:
        raise ValueError("Predictor metrics have no GRU onset/all row")
    onset_f1 = float(onset_rows.iloc[-1]["f1"])
    checks.append(
        (
            "Predictor onset F1",
            _result(onset_f1 >= 0.7, f"GRU validation onset F1={onset_f1:.4f}; threshold=0.7000"),
        )
    )
    results.extend({"check": name, **value} for name, value in checks)
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/episode_metrics.parquet")
    parser.add_argument("--paired", default="results/tables/paired_tests.csv")
    parser.add_argument("--predictor", default="results/predictor/metrics.csv")
    parser.add_argument("--out", default="results/success_criteria.md")
    parser.add_argument("--grid", default="configs/grid/main.yaml")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--repro-method", default="presto_dqn")
    parser.add_argument("--repro-topology", default="nsfnet")
    parser.add_argument("--repro-scenario", default="mixed")
    parser.add_argument("--repro-training-seed", default="0")
    args = parser.parse_args()
    results = check_success(args.metrics, args.paired, args.predictor)
    reproducibility = _reproducibility_result(
        args.grid,
        args.raw_dir,
        args.repro_method,
        args.repro_topology,
        args.repro_scenario,
        args.repro_training_seed,
    )
    results.loc[len(results)] = {
        "check": "Reproducibility",
        **reproducibility,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 5 Success Criteria", "", "| Check | Result | Details |", "|---|---|---|"]
    lines.extend(
        f"| {row.check} | {row.status} | {row.detail} |" for row in results.itertuples(index=False)
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
