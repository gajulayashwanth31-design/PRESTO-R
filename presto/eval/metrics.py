"""Episode-level routing metrics computed from rollout and ground-truth records."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _links(value) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(link) for link in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    return [int(link) for link in parsed]


def _truth_arrays(
    steps: pd.DataFrame,
    sim_up: np.ndarray | None,
    sim_bad: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_links = int(steps["n_links"].max()) if "n_links" in steps and not steps.empty else 0
    max_tick = int(steps["t"].max()) if not steps.empty else -1
    if sim_up is None:
        up = np.ones((max_tick + 1, n_links), dtype=bool)
        if "down_links" in steps:
            for row in steps.itertuples(index=False):
                for link in _links(row.down_links):
                    up[int(row.t), link] = False
    else:
        up = np.asarray(sim_up, dtype=bool)
        if up.ndim != 2:
            raise ValueError("sim_up must have shape (ticks, links)")
    if sim_bad is not None:
        bad = np.asarray(sim_bad, dtype=bool)
        if bad.shape != up.shape:
            raise ValueError("sim_bad must have the same shape as sim_up")
    elif "true_bad_links" in steps:
        bad = np.zeros_like(up)
        for row in steps.itertuples(index=False):
            for link in _links(row.true_bad_links):
                bad[int(row.t), link] = True
        bad |= ~up
    else:
        bad = ~up.copy()
    if bad.shape[1] != up.shape[1]:
        raise ValueError("sim_up and sim_bad must have the same link dimension")
    return up, bad


def episode_metrics(
    steps: pd.DataFrame,
    events: pd.DataFrame | None = None,
    sim_up: np.ndarray | None = None,
    H: int = 5,
    qos_ok_loss: float = 0.02,
    sim_bad: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute delivery, switching, recovery, proactive, energy and reward metrics.

    A switch is unnecessary if its old path is healthy now and throughout the
    next H ticks. Failure events are true up-to-down transitions whose link was
    on the selected path H ticks earlier. Proactive events have already left
    that link by failure time and count as zero-tick recoveries. Otherwise,
    recovery is the first valid selected path with loss <= qos_ok_loss, capped
    at the last recorded episode tick. If exact ``sim_bad`` truth is omitted,
    rollout ``true_bad_links`` is used; only schedule ``sim_up`` or recorded
    down links are used as a fallback.
    """
    required = {
        "t",
        "switched",
        "path_links",
        "path_loss",
        "path_delay",
        "delivered",
        "demand",
        "hops",
        "reward",
    }
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"steps is missing required columns: {', '.join(sorted(missing))}")
    if H < 1 or not 0.0 <= qos_ok_loss <= 1.0:
        raise ValueError("H must be positive and qos_ok_loss must be in [0, 1]")
    if steps.empty:
        raise ValueError("steps must contain at least one episode row")
    if steps["scenario_seed"].nunique() > 1:
        raise ValueError("episode_metrics accepts one episode at a time")

    steps = steps.sort_values("t").reset_index(drop=True)
    up, bad = _truth_arrays(steps, sim_up, sim_bad)
    tick_rows = {int(row.t): row for row in steps.itertuples(index=False)}
    switches = steps[steps["switched"].astype(bool)]
    unnecessary = 0
    for row in switches.itertuples(index=False):
        tick = int(row.t)
        if hasattr(row, "previous_path_links"):
            old_path = _links(row.previous_path_links)
        else:
            prior_rows = steps[steps["t"] < tick]
            old_path = _links(prior_rows.iloc[-1]["path_links"]) if not prior_rows.empty else []
        if not old_path:
            continue
        horizon_end = min(tick + H, bad.shape[0] - 1)
        old_path_bad = np.any(bad[tick : horizon_end + 1, old_path])
        if not old_path_bad:
            unnecessary += 1

    failure_events = 0
    proactive_events = 0
    recovery_times: list[int] = []
    for link in range(up.shape[1]):
        failure_ticks = np.flatnonzero((~up[1:, link]) & up[:-1, link]) + 1
        for failure_tick in failure_ticks:
            prior_row = tick_rows.get(int(failure_tick - H))
            failure_row = tick_rows.get(int(failure_tick))
            if prior_row is None or failure_row is None:
                continue
            prior_path = _links(
                prior_row.previous_path_links
                if hasattr(prior_row, "previous_path_links")
                else prior_row.path_links
            )
            if link not in prior_path:
                continue
            failure_events += 1
            active_path = _links(failure_row.path_links)
            if link not in active_path:
                proactive_events += 1
                recovery_times.append(0)
                continue
            recovered = None
            for tick in sorted(t for t in tick_rows if t >= failure_tick):
                row = tick_rows[tick]
                path = _links(row.path_links)
                valid = bool(getattr(row, "valid_chosen", np.all(up[tick, path])))
                if valid and float(row.path_loss) <= qos_ok_loss:
                    recovered = tick - int(failure_tick)
                    break
            recovery_times.append(
                int(recovered)
                if recovered is not None
                else max(0, int(steps["t"].max()) - int(failure_tick))
            )

    delivered = steps["delivered"].to_numpy(dtype=np.float64)
    demand = steps["demand"].to_numpy(dtype=np.float64)
    delay = steps["path_delay"].to_numpy(dtype=np.float64)
    total_demand = float(np.sum(demand))
    delivered_total = float(np.sum(delivered))
    return {
        "pdr": delivered_total / total_demand if total_demand > 0 else float("nan"),
        "delay_mean": (
            float(np.average(delay, weights=delivered)) if np.sum(delivered) > 0 else float("nan")
        ),
        "throughput": float(np.mean(delivered)),
        "switches": int(steps["switched"].astype(bool).sum()),
        "unnecessary_switch_rate": (unnecessary / len(switches) if len(switches) else float("nan")),
        "recovery_time": float(np.mean(recovery_times)) if recovery_times else float("nan"),
        "proactive_rate": (proactive_events / failure_events if failure_events else float("nan")),
        "energy": float(steps["hops"].sum()),
        "reward": float(steps["reward"].sum()),
        "failure_events": failure_events,
        "scheduled_events": 0 if events is None else int(len(events)),
    }


def rollout_metrics(
    steps: pd.DataFrame,
    events: pd.DataFrame | None = None,
    H: int = 5,
    qos_ok_loss: float = 0.02,
) -> pd.DataFrame:
    """Return one metrics row per scenario seed in a rollout result."""
    if "scenario_seed" not in steps:
        raise ValueError("steps must include scenario_seed")
    rows = []
    for seed, episode_steps in steps.groupby("scenario_seed", sort=True):
        episode_events = (
            events[events["scenario_seed"] == seed]
            if events is not None and "scenario_seed" in events
            else events
        )
        result = episode_metrics(
            episode_steps,
            episode_events,
            H=H,
            qos_ok_loss=qos_ok_loss,
        )
        result["scenario_seed"] = int(seed)
        result["scenario"] = str(episode_steps["scenario"].iloc[0])
        rows.append(result)
    return pd.DataFrame(rows)


def compute_all(
    raw_dir: str | Path = "results/raw",
    output: str | Path = "results/episode_metrics.parquet",
    H: int | None = None,
) -> pd.DataFrame:
    """Compute episode metrics from every raw step-log parquet in the evaluation grid."""
    if H is not None and H < 1:
        raise ValueError("H must be positive")
    root = Path(raw_dir)
    paths = sorted(root.glob("*/*/*/*.steps.parquet"))
    if not paths:
        raise FileNotFoundError(f"No step-log parquet files found under {root}")

    records = []
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) != 4:
            raise ValueError(f"Unexpected raw step-log path: {path}")
        method, topology, scenario, filename = relative.parts
        training_seed = filename.removesuffix(".steps.parquet").removeprefix("s")
        steps = pd.read_parquet(path)
        events_path = path.with_name(filename.removesuffix(".steps.parquet") + ".events.parquet")
        if not events_path.is_file():
            raise FileNotFoundError(f"Missing event log for raw step log: {events_path}")
        events = pd.read_parquet(events_path)
        config_path = path.with_name(filename.removesuffix(".steps.parquet") + ".config.yaml")
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing resolved config for raw log: {config_path}")
        cell_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        risk_config = cell_config.get("agent", {}).get("risk_model")
        risk_horizon = risk_config.get("horizon", 5) if isinstance(risk_config, dict) else 5
        horizon = int(cell_config.get("predictor", {}).get("horizon", risk_horizon))
        episodes = rollout_metrics(steps, events, H=horizon if H is None else H)
        for metric_row in episodes.to_dict(orient="records"):
            metric_row.update(
                {
                    "method": method,
                    "topology": topology,
                    "scenario": scenario,
                    "training_seed": training_seed,
                    "raw_steps": str(path),
                    "raw_events": str(events_path) if events_path.is_file() else "",
                    "raw_config": str(config_path),
                    "lambda": float(cell_config.get("reward", {}).get("lam", 0.0)),
                    "horizon": horizon,
                }
            )
            records.append(metric_row)

    result = pd.DataFrame(records)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="zstd")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute episode metrics from raw grid logs")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--out", default="results/episode_metrics.parquet")
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    result = compute_all(args.raw_dir, args.out, args.horizon)
    print(f"Saved {len(result)} episode metric rows to {args.out}")


if __name__ == "__main__":
    main()
