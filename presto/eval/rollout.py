"""Deterministic evaluation rollouts for routing policies."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from presto.env import RoutingEnv
from presto.predictor.features import bad_link_mask
from presto.risk import RiskProvider
from presto.policies.base import Policy


def _validate_seeds(seeds: Sequence[int], cfg: dict, allow_training_seeds: bool) -> list[int]:
    result = [int(seed) for seed in seeds]
    if not result:
        raise ValueError("At least one rollout seed is required")
    if len(result) != len(set(result)):
        raise ValueError("Rollout seeds must be unique")
    if allow_training_seeds:
        return result
    ranges = (cfg.get("seeds", {}).get("val"), cfg.get("seeds", {}).get("test"))
    if not all(isinstance(bounds, (list, tuple)) and len(bounds) == 2 for bounds in ranges):
        raise ValueError("Config must define validation and test seed ranges")
    if any(not any(int(low) <= seed < int(high) for low, high in ranges) for seed in result):
        raise ValueError("Evaluation rollouts must use validation or test seeds, not training seeds")
    return result


def rollout(
    policy: Policy,
    cfg: dict,
    seeds: Sequence[int],
    risk: RiskProvider | None = None,
    out_path: str | Path | None = None,
    *,
    scenario: str | None = None,
    allow_training_seeds: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one episode per fixed scenario seed and return step and event records.

    Ground-truth ``true_bad_links`` and ``down_links`` are retained for offline
    metrics only; they are never included in policy observations or info.
    If ``out_path`` is supplied, writes sibling ``.steps.parquet`` and
    ``.events.parquet`` files.
    """
    episode_seeds = _validate_seeds(seeds, cfg, allow_training_seeds)
    scenario_name = scenario or str(cfg["scenario"]["name"])
    env = RoutingEnv(cfg, risk=risk)
    step_rows: list[dict] = []
    event_rows: list[dict] = []
    try:
        for episode_seed in episode_seeds:
            obs, reset_info = env.reset(
                seed=episode_seed,
                options={"scenario_seed": episode_seed, "scenario": scenario_name},
            )
            policy.reset(reset_info)
            current_info = reset_info
            for event in reset_info["events"]:
                event_rows.append(
                    {
                        "scenario_seed": episode_seed,
                        "scenario": scenario_name,
                        "kind": event.kind,
                        "links": json.dumps(sorted(set(event.links))),
                        "start": int(event.start),
                        "precursor": int(event.precursor),
                        "duration": int(event.duration),
                        "params": json.dumps(event.params, sort_keys=True),
                    }
                )
            for _ in range(int(env.cfg["sim"]["episode_len"])):
                before = current_info
                previous_path = list(before["paths"][before["current"]])
                action = int(policy.act(obs, before))
                if not env.action_space.contains(action):
                    raise ValueError(f"Policy {policy.name!r} selected invalid action {action}")
                valid_chosen = bool(before["valid"][action])
                risk_before = before["risk"]
                obs, reward, terminated, truncated, info = env.step(action)
                current_info = info
                true_stats = env._last_true_stats
                if true_stats is None or env.sim is None:
                    raise RuntimeError("Environment did not retain true simulator metrics")
                bad = bad_link_mask(
                    true_stats.up[None, :],
                    true_stats.loss[None, :],
                    true_stats.delay[None, :],
                    env.sim.base_delay,
                    float(env.cfg.get("predictor", {}).get("bad_loss", 0.05)),
                    float(env.cfg.get("predictor", {}).get("bad_delay_factor", 3.0)),
                )[0]
                links = list(info["path_links"])
                health = env.sim.schedules["health"][int(info["t"]), links]
                qos = info["qos"]
                row = {
                    "scenario_seed": episode_seed,
                    "scenario": scenario_name,
                    "t": int(info["t"]),
                    "action": int(action),
                    "switched": bool(info["switched"]),
                    "valid_chosen": valid_chosen,
                    "path_delay": float(qos["delay"]),
                    "path_loss": float(qos["loss"]),
                    "delivered": float(qos["delivered"]),
                    "demand": float(env.sim.managed_rate),
                    "hops": int(qos["hops"]),
                    "reward": float(reward),
                    "path_links": json.dumps(links),
                    "previous_path_links": json.dumps(previous_path),
                    "max_true_health": float(np.max(health)),
                    "max_risk": (
                        float(np.max(risk_before[links])) if len(links) else 0.0
                    ),
                    "true_bad_links": json.dumps(np.flatnonzero(bad).tolist()),
                    "down_links": json.dumps(np.flatnonzero(~true_stats.up).tolist()),
                    "n_links": int(env.sim.n_links),
                }
                row.update({f"reward_{key}": float(value) for key, value in info["reward_terms"].items()})
                step_rows.append(row)
                if terminated or truncated:
                    break
    finally:
        env.close()

    steps = pd.DataFrame(step_rows)
    events = pd.DataFrame(
        event_rows,
        columns=("scenario_seed", "scenario", "kind", "links", "start", "precursor", "duration", "params"),
    )
    if out_path is not None:
        base = Path(out_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        steps.to_parquet(base.with_suffix(".steps.parquet"), index=False)
        events.to_parquet(base.with_suffix(".events.parquet"), index=False)
    return steps, events
