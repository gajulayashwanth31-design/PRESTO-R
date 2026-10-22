"""Validation-only diagnostics for learned routing policies."""

import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from presto.agents.train import build_risk
from presto.eval.metrics import rollout_metrics
from presto.eval.rollout import rollout
from presto.policies.base import Policy
from presto.policies.sb3_policy import SB3Policy


class _RecordObservations(Policy):
    def __init__(self, model: SB3Policy):
        self.model = model
        self.name = model.name
        self.observations: dict[int, list[np.ndarray]] = {}

    def reset(self, info: dict) -> None:
        self.seed = int(info["scenario_seed"])
        self.observations[self.seed] = []

    def act(self, obs: np.ndarray, info: dict) -> int:
        self.observations[self.seed].append(obs.copy())
        return self.model.act(obs, info)


class _PermuteRiskFeatures(Policy):
    def __init__(
        self, model: SB3Policy, observations: dict[int, list[np.ndarray]], k: int, seed: int
    ):
        self.model = model
        self.name = f"{model.name}_risk_permuted"
        self.observations = observations
        self.k = k
        self.rng = np.random.default_rng(seed)

    def reset(self, info: dict) -> None:
        self.seed = int(info["scenario_seed"])
        self.tick = 0

    def act(self, obs: np.ndarray, info: dict) -> int:
        result = obs.copy()
        history = self.observations[self.seed]
        if len(history) > 1:
            candidate = int(self.rng.integers(len(history) - 1))
            if candidate >= self.tick:
                candidate += 1
            donor = history[candidate]
            for path_index in range(self.k):
                start = path_index * 8
                result[start + 5 : start + 7] = donor[start + 5 : start + 7]
        self.tick += 1
        return self.model.act(result, info)


def risk_permutation_test(
    cfg: dict,
    model_path: str | Path,
    seeds: list[int],
    scenario: str = "mixed",
    output: str | Path = "results/diagnostics/risk_permutation.csv",
) -> pd.DataFrame:
    """Compare normal policy to episode-local temporal permutation of risk features."""
    k = int(cfg["paths"]["k"])
    if not bool(cfg["agent"].get("use_risk", False)):
        raise ValueError("Risk permutation requires an agent configured with use_risk=true")
    model = SB3Policy(str(model_path), algo=str(cfg["agent"].get("algo", "dqn")))
    recorder = _RecordObservations(model)
    baseline_steps, baseline_events = rollout(
        recorder,
        cfg,
        seeds,
        build_risk(cfg),
        scenario=scenario,
    )
    permuted = _PermuteRiskFeatures(model, recorder.observations, k, seed=0)
    permuted_steps, permuted_events = rollout(
        permuted,
        cfg,
        seeds,
        build_risk(cfg),
        scenario=scenario,
    )
    horizon = int(cfg.get("predictor", {}).get("horizon", 5))
    records = []
    for name, steps, events in (
        ("normal", baseline_steps, baseline_events),
        ("risk_permuted", permuted_steps, permuted_events),
    ):
        metrics = rollout_metrics(steps, events, H=horizon)
        means = metrics.select_dtypes(include="number").mean()
        records.append(
            {
                "condition": name,
                "pdr_mean": float(means["pdr"]),
                "switches_mean": float(means["switches"]),
                "proactive_rate_mean": float(means["proactive_rate"]),
                "recovery_time_mean": float(means["recovery_time"]),
                "episodes": int(len(metrics)),
            }
        )
    result = pd.DataFrame(records)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def event_trace(
    cfg: dict,
    models: dict[str, str | Path],
    seeds: list[int],
    scenario: str = "mixed",
    output: str | Path = "results/diagnostics/event_trace.png",
) -> Path:
    """Plot risk, action index, and loss around an active-path precursor failure."""
    if not models:
        raise ValueError("At least one trained model is required")
    steps_by_method: dict[str, pd.DataFrame] = {}
    event_records: dict[str, pd.DataFrame] = {}
    for method, path in models.items():
        policy = SB3Policy(
            str(path),
            algo=str(cfg["agent"].get("algo", "dqn")),
            name=method,
        )
        method_cfg = deepcopy(cfg)
        k = int(method_cfg["paths"]["k"])
        observation_dim = int(policy.model.observation_space.shape[0])
        no_risk_dim = k * 6 + k + 1
        risk_dim = k * 8 + k + 1
        if observation_dim == risk_dim:
            method_cfg.setdefault("agent", {})["use_risk"] = True
        elif observation_dim == no_risk_dim:
            method_cfg.setdefault("agent", {})["use_risk"] = False
        else:
            raise ValueError(
                f"Model {method!r} observation size {observation_dim} does not match "
                f"expected risk/no-risk sizes {risk_dim}/{no_risk_dim}"
            )
        steps, events = rollout(
            policy,
            method_cfg,
            seeds,
            build_risk(method_cfg),
            scenario=scenario,
        )
        steps_by_method[method] = steps
        event_records[method] = events

    selected: tuple[int, int, int] | None = None
    reference_events = event_records[next(iter(models))]
    for event in reference_events.itertuples(index=False):
        if event.kind not in {"link_failure", "node_failure"} or int(event.precursor) <= 0:
            continue
        link_ids = set(json.loads(event.links))
        frame = steps_by_method[next(iter(models))]
        prior = frame[
            (frame["scenario_seed"] == event.scenario_seed) & (frame["t"] < int(event.start))
        ]
        if prior.empty or not link_ids.intersection(json.loads(prior.iloc[-1]["path_links"])):
            continue
        selected = int(event.scenario_seed), int(event.start), int(event.precursor)
        break
    if selected is None:
        raise RuntimeError("No validation episode contains an active-path precursor failure")

    episode_seed, failure_tick, precursor = selected
    figure, axes = plt.subplots(
        len(models), 1, figsize=(9, 3.2 * len(models)), sharex=True, constrained_layout=True
    )
    if len(models) == 1:
        axes = [axes]
    for axis, (method, frame) in zip(axes, steps_by_method.items(), strict=True):
        episode = frame[
            (frame["scenario_seed"] == episode_seed)
            & (frame["t"] >= failure_tick - precursor - 5)
            & (frame["t"] <= failure_tick + 10)
        ].sort_values("t")
        axis.plot(episode["t"], episode["max_risk"], label="Max risk on selected path")
        axis.plot(episode["t"], episode["path_loss"], label="Path loss", alpha=0.8)
        axis.set_ylabel("Risk / loss")
        axis.set_title(method)
        action_axis = axis.twinx()
        action_axis.step(episode["t"], episode["action"], where="post", color="gray", alpha=0.55)
        action_axis.set_ylabel("Action (path)")
        axis.axvspan(failure_tick - precursor, failure_tick, color="tab:orange", alpha=0.15)
        axis.axvline(failure_tick, color="tab:red", linestyle=":")
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Simulator tick")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)
    return output_path


def main() -> None:
    import argparse
    from presto.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent/presto_dqn.yaml")
    parser.add_argument("--model")
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--out", default="results/diagnostics/risk_permutation.csv")
    parser.add_argument("--event-models", nargs="+", metavar="NAME=PATH")
    parser.add_argument("--event-out", default="results/diagnostics/event_trace.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    low, high = cfg["seeds"]["val"]
    seeds = list(range(int(low), int(high)))
    if args.model:
        result = risk_permutation_test(
            cfg,
            args.model,
            seeds,
            scenario=args.scenario,
            output=args.out,
        )
        print(result.to_string(index=False))
    if args.event_models:
        models = {}
        for item in args.event_models:
            if "=" not in item:
                parser.error(f"Event models must use NAME=PATH syntax: {item}")
            name, path = item.split("=", 1)
            if not name or not path:
                parser.error(f"Event models must use NAME=PATH syntax: {item}")
            models[name] = path
        trace = event_trace(cfg, models, seeds, scenario=args.scenario, output=args.event_out)
        print(f"Saved event trace to {trace}")
    if not args.model and not args.event_models:
        parser.error("Provide --model for risk permutation and/or --event-models for event trace")


if __name__ == "__main__":
    main()
