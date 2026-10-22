"""Compare learned and oracle risk on validation-episode failure traces."""

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from presto.config import load_config
from presto.env import RoutingEnv
from presto.risk import LearnedRisk, OracleRisk


def _validation_episodes(
    dataset_path: str | Path, scenario: str, n_episodes: int
) -> list[tuple[int, str]]:
    with np.load(dataset_path, allow_pickle=False) as data:
        required = ("val_meta", "val_scenario", "val_topology")
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"Dataset is missing validation arrays: {', '.join(missing)}")
        selected = data["val_scenario"] == scenario
        episodes = list(
            dict.fromkeys(
                (int(seed), str(topology))
                for seed, topology in zip(
                    data["val_meta"][selected, 0],
                    data["val_topology"][selected],
                    strict=True,
                )
            )
        )
    if len(episodes) < n_episodes:
        raise ValueError(
            f"Validation split has {len(episodes)} {scenario!r} episodes; "
            f"{n_episodes} requested"
        )
    return episodes[:n_episodes]


def plot_risk_trace(
    cfg: dict,
    model_path: str | Path,
    output_path: str | Path,
    dataset_path: str | Path,
    scenario: str = "mixed",
    n_episodes: int = 20,
) -> Path:
    """Run validation episodes and save risk curves around one observed failure."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    episodes = _validation_episodes(dataset_path, scenario, n_episodes)
    horizon = int(cfg["predictor"]["horizon"])
    traces = []
    for topology in dict.fromkeys(topology for _, topology in episodes):
        episode_cfg = deepcopy(cfg)
        topology_cfg = load_config(Path("configs") / "topo" / f"{topology}.yaml")
        episode_cfg["topology"] = topology_cfg["topology"]
        env = RoutingEnv(episode_cfg, risk=LearnedRisk(str(model_path)))
        try:
            for seed, episode_topology in episodes:
                if episode_topology != topology:
                    continue
                _, reset_info = env.reset(
                    seed=seed, options={"scenario_seed": seed, "scenario": scenario}
                )
                oracle = OracleRisk(horizon)
                oracle.reset(env.sim)
                candidate_events = [
                    event
                    for event in reset_info["events"]
                    if event.kind in {"link_failure", "node_failure"}
                ]
                if not candidate_events:
                    continue
                event = next(
                    (item for item in candidate_events if item.precursor > 0),
                    candidate_events[0],
                )
                link = min(event.links)
                ticks = [int(reset_info["t"])]
                learned_values = [float(reset_info["risk"][link])]
                oracle_values = [float(oracle(env.telemetry, env.sim.t - 1)[link])]
                done = False
                while not done:
                    _, _, terminated, truncated, info = env.step(0)
                    ticks.append(int(info["t"]))
                    learned_values.append(float(info["risk"][link]))
                    oracle_values.append(float(oracle(env.telemetry, env.sim.t - 1)[link]))
                    done = terminated or truncated
                traces.append((seed, event, np.asarray(ticks), learned_values, oracle_values))
        finally:
            env.close()

    if not traces:
        seeds = [seed for seed, _ in episodes]
        raise RuntimeError(f"No link or node failure was observed in validation seeds {seeds}")
    chosen_trace = next((trace for trace in traces if trace[1].precursor > 0), traces[0])
    seed, event, ticks, learned_values, oracle_values = chosen_trace
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.plot(ticks, learned_values, label="Learned GRU risk")
    axis.plot(ticks, oracle_values, label="Oracle schedule risk", linestyle="--")
    failure_start = event.start
    axis.axvline(failure_start, color="tab:red", linestyle=":", label="Failure")
    if event.precursor:
        axis.axvspan(
            max(ticks[0], failure_start - event.precursor),
            failure_start,
            color="tab:orange",
            alpha=0.18,
            label="Precursor",
        )
    axis.axvspan(
        failure_start,
        failure_start + event.duration,
        color="tab:red",
        alpha=0.10,
        label="Down interval",
    )
    axis.set(xlabel="Simulator tick", ylabel="Link risk", ylim=(-0.02, 1.02))
    axis.legend()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    print(f"Risk trace uses validation seed {seed}, link {min(event.links)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/predictor/gru.yaml")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--out", default="results/predictor/risk_trace.png")
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    model_path = args.model or (
        Path(cfg["paths"]["models"])
        / "predictor"
        / f"gru_W{cfg['predictor']['window']}_H{cfg['predictor']['horizon']}.pt"
    )
    dataset_path = args.data or (
        Path(cfg["paths"]["data"])
        / f"predictor_W{cfg['predictor']['window']}_H{cfg['predictor']['horizon']}.npz"
    )
    path = plot_risk_trace(cfg, model_path, args.out, dataset_path, args.scenario, args.episodes)
    print(f"Saved risk trace to {path}")


if __name__ == "__main__":
    main()
