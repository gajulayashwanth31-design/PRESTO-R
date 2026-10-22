"""Run one deterministic mixed-scenario episode and plot its true QoS and events."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from presto.config import load_config
from presto.env import RoutingEnv


def main() -> None:
    cfg = load_config("configs/scenario/mixed.yaml", ["agent.use_risk=false"])
    env = RoutingEnv(cfg)
    _, reset_info = env.reset(seed=0, options={"scenario_seed": 17, "scenario": "mixed"})
    rows = []
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        active = info["path_links"]
        rows.append(
            (
                info["qos"]["delay"],
                info["qos"]["loss"],
                float(np.min(env.sim.schedules["health"][info["t"], active])),
            )
        )

    values = np.asarray(rows)
    ticks = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].plot(ticks, values[:, 0])
    axes[0].set_ylabel("Path delay (ms)")
    axes[1].plot(ticks, values[:, 1])
    axes[1].set_ylabel("Path loss")
    axes[2].plot(ticks, values[:, 2])
    axes[2].set_ylabel("Minimum path health")
    for event in reset_info["events"]:
        start = max(0, event.start - event.precursor - int(cfg["sim"]["warmup"]))
        end = min(len(rows), event.start + event.duration - int(cfg["sim"]["warmup"]))
        if end >= 0 and start < len(rows):
            for axis in axes:
                axis.axvspan(start, max(start + 1, end), color="tab:red", alpha=0.12)
    axes[3].set_ylabel("Event")
    axes[3].set_yticks([])
    axes[3].set_xlabel("Episode tick")
    output = Path("results/smoke/episode.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    env.close()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
