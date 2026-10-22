"""Plot validation rewards and episode switch counts for one or more runs."""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _run_directories(patterns: list[str]) -> list[Path]:
    runs = sorted(
        {
            Path(match)
            for pattern in patterns
            for match in glob.glob(pattern)
            if Path(match).is_dir()
        }
    )
    if not runs:
        raise FileNotFoundError(f"No training run directories matched: {patterns}")
    return runs


def plot_training(runs: list[Path], output: Path) -> Path:
    validation = []
    switch_curves = []
    for run in runs:
        evaluations = run / "evaluations.npz"
        if evaluations.is_file():
            with np.load(evaluations, allow_pickle=False) as data:
                timesteps = data["timesteps"].astype(np.float64)
                rewards = data["results"].mean(axis=1)
            validation.append((timesteps, rewards))
        log = run / "log.csv"
        if log.is_file():
            frame = pd.read_csv(log)
            if not frame.empty:
                switch_curves.append(
                    (frame["timesteps"].to_numpy(), frame["switches"].to_numpy(dtype=float))
                )
    if not validation and not switch_curves:
        raise FileNotFoundError("Runs contain neither evaluations.npz nor log.csv")

    figure, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    if validation:
        grid = np.unique(np.concatenate([curve[0] for curve in validation]))
        values = np.vstack(
            [np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in validation]
        )
        mean = np.nanmean(values, axis=0)
        spread = np.nanstd(values, axis=0)
        axes[0].plot(grid, mean, label="Validation reward")
        axes[0].fill_between(grid, mean - spread, mean + spread, alpha=0.2)
    if switch_curves:
        for index, (x, y) in enumerate(switch_curves):
            axes[1].plot(x, y, alpha=0.65, label=f"Run {index + 1}")
        axes[1].set_ylabel("Switches per training episode")
    axes[0].set_ylabel("Mean validation reward")
    axes[0].set_xlabel("Timesteps")
    axes[1].set_xlabel("Timesteps")
    if switch_curves:
        axes[1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories or glob patterns")
    parser.add_argument("--out", default="results/training_curves.png")
    args = parser.parse_args()
    output = plot_training(_run_directories(args.runs), Path(args.out))
    print(f"Saved training curves to {output}")


if __name__ == "__main__":
    main()
