"""Generate publication-ready figures from completed Phase 5 artifacts."""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

from presto.config import load_config
from presto.eval.diagnostics import event_trace
from presto.predictor.evaluate import _predict
from presto.predictor.trace import plot_risk_trace

METHOD_STYLE = {
    "sp_failover": ("#999999", "Shortest-path failover"),
    "reactive_dqn": ("#0072B2", "Reactive DQN"),
    "predictive_dqn": ("#E69F00", "Predictive DQN"),
    "presto_dqn": ("#009E73", "PRESTO-R"),
}
SCENARIOS = ("steady", "congestion", "node_failure", "fluctuation", "mixed")
FIGURE_DIR = Path("results/figures")


def _save(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_architecture() -> None:
    figure, axis = plt.subplots(figsize=(8.2, 3.3))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.02, 0.56, 0.17, 0.27, "Network simulator\n(topology, traffic,\nfailures)"),
        (0.23, 0.56, 0.17, 0.27, "Telemetry\n(windowed link\nmeasurements)"),
        (0.44, 0.56, 0.17, 0.27, "GRU predictor\n(link risk over H)"),
        (0.65, 0.56, 0.17, 0.27, "PRESTO-R policy\n(DQN + switch\npenalty)"),
        (0.83, 0.1, 0.15, 0.27, "Evaluation\n(PDR, delay,\nswitches)"),
    ]
    for x, y, width, height, label in boxes:
        axis.add_patch(
            plt.Rectangle(
                (x, y), width, height, facecolor="#EAF2F8", edgecolor="#34495E", linewidth=1.2
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8)
    arrows = [
        ((0.19, 0.695), (0.23, 0.695), "link state"),
        ((0.40, 0.695), (0.44, 0.695), "telemetry"),
        ((0.61, 0.695), (0.65, 0.695), "risk features"),
        ((0.82, 0.56), (0.88, 0.37), "route actions"),
        ((0.83, 0.21), (0.19, 0.56), "traffic / failure outcomes"),
    ]
    for start, end, label in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#34495E", "lw": 1.1},
        )
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        axis.text(*midpoint, label, ha="center", va="center", fontsize=7, color="#34495E")
    _save(figure, "F1_architecture")


def plot_predictor_quality() -> None:
    data_path = Path("data/predictor_W10_H5.npz")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing held-out predictor dataset: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        labels = data["test_labels"].astype(bool)
        onset = data["test_onset"].astype(bool)
        probabilities = {}
        for predictor in ("gru", "logreg"):
            model_path = Path("models/predictor") / (
                "gru_W10_H5.pt" if predictor == "gru" else "logreg_W10_H5.joblib"
            )
            scores, _ = _predict(
                predictor,
                model_path,
                data["test_windows"],
                data["test_summaries"],
                "cpu",
            )
            probabilities[predictor] = scores

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    styles = {"gru": ("#009E73", "GRU"), "logreg": ("#D55E00", "Logistic regression")}
    for model, (color, label) in styles.items():
        scores = probabilities[model]
        precision, recall, _ = precision_recall_curve(labels[onset], scores[onset])
        axes[0].plot(recall, precision, color=color, label=label)
        fraction_positive, mean_probability = calibration_curve(
            labels, scores, n_bins=10, strategy="quantile"
        )
        axes[1].plot(mean_probability, fraction_positive, "o-", color=color, label=label)
    axes[0].set(xlabel="Recall (onset samples)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    _save(figure, "F2_predictor_quality")


def plot_risk_trace_figure() -> None:
    cfg = load_config("configs/predictor/gru.yaml")
    plot_risk_trace(
        cfg,
        "models/predictor/gru_W10_H5.pt",
        FIGURE_DIR / "F3_risk_trace.png",
        "data/predictor_W10_H5.npz",
        scenario="mixed",
        n_episodes=20,
    )


def _seed_metrics(episodes: pd.DataFrame, topology: str, methods: set[str]) -> pd.DataFrame:
    selected = episodes[
        (episodes["topology"] == topology)
        & episodes["method"].isin(methods)
        & episodes["scenario"].isin(SCENARIOS)
    ]
    if selected.empty:
        raise ValueError(f"No {topology} test results for methods {sorted(methods)}")
    return selected.groupby(["method", "scenario", "training_seed"], as_index=False)[
        ["pdr", "recovery_time", "switches"]
    ].mean()


def plot_main_results(episodes: pd.DataFrame, topology: str) -> None:
    seed_metrics = _seed_metrics(episodes, topology, set(METHOD_STYLE))
    metrics = (
        ("pdr", "Packet delivery ratio"),
        ("recovery_time", "Recovery time (ticks)"),
        ("switches", "Route switches per episode"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.1), constrained_layout=True)
    x = np.arange(len(SCENARIOS))
    methods = [method for method in METHOD_STYLE if method in set(seed_metrics["method"])]
    width = 0.78 / max(len(methods), 1)
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        for index, method in enumerate(methods):
            values = seed_metrics[seed_metrics["method"] == method]
            means = [values.loc[values["scenario"] == s, metric].mean() for s in SCENARIOS]
            deviations = [values.loc[values["scenario"] == s, metric].std() for s in SCENARIOS]
            deviations = [0.0 if pd.isna(value) else value for value in deviations]
            offset = (index - (len(methods) - 1) / 2) * width
            color, display = METHOD_STYLE[method]
            axis.bar(
                x + offset,
                means,
                width,
                yerr=deviations,
                capsize=2,
                color=color,
                label=display,
                linewidth=0,
            )
        axis.set_ylabel(label)
        axis.set_xticks(x, SCENARIOS, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    _save(figure, f"F{'4' if topology == 'nsfnet' else '9'}_{topology}_results")


def plot_lambda_tradeoff(episodes: pd.DataFrame) -> None:
    mixed = episodes[(episodes["topology"] == "nsfnet") & (episodes["scenario"] == "mixed")]
    if mixed.empty:
        raise ValueError("No NSFNET mixed test results for the lambda trade-off figure")
    figure, axis = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
    for method in ("reactive_dqn", "predictive_dqn", "presto_dqn", "sp_failover"):
        selected = mixed[mixed["method"] == method]
        if selected.empty:
            continue
        seed_means = selected.groupby("training_seed")[["switches", "pdr"]].mean()
        color, display = METHOD_STYLE[method]
        axis.scatter(
            seed_means["switches"].mean(), seed_means["pdr"].mean(), color=color, label=display
        )
    lambdas = mixed[mixed["method"].str.startswith("presto_lam")]
    lambda_means = lambdas.groupby(["lambda", "method"])[["switches", "pdr"]].mean().reset_index()
    lambda_means = lambda_means.sort_values("lambda")
    axis.plot(
        lambda_means["switches"],
        lambda_means["pdr"],
        "o-",
        color="#CC79A7",
        label="PRESTO-R (lambda sweep)",
    )
    for _, row in lambda_means.iterrows():
        axis.annotate(
            f"λ={row['lambda']:.2g}",
            (row["switches"], row["pdr"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set(xlabel="Route switches per episode", ylabel="Packet delivery ratio")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7)
    _save(figure, "F5_lambda_tradeoff")


def plot_event_trace_figure() -> None:
    cfg = load_config("configs/agent/presto_dqn.yaml")
    event_trace(
        cfg,
        {
            "Reactive DQN": "runs/reactive_dqn/seed_0/best_model.zip",
            "Predictive DQN": "runs/predictive_dqn/seed_0/best_model.zip",
            "PRESTO-R": "runs/presto_dqn/seed_0/best_model.zip",
        },
        list(range(10000, 10020)),
        scenario="mixed",
        output=FIGURE_DIR / "F6_event_trace.png",
    )


def plot_horizon_sweep() -> None:
    path = Path("results/tables/horizon_sweep.csv")
    frame = pd.read_csv(path)
    required = {"horizon", "predictor_onset_f1", "proactive_rate_mean"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Horizon summary lacks columns: {sorted(required - set(frame.columns))}")
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)
    for scenario, color, label in (
        ("mixed", "#0072B2", "Mixed"),
        ("node_failure", "#D55E00", "Node failure"),
    ):
        selected = frame[frame["scenario"] == scenario].sort_values("horizon")
        if not selected.empty:
            axes[1].plot(
                selected["horizon"],
                selected["proactive_rate_mean"],
                "o-",
                color=color,
                label=label,
            )
    predictor = frame[["horizon", "predictor_onset_f1"]].drop_duplicates().sort_values("horizon")
    axes[0].plot(predictor["horizon"], predictor["predictor_onset_f1"], "o-", color="#009E73")
    axes[0].set(xlabel="Prediction horizon H (ticks)", ylabel="Predictor onset F1")
    axes[1].set(xlabel="Prediction horizon H (ticks)", ylabel="PRESTO-R proactive rate")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    _save(figure, "F7_horizon_sweep")


def _training_curves(methods: dict[str, str]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    for method, pattern in methods.items():
        runs = sorted(Path(path) for path in glob.glob(pattern))
        evaluations = []
        switch_logs = []
        for run in runs:
            evaluation_path = run / "evaluations.npz"
            if evaluation_path.is_file():
                with np.load(evaluation_path, allow_pickle=False) as data:
                    evaluations.append(
                        (data["timesteps"].astype(float), data["results"].mean(axis=1))
                    )
            log_path = run / "log.csv"
            if log_path.is_file():
                log = pd.read_csv(log_path)
                if not log.empty:
                    switch_logs.append(
                        (log["timesteps"].to_numpy(float), log["switches"].to_numpy(float))
                    )
        color, display = METHOD_STYLE[method]
        for axis, curves, ylabel in (
            (axes[0], evaluations, "Mean validation reward"),
            (axes[1], switch_logs, "Switches per training episode"),
        ):
            if not curves:
                continue
            grid = np.unique(np.concatenate([curve[0] for curve in curves]))
            matrix = np.vstack(
                [np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves]
            )
            mean = np.nanmean(matrix, axis=0)
            deviation = np.nanstd(matrix, axis=0)
            axis.plot(grid, mean, color=color, label=display)
            axis.fill_between(grid, mean - deviation, mean + deviation, color=color, alpha=0.15)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Training timesteps")
            axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    _save(figure, "F8_learning_curves")


def generate_all() -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_parquet("results/episode_metrics.parquet")
    plot_architecture()
    plot_predictor_quality()
    plot_risk_trace_figure()
    plot_main_results(episodes, "nsfnet")
    plot_lambda_tradeoff(episodes)
    plot_event_trace_figure()
    plot_horizon_sweep()
    _training_curves(
        {
            "reactive_dqn": "runs/reactive_dqn/seed_[0-4]",
            "predictive_dqn": "runs/predictive_dqn/seed_[0-4]",
            "presto_dqn": "runs/presto_dqn/seed_[0-4]",
        }
    )
    plot_main_results(episodes, "rgg")
    return sorted(FIGURE_DIR.glob("F*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    outputs = generate_all()
    print(f"Generated {len(outputs)} figures under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
