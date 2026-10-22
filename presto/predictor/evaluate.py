"""Evaluate predictor calibration and onset performance on held-out episodes."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from presto.config import load_config
from presto.predictor.features import best_f1_threshold
from presto.predictor.gru import LinkGRU, predict_probabilities


def _predict(
    predictor_type: str,
    model_path: str | Path,
    windows: np.ndarray,
    summaries: np.ndarray,
    device: str,
) -> tuple[np.ndarray, float]:
    if predictor_type == "gru":
        target_device = torch.device(device)
        torch.set_num_threads(1)
        checkpoint = torch.load(model_path, map_location=target_device, weights_only=False)
        model = LinkGRU(**checkpoint["model_kwargs"]).to(target_device)
        model.load_state_dict(checkpoint["state_dict"])
        probabilities = predict_probabilities(
            model,
            windows,
            np.asarray(checkpoint["mean"], dtype=np.float32),
            np.asarray(checkpoint["std"], dtype=np.float32),
            target_device,
        )
        return probabilities, float(checkpoint["threshold"])
    if predictor_type == "logreg":
        checkpoint = joblib.load(model_path)
        return (
            checkpoint["model"].predict_proba(summaries)[:, 1],
            float(checkpoint["threshold"]),
        )
    raise ValueError(f"Unsupported predictor type: {predictor_type}")


def _median_lead_time(
    probabilities: np.ndarray,
    threshold: float,
    meta: np.ndarray,
    bad_now: np.ndarray,
    horizon: int,
) -> float:
    """Median ticks between an early threshold crossing and a later bad transition."""
    leads: list[int] = []
    episode_links = np.unique(meta[:, [0, 2]], axis=0)
    for seed, link in episode_links:
        rows = np.flatnonzero((meta[:, 0] == seed) & (meta[:, 2] == link))
        rows = rows[np.argsort(meta[rows, 1])]
        ticks = meta[rows, 1]
        bad = bad_now[rows]
        probabilities_for_link = probabilities[rows]
        transitions = np.flatnonzero(bad & np.concatenate(([True], ~bad[:-1])))
        for transition in transitions:
            bad_tick = int(ticks[transition])
            prior = (
                (ticks < bad_tick)
                & (ticks >= bad_tick - horizon)
                & (probabilities_for_link >= threshold)
            )
            if np.any(prior):
                leads.append(bad_tick - int(ticks[np.flatnonzero(prior)[0]]))
    return float(np.median(leads)) if leads else float("nan")


def _metric_row(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    model: str,
    subset: str,
    group: str,
    meta: np.ndarray,
    bad_now: np.ndarray,
    horizon: int,
    lead_time_probabilities: np.ndarray | None = None,
    lead_time_meta: np.ndarray | None = None,
    lead_time_bad_now: np.ndarray | None = None,
) -> dict:
    if labels.size == 0:
        return {
            "model": model,
            "subset": subset,
            "group": group,
            "threshold": threshold,
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "brier": float("nan"),
            "median_lead_time": float("nan"),
            "samples": 0,
        }
    labels = labels.astype(bool)
    predicted = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    has_both_classes = np.unique(labels).size == 2
    return {
        "model": model,
        "subset": subset,
        "group": group,
        "threshold": threshold,
        "pr_auc": float(average_precision_score(labels, probabilities)) if labels.any() else 0.0,
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if has_both_classes else float("nan")
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(brier_score_loss(labels, probabilities)),
        "median_lead_time": _median_lead_time(
            probabilities if lead_time_probabilities is None else lead_time_probabilities,
            threshold,
            meta if lead_time_meta is None else lead_time_meta,
            bad_now if lead_time_bad_now is None else lead_time_bad_now,
            horizon,
        ),
        "samples": int(labels.size),
    }


def evaluate_dataset(
    cfg: dict,
    dataset_path: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
) -> Path:
    """Write test metrics using thresholds tuned on matching validation subsets."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor_type = cfg["predictor"]["type"]
    horizon = int(cfg["predictor"]["horizon"])
    with np.load(dataset_path, allow_pickle=False) as data:
        val_probs, _ = _predict(
            predictor_type,
            model_path,
            data["val_windows"],
            data["val_summaries"],
            device,
        )
        val_labels = data["val_labels"]
        val_onset = data["val_onset"].astype(bool)
        threshold_all = best_f1_threshold(val_labels, val_probs)
        threshold_onset = best_f1_threshold(val_labels[val_onset], val_probs[val_onset])
        test_probs, _ = _predict(
            predictor_type,
            model_path,
            data["test_windows"],
            data["test_summaries"],
            device,
        )
        labels = data["test_labels"]
        onset = data["test_onset"].astype(bool)
        meta = data["test_meta"]
        bad_now = data["test_bad_now"].astype(bool)
        scenario = data["test_scenario"]
        topology = data["test_topology"]
        failure_type = data["test_failure_type"]
        val_groups = {
            "scenario": data["val_scenario"],
            "topology": data["val_topology"],
            "failure": data["val_failure_type"],
        }

    rows = [
        _metric_row(
            labels,
            test_probs,
            threshold_all,
            model=predictor_type,
            subset="all",
            group="all",
            meta=meta,
            bad_now=bad_now,
            horizon=horizon,
        ),
        _metric_row(
            labels[onset],
            test_probs[onset],
            threshold_onset,
            model=predictor_type,
            subset="onset",
            group="all",
            meta=meta,
            bad_now=bad_now,
            horizon=horizon,
            lead_time_probabilities=test_probs,
            lead_time_meta=meta,
            lead_time_bad_now=bad_now,
        ),
    ]
    for subset_name, test_groups in (("scenario", scenario), ("topology", topology)):
        validation_groups = val_groups[subset_name]
        for value in np.unique(test_groups):
            validation_selected = val_onset & (validation_groups == value)
            selected = onset & (test_groups == value)
            if not np.any(selected):
                continue
            group_threshold = best_f1_threshold(
                val_labels[validation_selected], val_probs[validation_selected]
            )
            rows.append(
                _metric_row(
                    labels[selected],
                    test_probs[selected],
                    group_threshold,
                    model=predictor_type,
                    subset="onset",
                    group=f"{subset_name}:{value}",
                    meta=meta[selected],
                    bad_now=bad_now[selected],
                    horizon=horizon,
                )
            )
    for value, name in ((1, "precursor"), (0, "sudden")):
        selected = onset & (failure_type == value)
        if not np.any(selected):
            continue
        validation_selected = val_onset & (val_groups["failure"] == value)
        group_threshold = best_f1_threshold(
            val_labels[validation_selected], val_probs[validation_selected]
        )
        rows.append(
            _metric_row(
                labels[selected],
                test_probs[selected],
                group_threshold,
                model=predictor_type,
                subset="onset",
                group=f"failure:{name}",
                meta=meta[selected],
                bad_now=bad_now[selected],
                horizon=horizon,
            )
        )

    metrics_path = output_dir / "metrics.csv"
    metrics = pd.DataFrame(rows)
    if metrics_path.is_file():
        previous = pd.read_csv(metrics_path)
        if "model" in previous:
            previous = previous[previous["model"] != predictor_type]
            metrics = pd.concat((previous, metrics), ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    calibration_mask = onset
    if np.any(calibration_mask):
        observed_fraction, mean_probability = calibration_curve(
            labels[calibration_mask],
            test_probs[calibration_mask],
            n_bins=10,
            strategy="uniform",
        )
        figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
        axis.plot(mean_probability, observed_fraction, marker="o", label=predictor_type)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        axis.set(xlabel="Mean predicted risk", ylabel="Observed bad-link frequency")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.legend()
        figure.savefig(output_dir / f"calibration_{predictor_type}.png", dpi=200)
        figure.savefig(output_dir / "calibration.png", dpi=200)
        plt.close(figure)
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/predictor/gru.yaml")
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--out-dir", default="results/predictor")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    dataset_path = args.data or (
        Path(cfg["paths"]["data"])
        / f"predictor_W{cfg['predictor']['window']}_H{cfg['predictor']['horizon']}.npz"
    )
    extension = "joblib" if cfg["predictor"]["type"] == "logreg" else "pt"
    model_path = args.model or (
        Path(cfg["paths"]["models"])
        / "predictor"
        / f"{cfg['predictor']['type']}_W{cfg['predictor']['window']}_"
        f"H{cfg['predictor']['horizon']}.{extension}"
    )
    metrics = evaluate_dataset(cfg, dataset_path, model_path, args.out_dir, args.device)
    print(f"Saved predictor metrics to {metrics}")


if __name__ == "__main__":
    main()
