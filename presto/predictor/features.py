"""Label construction and predictor evaluation helpers."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def bad_link_mask(
    up: np.ndarray,
    loss: np.ndarray,
    delay: np.ndarray,
    base_delay: np.ndarray,
    bad_loss: float,
    bad_delay_factor: float,
) -> np.ndarray:
    """Apply the Phase 2 ground-truth definition to true link metrics."""
    if up.shape != loss.shape or up.shape != delay.shape or up.ndim != 2:
        raise ValueError("up, loss, and delay must share shape (ticks, links)")
    if base_delay.shape != (up.shape[1],):
        raise ValueError("base_delay must contain one value per link")
    return (~up.astype(bool)) | (loss > bad_loss) | (delay > bad_delay_factor * base_delay[None, :])


def sample_ticks(total_ticks: int, warmup: int, horizon: int) -> np.ndarray:
    """Return sample ticks with a full warm-up and complete future label horizon."""
    if horizon < 1 or warmup < 0:
        raise ValueError("warmup must be non-negative and horizon must be positive")
    stop = total_ticks - horizon
    return np.arange(warmup, max(warmup, stop), dtype=np.int32)


def build_future_labels(
    bad: np.ndarray, ticks: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Label any bad event in (t, t+H] and return onset status at t."""
    if bad.ndim != 2 or horizon < 1:
        raise ValueError("bad must have shape (ticks, links) and horizon must be positive")
    if ticks.size and (ticks.min() < 0 or ticks.max() + horizon >= bad.shape[0]):
        raise ValueError("Sample ticks do not have a complete future horizon")
    labels = np.zeros((len(ticks), bad.shape[1]), dtype=bool)
    for offset in range(1, horizon + 1):
        labels |= bad[ticks + offset]
    onset = ~bad[ticks]
    return labels, onset


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select an F1-maximizing threshold on validation data only."""
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have matching shapes")
    if labels.size == 0 or not labels.any():
        raise ValueError("Cannot tune an F1 threshold without positive validation samples")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(np.float64).eps)
    )
    return float(thresholds[int(np.argmax(f1))])
