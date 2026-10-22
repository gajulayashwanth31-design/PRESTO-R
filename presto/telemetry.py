"""Per-link ring buffer and derived telemetry features."""

import numpy as np

from presto.sim.simulator import LinkStats

RAW_FEATURES = ["delay_rel", "loss", "queue_rel", "util", "up"]


class Telemetry:
    """Store the most recent observed samples for every link."""

    def __init__(self, n_links: int, window: int, base_delay: np.ndarray, qmax: float):
        if n_links < 1 or window < 1:
            raise ValueError("Telemetry requires at least one link and one history tick")
        if base_delay.shape != (n_links,):
            raise ValueError("Base-delay vector has the wrong shape")
        self.n_links = n_links
        self.window_size = window
        self.base_delay = base_delay.astype(np.float64, copy=True)
        self.qmax = float(qmax)
        self._slope_weights = []
        for count in range(window + 1):
            times = np.arange(count, dtype=np.float64)
            centered = times - times.mean() if count else times
            denominator = np.dot(centered, centered)
            self._slope_weights.append(
                centered / denominator if denominator else np.zeros(count, dtype=np.float64)
            )
        self._buffer = np.zeros((window, n_links, len(RAW_FEATURES)), dtype=np.float64)
        self._count = 0
        self._next = 0

    def reset(self) -> None:
        self._buffer.fill(0.0)
        self._count = 0
        self._next = 0

    def push(self, obs: LinkStats) -> None:
        """Normalise and append one noisy link-measurement sample."""
        for field_name in ("delay", "loss", "queue", "util", "up"):
            if getattr(obs, field_name).shape != (self.n_links,):
                raise ValueError(f"Telemetry field {field_name!r} has the wrong shape")
        sample = self._buffer[self._next]
        sample[:, 0] = np.maximum(obs.delay / self.base_delay, 1.0)
        sample[:, 1] = np.clip(obs.loss, 0.0, 1.0)
        sample[:, 2] = np.clip(obs.queue / self.qmax, 0.0, 1.0)
        sample[:, 3] = np.maximum(obs.util, 0.0)
        sample[:, 4] = obs.up
        self._next = (self._next + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)

    def window(self) -> np.ndarray:
        """Return (links, window, features) in oldest-first order, zero-padded."""
        result = np.zeros((self.n_links, self.window_size, len(RAW_FEATURES)), dtype=np.float64)
        if not self._count:
            return result
        start = (self._next - self._count) % self.window_size
        indices = (start + np.arange(self._count)) % self.window_size
        result[:, -self._count :, :] = np.transpose(self._buffer[indices], (1, 0, 2))
        return result

    def summary(self) -> np.ndarray:
        """Return per-link last/mean/slope/max features plus up-flip count."""
        history = self.window()
        recent = history[:, -self._count :, :] if self._count else history
        if self._count:
            last = recent[:, -1, :]
            mean = recent.mean(axis=1)
            max_values = recent.max(axis=1)
            slope = np.einsum("w,ewf->ef", self._slope_weights[self._count], recent)
            flips = np.count_nonzero(np.diff(recent[:, :, 4], axis=1), axis=1, keepdims=True)
        else:
            last = mean = max_values = np.zeros((self.n_links, len(RAW_FEATURES)))
            slope = np.zeros_like(mean)
            flips = np.zeros((self.n_links, 1))
        return np.concatenate((last, mean, slope, max_values, flips), axis=1)

    def latest(self) -> np.ndarray:
        """Return the latest observed feature row per link, or zeros before first push."""
        if not self._count:
            return np.zeros((self.n_links, len(RAW_FEATURES)), dtype=np.float64)
        return self._buffer[(self._next - 1) % self.window_size].copy()
