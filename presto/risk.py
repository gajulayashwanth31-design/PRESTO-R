"""Risk providers for current, oracle, and learned link degradation."""

from typing import TYPE_CHECKING

import numpy as np
import joblib
import torch

if TYPE_CHECKING:
    from presto.sim.simulator import NetworkSimulator
    from presto.telemetry import Telemetry


class RiskProvider:
    def reset(self, sim: "NetworkSimulator") -> None:
        self.sim = sim

    def __call__(self, telemetry: "Telemetry", t: int) -> np.ndarray:
        raise NotImplementedError


class ZeroRisk(RiskProvider):
    def __call__(self, telemetry: "Telemetry", t: int) -> np.ndarray:
        return np.zeros(telemetry.n_links, dtype=np.float64)


class OracleRisk(RiskProvider):
    """Use the known health and failure schedules as a debugging upper bound."""

    def __init__(self, horizon: int = 5):
        if horizon < 1:
            raise ValueError("Oracle horizon must be positive")
        self.horizon = horizon

    def __call__(self, telemetry: "Telemetry", t: int) -> np.ndarray:
        if not hasattr(self, "sim"):
            raise RuntimeError("Call reset(sim) before requesting oracle risk")
        schedules = self.sim.schedules
        start = max(0, t + 1)
        stop = min(schedules["up"].shape[0], start + self.horizon)
        if start >= stop:
            return np.zeros(telemetry.n_links, dtype=np.float64)
        future_down = np.any(~schedules["up"][start:stop], axis=0)
        future_degraded = np.any(schedules["health"][start:stop] < 1.0, axis=0)
        future_congestion = np.any(schedules["mult"][start:stop] > 1.0, axis=0)
        return np.logical_or.reduce((future_down, future_degraded, future_congestion)).astype(
            np.float64
        )


class LearnedRisk(RiskProvider):
    """Load a trained GRU and return one calibrated bad-link probability per link."""

    def __init__(self, path: str, device: str = "cpu"):
        from presto.predictor.gru import LinkGRU

        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model = LinkGRU(**checkpoint["model_kwargs"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(checkpoint["std"], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def __call__(self, telemetry: "Telemetry", t: int) -> np.ndarray:
        windows = telemetry.window()
        if windows.shape[-1] != self.mean.shape[0]:
            raise ValueError("Telemetry feature count does not match the trained GRU")
        inputs = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        probabilities = torch.sigmoid(self.model((inputs - self.mean) / self.std))
        return probabilities.cpu().numpy()


class LogRegRisk(RiskProvider):
    """Load the logistic-regression baseline and score telemetry summaries."""

    def __init__(self, path: str):
        checkpoint = joblib.load(path)
        self.model = checkpoint["model"] if isinstance(checkpoint, dict) else checkpoint

    def __call__(self, telemetry: "Telemetry", t: int) -> np.ndarray:
        probabilities = self.model.predict_proba(telemetry.summary())[:, 1]
        return np.clip(probabilities, 0.0, 1.0)
