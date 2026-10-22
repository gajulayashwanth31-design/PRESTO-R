"""Shared GRU model for per-link telemetry histories."""

import numpy as np
import torch
from torch import nn


class LinkGRU(nn.Module):
    """Consume one link's telemetry window and emit a bad-within-horizon logit."""

    def __init__(
        self, n_features: int = 5, hidden: int = 32, layers: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1]).squeeze(-1)


@torch.no_grad()
def predict_probabilities(
    model: LinkGRU,
    windows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict probabilities in bounded batches to avoid large inference allocations."""
    model.eval()
    probabilities = []
    for start in range(0, len(windows), batch_size):
        batch = (windows[start : start + batch_size] - mean) / std
        inputs = torch.as_tensor(batch, dtype=torch.float32, device=device)
        probabilities.append(torch.sigmoid(model(inputs)).cpu().numpy())
    if not probabilities:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(probabilities)
