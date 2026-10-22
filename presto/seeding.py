"""Deterministic random stream and global library seeding helpers."""

import random

import numpy as np
import torch

STREAMS = {"topology": 0, "traffic": 1, "failures": 2, "noise": 3, "flow": 4}


def make_rng(seed: int, stream: str) -> np.random.Generator:
    """Create a deterministic NumPy generator isolated to a named stream."""
    return np.random.default_rng([seed, STREAMS[stream]])


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
