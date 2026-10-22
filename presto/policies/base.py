"""Common interface for routing policies."""

from typing import Any

import numpy as np


class Policy:
    name = "policy"

    def reset(self, info: dict[str, Any]) -> None:
        """Reset policy state after an environment reset."""

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        raise NotImplementedError
