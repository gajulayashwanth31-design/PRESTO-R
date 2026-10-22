"""Routing reward and its interpretable component terms."""

from dataclasses import dataclass


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 4.0
    gamma: float = 0.0
    lam: float = 0.1
    mu: float = 0.05


def compute_reward(
    qos: dict,
    switched: bool,
    ref_delay: float,
    max_hops: int,
    demand: float,
    w: RewardWeights,
) -> tuple[float, dict[str, float]]:
    """Score one action using normalized delay/loss, optional throughput, switching and hops."""
    if ref_delay <= 0 or max_hops < 1 or demand <= 0:
        raise ValueError("ref_delay, max_hops, and demand must be positive")
    delay_term = min(float(qos["delay"]) / ref_delay, 3.0) / 3.0
    loss_term = float(qos["loss"])
    throughput_term = float(qos["delivered"]) / demand
    switch_term = float(switched)
    energy_term = float(qos["hops"]) / max_hops
    terms = {
        "D": -w.alpha * delay_term,
        "L": -w.beta * loss_term,
        "T": w.gamma * throughput_term,
        "S": -w.lam * switch_term,
        "E": -w.mu * energy_term,
    }
    reward = terms["D"] + terms["L"] + terms["T"] + terms["S"] + terms["E"]
    return float(reward), terms
