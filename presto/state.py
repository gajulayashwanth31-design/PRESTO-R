"""Candidate-path selection and normalized agent observations."""

from math import prod

import networkx as nx
import numpy as np

PATH_FEATURES = [
    "delay_norm",
    "loss",
    "bneck_queue",
    "bneck_util",
    "hops_norm",
    "max_risk",
    "mean_risk",
    "valid",
]


def candidate_paths(
    graph: nx.Graph, src: int, dst: int, k: int, lid: dict[tuple[int, int], int]
) -> list[list[int]]:
    """Get up to k loop-free paths, ordered by total base delay."""
    if src == dst or k < 1:
        raise ValueError("Candidate paths require distinct endpoints and k >= 1")
    paths: list[list[int]] = []
    try:
        node_paths = nx.shortest_simple_paths(graph, src, dst, weight="base_delay")
        for nodes in node_paths:
            paths.append([lid[(min(u, v), max(u, v))] for u, v in zip(nodes, nodes[1:])])
            if len(paths) == k:
                break
    except nx.NetworkXNoPath:
        return []
    return paths


def choose_flow(
    graph: nx.Graph, rng: np.random.Generator, k: int, lid: dict[tuple[int, int], int]
) -> tuple[int, int, list[list[int]]]:
    """Select endpoints with k alternatives and limited overlap between the best two."""
    nodes = np.asarray(sorted(graph.nodes), dtype=np.int64)
    for _ in range(10_000):
        src, dst = (int(value) for value in rng.choice(nodes, size=2, replace=False))
        paths = candidate_paths(graph, src, dst, k, lid)
        if len(paths) < k:
            continue
        shared = len(set(paths[0]) & set(paths[1]))
        if shared <= 0.5 * min(len(paths[0]), len(paths[1])):
            return src, dst, paths
    raise RuntimeError(f"Could not find {k} sufficiently diverse paths after 10000 flow samples")


def build_observation(
    latest: np.ndarray,
    risk: np.ndarray,
    paths: list[list[int]],
    current: int,
    steps_since_switch: int,
    ref_delay: float,
    max_hops: int,
    use_risk: bool,
    k: int,
) -> np.ndarray:
    """Build the fixed-size path-feature observation, including empty-path padding."""
    if k < 1 or ref_delay <= 0 or max_hops < 1:
        raise ValueError("k, ref_delay, and max_hops must be positive")
    if latest.ndim != 2 or latest.shape[1] != 5:
        raise ValueError("latest must have shape (links, 5)")
    if risk.shape != (latest.shape[0],):
        raise ValueError("risk must have one value per link")
    if not 0 <= current < k:
        raise ValueError("current path index is outside the action space")

    row_size = 8 if use_risk else 6
    rows: list[list[float]] = []
    for path in paths[:k]:
        if not path:
            rows.append([0.0] * row_size)
            continue
        if any(link < 0 or link >= latest.shape[0] for link in path):
            raise ValueError("Candidate path contains an invalid link id")
        path_delay = sum(float(latest[link, 0]) for link in path)
        path_loss = 1.0 - prod(1.0 - min(max(float(latest[link, 1]), 0.0), 1.0) for link in path)
        row = [
            min(max(path_delay / ref_delay, 0.0), 3.0) / 3.0,
            path_loss,
            max(float(latest[link, 2]) for link in path),
            min(max(max(float(latest[link, 3]) for link in path), 0.0), 2.0) / 2.0,
            len(path) / max_hops,
        ]
        if use_risk:
            path_risk = [float(risk[link]) for link in path]
            row.extend((max(path_risk), sum(path_risk) / len(path_risk)))
        row.append(float(all(latest[link, 4] > 0.5 for link in path)))
        rows.append(row)
    rows.extend([[0.0] * row_size for _ in range(k - len(rows))])
    encoded = np.asarray(rows, dtype=np.float32).reshape(k, row_size)
    current_one_hot = np.zeros(k, dtype=np.float32)
    current_one_hot[current] = 1.0
    result = np.concatenate(
        (
            encoded.ravel(),
            current_one_hot,
            np.asarray([min(max(steps_since_switch, 0), 20) / 20.0], dtype=np.float32),
        )
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)
