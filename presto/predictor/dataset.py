"""Build episode-separated per-link windows and future degradation labels."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from presto.config import load_config
from presto.predictor.features import bad_link_mask, build_future_labels, sample_ticks
from presto.seeding import make_rng
from presto.sim.simulator import NetworkSimulator
from presto.state import choose_flow
from presto.telemetry import Telemetry

_ARRAY_FIELDS = (
    "windows",
    "summaries",
    "labels",
    "onset",
    "meta",
    "scenario",
    "topology",
    "failure_type",
    "bad_now",
)


def _episode_config(cfg: dict, scenario: str, topology: str) -> dict:
    episode_cfg = deepcopy(cfg)
    scenario_cfg = load_config(Path("configs") / "scenario" / f"{scenario}.yaml")
    topology_cfg = load_config(Path("configs") / "topo" / f"{topology}.yaml")
    episode_cfg["scenario"] = scenario_cfg["scenario"]
    episode_cfg["topology"] = topology_cfg["topology"]
    return episode_cfg


def _failure_type_schedule(
    events: list, n_links: int, total_ticks: int, horizon: int
) -> np.ndarray:
    """Mark pre-failure samples as precursor or sudden for diagnostic breakdowns."""
    result = np.full((total_ticks, n_links), -1, dtype=np.int8)
    for event in events:
        if event.kind in {"link_failure", "node_failure"}:
            failure_tick = event.start
            precursor = event.precursor
        elif event.kind == "fluctuation" and event.params.get("ends_in_failure", False):
            failure_tick = event.start + event.duration
            precursor = event.duration
        else:
            continue
        start = max(0, failure_tick - (precursor if precursor else horizon) - horizon)
        stop = min(total_ticks, failure_tick)
        if start >= stop:
            continue
        category = 1 if precursor else 0
        for link in sorted(set(event.links)):
            target = result[start:stop, link]
            if category:
                target[:] = np.where(target == -1, category, np.maximum(target, category))
            else:
                target[target == -1] = category
    return result


def collect_episode(
    cfg: dict,
    scenario_seed: int,
    scenario: str | None = None,
    topology: str | None = None,
) -> dict[str, np.ndarray]:
    """Collect observed histories and true future labels for one deterministic episode."""
    scenario_name = scenario or cfg["scenario"]["name"]
    topology_name = topology or cfg["topology"]["name"]
    episode_cfg = _episode_config(cfg, scenario_name, topology_name)
    horizon = int(cfg["predictor"]["horizon"])
    window = int(cfg["predictor"]["window"])
    sim = NetworkSimulator(episode_cfg, scenario_seed)
    if int(episode_cfg["sim"]["warmup"]) < window:
        raise ValueError("Simulation warmup must be at least predictor.window")
    src, dst, paths = choose_flow(
        sim.graph,
        make_rng(scenario_seed, "flow"),
        int(episode_cfg["paths"]["k"]),
        sim.edge_ids,
    )
    candidate_links = {link for path in paths for link in path}
    sim.reset(src, dst, candidate_links)
    telemetry = Telemetry(sim.n_links, window, sim.base_delay, sim.qmax)
    telemetry.reset()

    total_ticks = sim.total_ticks
    true_loss = np.empty((total_ticks, sim.n_links), dtype=np.float32)
    true_delay = np.empty_like(true_loss)
    true_up = np.empty((total_ticks, sim.n_links), dtype=bool)
    ticks = sample_ticks(total_ticks, int(episode_cfg["sim"]["warmup"]), horizon)
    if not ticks.size:
        raise ValueError("Episode length must leave samples after the warmup and label horizon")
    windows: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    current_path = 0
    sample_tick_set = set(ticks.tolist())
    for tick in range(total_ticks):
        stats = sim.step(paths[current_path])
        true_loss[tick] = stats.loss
        true_delay[tick] = stats.delay
        true_up[tick] = stats.up
        observed = sim.observe(stats)
        telemetry.push(observed)
        if tick in sample_tick_set:
            windows.append(telemetry.window().astype(np.float32))
            summaries.append(telemetry.summary().astype(np.float32))

        valid = [bool(path) and bool(np.all(observed.up[path])) for path in paths]
        if not valid[current_path]:
            current_path = next((i for i, is_valid in enumerate(valid) if is_valid), current_path)

    bad = bad_link_mask(
        true_up,
        true_loss,
        true_delay,
        sim.base_delay,
        float(cfg["predictor"]["bad_loss"]),
        float(cfg["predictor"]["bad_delay_factor"]),
    )
    labels, onset = build_future_labels(bad, ticks, horizon)
    failure_types = _failure_type_schedule(sim.events, sim.n_links, total_ticks, horizon)[ticks]
    bad_now = bad[ticks]
    link_ids = np.tile(np.arange(sim.n_links, dtype=np.int32), len(ticks))
    sample_ticks_flat = np.repeat(ticks, sim.n_links)
    return {
        "windows": np.concatenate(windows, axis=0).astype(np.float32),
        "summaries": np.concatenate(summaries, axis=0).astype(np.float32),
        "labels": labels.reshape(-1).astype(np.int8),
        "onset": onset.reshape(-1).astype(bool),
        "meta": np.column_stack(
            (
                np.full(link_ids.shape, scenario_seed, dtype=np.int32),
                sample_ticks_flat,
                link_ids,
            )
        ),
        "scenario": np.full(link_ids.shape, scenario_name, dtype="U32"),
        "topology": np.full(link_ids.shape, topology_name, dtype="U16"),
        "failure_type": failure_types.reshape(-1).astype(np.int8),
        "bad_now": bad_now.reshape(-1).astype(bool),
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "positive": int(np.count_nonzero(labels)),
        "negative": int(labels.size - np.count_nonzero(labels)),
        "samples": int(labels.size),
    }


def build_dataset(cfg: dict, out_path: str | Path, seed: int = 0) -> Path:
    """Collect configured topology/scenario episodes and save episode-isolated splits."""
    data_cfg = cfg["predictor"]["data"]
    scenarios = list(data_cfg["scenario_mix"])
    topologies = list(data_cfg["topologies"])
    episodes_per_combo = int(data_cfg["episodes_per_combo"])
    if episodes_per_combo < 3:
        raise ValueError("At least three episodes per scenario/topology are required")
    combos = [(scenario, topology) for scenario in scenarios for topology in topologies]
    total_episodes = len(combos) * episodes_per_combo
    train_max = int(cfg["seeds"]["train_max"])
    if total_episodes > train_max:
        raise ValueError("Dataset episode count exceeds the configured training seed range")

    split_ratios = np.asarray(data_cfg["split_by_episode"], dtype=np.float64)
    if (
        split_ratios.shape != (3,)
        or np.any(split_ratios < 0)
        or not np.isclose(split_ratios.sum(), 1.0)
    ):
        raise ValueError("split_by_episode must contain three non-negative ratios summing to 1")
    n_train = int(episodes_per_combo * split_ratios[0])
    n_val = int(episodes_per_combo * split_ratios[1])
    if min(n_train, n_val, episodes_per_combo - n_train - n_val) < 1:
        raise ValueError("Each scenario/topology needs at least one episode in every split")

    rng = np.random.default_rng(seed)
    unique_seeds = rng.choice(train_max, size=total_episodes, replace=False).astype(np.int32)
    split_records: dict[str, dict[str, list[np.ndarray]]] = {
        name: {field: [] for field in _ARRAY_FIELDS} for name in ("train", "val", "test")
    }
    raw_counts = {name: {"positive": 0, "negative": 0, "samples": 0} for name in split_records}
    episode_counts = {name: 0 for name in split_records}
    seed_offset = 0
    for scenario, topology in combos:
        combo_seeds = unique_seeds[seed_offset : seed_offset + episodes_per_combo].copy()
        seed_offset += episodes_per_combo
        rng.shuffle(combo_seeds)
        assignments = [
            ("train", combo_seeds[:n_train]),
            ("val", combo_seeds[n_train : n_train + n_val]),
            ("test", combo_seeds[n_train + n_val :]),
        ]
        for split, episode_seeds in assignments:
            for scenario_seed in episode_seeds:
                episode = collect_episode(
                    cfg, int(scenario_seed), scenario=scenario, topology=topology
                )
                labels = episode["labels"]
                raw_counts[split]["positive"] += int(np.count_nonzero(labels))
                raw_counts[split]["negative"] += int(labels.size - np.count_nonzero(labels))
                raw_counts[split]["samples"] += int(labels.size)
                episode_counts[split] += 1
                if split == "train":
                    keep = labels.astype(bool) | (
                        rng.random(labels.size) < float(data_cfg["neg_subsample"])
                    )
                    episode = {key: value[keep] for key, value in episode.items()}
                for field in _ARRAY_FIELDS:
                    split_records[split][field].append(episode[field])

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {}
    kept_counts = {}
    for split, fields in split_records.items():
        for field, chunks in fields.items():
            archive[f"{split}_{field}"] = np.concatenate(chunks, axis=0)
        kept_counts[split] = _class_counts(archive[f"{split}_labels"])
    np.savez_compressed(output, **archive)

    report = {
        "seed": seed,
        "window": int(cfg["predictor"]["window"]),
        "horizon": int(cfg["predictor"]["horizon"]),
        "episodes": episode_counts,
        "raw_class_balance": raw_counts,
        "stored_class_balance": kept_counts,
        "negative_subsample_train_only": float(data_cfg["neg_subsample"]),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/predictor/gru.yaml")
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    default_out = Path(cfg["paths"]["data"]) / (
        f"predictor_W{cfg['predictor']['window']}_H{cfg['predictor']['horizon']}.npz"
    )
    output = build_dataset(cfg, args.out or default_out, seed=args.seed)
    print(f"Saved predictor dataset to {output}")
    print(output.with_suffix(".json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
