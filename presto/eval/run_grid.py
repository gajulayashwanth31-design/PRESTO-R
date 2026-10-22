"""Run a resumable method × topology × scenario evaluation grid."""

import argparse
import glob
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import yaml

from presto.agents.train import build_risk
from presto.config import load_config, save_config
from presto.eval.rollout import rollout
from presto.policies.heuristics import GreedyDelay, RiskThreshold, SPFailover
from presto.policies.sb3_policy import SB3Policy

_HEURISTICS = {"SPFailover": SPFailover, "GreedyDelay": GreedyDelay, "RiskThreshold": RiskThreshold}


def _episode_config(cfg: dict, topology: str, scenario: str) -> dict:
    result = deepcopy(cfg)
    topology_cfg = load_config(Path("configs") / "topo" / f"{topology}.yaml")
    scenario_cfg = load_config(Path("configs") / "scenario" / f"{scenario}.yaml")
    result["topology"] = topology_cfg["topology"]
    result["scenario"] = scenario_cfg["scenario"]
    result["topology"]["name"] = topology
    result["scenario"]["name"] = scenario
    return result


def _apply_evaluation_overrides(cfg: dict, overrides: dict) -> dict:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"Invalid evaluation override key: {dotted_key!r}")
        node = cfg
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                raise ValueError(f"Evaluation override path does not exist: {dotted_key!r}")
            node = child
        if parts[-1] not in node:
            raise ValueError(f"Evaluation override key does not exist: {dotted_key!r}")
        node[parts[-1]] = deepcopy(value)
    return cfg


def _run_cell(task: dict) -> str:
    method = task["method"]
    cfg = task["cfg"]
    method_type = task["method_cfg"]["type"]
    if method_type == "heuristic":
        class_name = task["method_cfg"]["cls"]
        policy_type = _HEURISTICS.get(class_name)
        if policy_type is None:
            raise ValueError(f"Unsupported heuristic class: {class_name}")
        kwargs = task["method_cfg"].get("kwargs", {})
        policy = policy_type(**kwargs)
        risk = build_risk(cfg) if cfg["agent"].get("use_risk", False) else None
        training_seed = "deterministic"
    elif method_type == "sb3":
        run_directory = Path(task["run_directory"])
        model_path = run_directory / "best_model.zip"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        policy = SB3Policy(
            str(model_path),
            algo=str(cfg["agent"].get("algo", "dqn")),
            name=method,
        )
        risk = build_risk(cfg)
        training_seed = run_directory.name.removeprefix("seed_")
    else:
        raise ValueError(f"Unsupported method type for {method}: {method_type}")

    output_stem = (
        Path(task["out_dir"]) / method / task["topology"] / task["scenario"] / f"s{training_seed}"
    )
    steps_path = output_stem.with_suffix(".steps.parquet")
    events_path = output_stem.with_suffix(".events.parquet")
    if steps_path.is_file() and events_path.is_file():
        return f"SKIP {method}/{task['topology']}/{task['scenario']}/s{training_seed}"
    save_config(cfg, output_stem.with_suffix(".config.yaml"))
    rollout(
        policy,
        cfg,
        task["test_seeds"],
        risk,
        out_path=output_stem,
        scenario=task["scenario"],
    )
    hashes = {
        "steps_sha256": hashlib.sha256(steps_path.read_bytes()).hexdigest(),
        "events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
    }
    output_stem.with_suffix(".sha256.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    return f"DONE {method}/{task['topology']}/{task['scenario']}/s{training_seed}"


def build_tasks(grid: dict, workers: int | None = None) -> list[dict]:
    """Resolve requested cells and model paths before any evaluation starts."""
    tasks = []
    seed_start, seed_stop = map(int, grid["test_seeds"])
    test_seeds = list(range(seed_start, seed_stop))
    if not test_seeds or seed_start < 10000 or seed_stop > 10200:
        raise ValueError("Grid test_seeds must be within the held-out range [10000, 10200)")

    for topology in grid["topologies"]:
        for scenario in grid["scenarios"]:
            for method, method_cfg in grid["methods"].items():
                if topology not in method_cfg.get("topologies", grid["topologies"]):
                    continue
                if scenario not in method_cfg.get("scenarios", grid["scenarios"]):
                    continue
                if method_cfg["type"] == "heuristic":
                    cfg_path = method_cfg.get("config", "configs/agent/reactive_dqn.yaml")
                    cfg = _episode_config(load_config(cfg_path), topology, scenario)
                    _apply_evaluation_overrides(cfg, method_cfg.get("evaluation_overrides", {}))
                    cfg["agent"]["name"] = method
                    if method_cfg.get("risk_model"):
                        cfg["agent"]["use_risk"] = True
                        cfg["agent"]["risk_model"] = method_cfg["risk_model"]
                    tasks.append(
                        {
                            "method": method,
                            "method_cfg": method_cfg,
                            "cfg": cfg,
                            "topology": topology,
                            "scenario": scenario,
                            "test_seeds": test_seeds,
                            "out_dir": grid.get("out_dir", "results/raw"),
                        }
                    )
                    continue
                if method_cfg["type"] != "sb3":
                    raise ValueError(f"Unsupported method type for {method}")
                training_seeds = method_cfg.get("training_seeds_by_topology", {}).get(
                    topology,
                    method_cfg.get(
                        "training_seeds",
                        grid.get("training_seeds_by_topology", {}).get(
                            topology, grid.get("training_seeds")
                        ),
                    ),
                )
                if not training_seeds:
                    raise ValueError(f"No training seeds configured for topology {topology}")
                for seed in training_seeds:
                    patterns = method_cfg.get("run_glob_by_topology", {})
                    run_glob = patterns.get(topology, method_cfg.get("run_glob"))
                    if not run_glob:
                        raise ValueError(
                            f"No run_glob configured for {method} on topology {topology}"
                        )
                    pattern = str(run_glob).format(method=method, topo=topology, seed=seed)
                    matches = sorted(Path(path) for path in glob.glob(pattern))
                    if not matches:
                        raise FileNotFoundError(
                            f"No trained run matches {method}, topology={topology}, "
                            f"seed={seed}: {pattern}"
                        )
                    for run_directory in matches:
                        cfg_path = run_directory / "config.yaml"
                        if not cfg_path.is_file():
                            raise FileNotFoundError(f"Missing resolved run config: {cfg_path}")
                        cfg = _episode_config(load_config(cfg_path), topology, scenario)
                        _apply_evaluation_overrides(cfg, method_cfg.get("evaluation_overrides", {}))
                        tasks.append(
                            {
                                "method": method,
                                "method_cfg": method_cfg,
                                "cfg": cfg,
                                "topology": topology,
                                "scenario": scenario,
                                "test_seeds": test_seeds,
                                "out_dir": grid.get("out_dir", "results/raw"),
                                "run_directory": str(run_directory),
                            }
                        )
    return tasks


def run_grid(
    grid_path: str | Path,
    workers: int = 1,
    topologies: list[str] | None = None,
    methods: list[str] | None = None,
) -> list[str]:
    """Run all configured cells, skipping cells whose two parquet files exist."""
    if workers < 1:
        raise ValueError("workers must be positive")
    grid = yaml.safe_load(Path(grid_path).read_text(encoding="utf-8"))
    if topologies is not None:
        unknown = set(topologies) - set(grid["topologies"])
        if unknown:
            raise ValueError(f"Requested topologies are not in the grid: {sorted(unknown)}")
        grid["topologies"] = topologies
    if methods is not None:
        unknown = set(methods) - set(grid["methods"])
        if unknown:
            raise ValueError(f"Requested methods are not in the grid: {sorted(unknown)}")
        grid["methods"] = {name: grid["methods"][name] for name in methods}
    tasks = build_tasks(grid)
    outcomes = []
    if workers == 1:
        for task in tasks:
            outcome = _run_cell(task)
            print(outcome, flush=True)
            outcomes.append(outcome)
        return outcomes

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_cell, task): task for task in tasks}
        for future in as_completed(futures):
            outcome = future.result()
            print(outcome, flush=True)
            outcomes.append(outcome)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="configs/grid/main.yaml")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--topology", action="append", dest="topologies")
    parser.add_argument("--method", action="append", dest="methods")
    args = parser.parse_args()
    run_grid(args.grid, args.workers, args.topologies, args.methods)


if __name__ == "__main__":
    main()
