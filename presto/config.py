"""Configuration loading and saving utilities."""

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively merged copy of two dictionaries."""
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_value(value: str) -> Any:
    """Parse a YAML scalar or collection used in a command-line override."""
    return yaml.safe_load(value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML defaults relative to ``configs/`` and apply dotted-key overrides."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration must contain a YAML mapping: {config_path}")

    defaults = raw.pop("defaults", [])
    if not isinstance(defaults, list):
        raise TypeError(f"'defaults' must be a list in {config_path}")

    cfg: dict[str, Any] = {}
    for default in defaults:
        if not isinstance(default, str):
            raise TypeError(f"Default paths must be strings in {config_path}")
        cfg = _deep_merge(cfg, load_config(Path("configs") / default))

    cfg = _deep_merge(cfg, raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override}")
        key, value = override.split("=", 1)
        parts = key.split(".")
        if not key or any(not part for part in parts):
            raise ValueError(f"Override must use a non-empty dotted key: {override}")

        node = cfg
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise TypeError(f"Cannot apply nested override beneath non-mapping key: {part}")
            node = child
        node[parts[-1]] = _parse_value(value)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write a configuration mapping as YAML, creating its parent directory if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
