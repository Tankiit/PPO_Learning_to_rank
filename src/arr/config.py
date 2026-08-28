from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .utils import deep_merge, parse_scalar, set_dotted, stable_hash, write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "arr" / "default.yaml"


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
    defaults_path: str | Path | None = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if defaults_path is not None and Path(defaults_path).exists():
        with Path(defaults_path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration root must be a mapping: {defaults_path}")
        config = loaded
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration root must be a mapping: {path}")
        config = deep_merge(config, loaded)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        set_dotted(config, key, parse_scalar(value))
    config["resolved_config_hash"] = stable_hash(config)
    return config


def save_resolved_config(config: dict[str, Any], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "resolved_config.json"
    write_json(path, config)
    return path
