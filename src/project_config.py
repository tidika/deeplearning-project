"""Helpers for loading the shared project configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

DEFAULT_CONFIG_PATH = Path("configs/data_config.example.json")


def load_project_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_config_value(config: Dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def get_nested_config_value(config: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_image_size(config: Dict[str, Any], default: Tuple[int, int] = (224, 224)) -> Tuple[int, int]:
    value = config.get("image_size", list(default))
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("config.image_size must be an int or a two-element list")


def get_image_normalization(
    config: Dict[str, Any],
    default_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    default_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mean = get_nested_config_value(config, ["image_normalization", "mean"], list(default_mean))
    std = get_nested_config_value(config, ["image_normalization", "std"], list(default_std))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("config.image_normalization mean/std must each contain 3 values")
    return (
        (float(mean[0]), float(mean[1]), float(mean[2])),
        (float(std[0]), float(std[1]), float(std[2])),
    )
