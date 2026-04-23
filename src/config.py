"""Shared config loading for all pipeline scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config, merge base configs listed under 'defaults', and
    inject '_config_path' so the resolved path is always available downstream.

    Base configs are resolved relative to the directory containing config_path.
    The override config wins on every top-level key (base_cfg.update(override)).
    """
    path = Path(config_path)
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if "defaults" in cfg:
        base_cfg: dict[str, Any] = {}
        for default in cfg.pop("defaults"):
            base_file = path.parent / f"{default}.yaml"
            if base_file.exists():
                with open(base_file) as bf:
                    base_cfg.update(yaml.safe_load(bf) or {})
        base_cfg.update(cfg)
        cfg = base_cfg

    cfg["_config_path"] = str(path)
    return cfg
