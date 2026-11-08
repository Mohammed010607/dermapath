"""YAML configuration loading with single-level inheritance.

A per-model config (e.g. ``configs/unet.yaml``) names a base with ``extends: base.yaml``.
The base is loaded first and the model config is deep-merged on top, so every experiment
shares the same data/training settings and overrides only its ``model`` section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single ``extends`` reference if present."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    return cfg
