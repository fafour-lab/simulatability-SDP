"""Configuration helpers for the CGRR experiment scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Expand user paths and resolve relative paths under the project root."""
    path = Path(os.path.expandvars(str(path))).expanduser()
    if path.is_absolute():
        return path
    base = Path(root).expanduser() if root is not None else PROJECT_ROOT
    return base.joinpath(path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and normalize key project paths."""
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("project", {})
    project_root = resolve_path(cfg["project"].get("root_dir", PROJECT_ROOT))
    cfg["project"]["root_dir"] = str(project_root)
    cfg["project"]["config_path"] = str(path)
    cfg["project"]["output_dir"] = str(resolve_path(cfg["project"]["output_dir"], project_root))
    if "dataset" in cfg and "path" in cfg["dataset"]:
        cfg["dataset"]["path"] = str(resolve_path(cfg["dataset"]["path"], project_root))
    if "qwen" in cfg and "model_id" in cfg["qwen"]:
        cfg["qwen"]["model_id"] = _expand_if_path_like(cfg["qwen"]["model_id"])
    if "qwen" in cfg and cfg["qwen"].get("cache_dir"):
        cfg["qwen"]["cache_dir"] = str(resolve_path(cfg["qwen"]["cache_dir"], project_root))
    if "proxy" in cfg and "model_id" in cfg["proxy"]:
        cfg["proxy"]["model_id"] = _expand_if_path_like(cfg["proxy"]["model_id"])
    if "proxy" in cfg and cfg["proxy"].get("cache_dir"):
        cfg["proxy"]["cache_dir"] = str(resolve_path(cfg["proxy"]["cache_dir"], project_root))
    return cfg


def output_dir(cfg: dict[str, Any], *parts: str) -> Path:
    """Return an output path under the configured project output directory."""
    return Path(cfg["project"]["output_dir"]).joinpath(*parts)


def project_seed(cfg: dict[str, Any]) -> int:
    return int(cfg.get("project", {}).get("seed", 2027))


def _expand_if_path_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = os.path.expandvars(value)
    if value.startswith("~") or value.startswith("/"):
        return str(Path(value).expanduser())
    return value
