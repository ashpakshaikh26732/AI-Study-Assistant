"""Central configuration loading.

Layers, later ones win:
  1. ``config.yaml``                      (committed defaults)
  2. ``config.local.yaml``                (git-ignored personal overrides)
  3. file named by ``$STUDY_ASSISTANT_CONFIG``
  4. environment shortcuts (used by the Colab notebook):
       STUDY_ASSISTANT_DATA_DIR   - replaces the leading ``data/`` of every data path
       STUDY_ASSISTANT_INDEX_DIR  - where the vector store lives
       STUDY_ASSISTANT_LLM_PROVIDER

Relative paths are resolved against the project root (never the current
directory), so the app behaves the same on a laptop and on Colab.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"

# Config keys holding filesystem paths.
_PATH_KEYS = [
    ("data", "raw_path"),
    ("data", "processed_path"),
    ("data", "cache_path"),
    ("data", "ocr_file_path"),
    ("memory", "sqlite_database_path"),
    ("rag_core", "database", "persist_directory"),
]


def deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` updated recursively with ``override`` (inputs untouched)."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _get(cfg: dict, keys: tuple):
    for key in keys:
        cfg = cfg.get(key) if isinstance(cfg, dict) else None
    return cfg


def _set(cfg: dict, keys: tuple, value) -> None:
    for key in keys[:-1]:
        cfg = cfg.setdefault(key, {})
    cfg[keys[-1]] = value


def resolve_path(value: str, data_dir: str | None = None) -> str:
    """Absolute version of a config path (see module docstring for the rules)."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    if data_dir and path.parts and path.parts[0] == "data":
        return str(Path(data_dir).expanduser().joinpath(*path.parts[1:]))
    return str(PROJECT_ROOT / path)


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the layered configuration and return it with absolute paths."""
    base_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")
    cfg = _load_yaml(base_path)

    extra_files = [LOCAL_CONFIG_PATH, os.environ.get("STUDY_ASSISTANT_CONFIG")]
    for extra in extra_files:
        if extra and Path(extra).exists():
            cfg = deep_merge(cfg, _load_yaml(Path(extra)))

    provider = os.environ.get("STUDY_ASSISTANT_LLM_PROVIDER")
    if provider:
        cfg.setdefault("llm", {})["provider"] = provider

    data_dir = os.environ.get("STUDY_ASSISTANT_DATA_DIR")
    for keys in _PATH_KEYS:
        value = _get(cfg, keys)
        if value:
            _set(cfg, keys, resolve_path(value, data_dir))

    index_dir = os.environ.get("STUDY_ASSISTANT_INDEX_DIR")
    if index_dir:
        _set(cfg, ("rag_core", "database", "persist_directory"), str(Path(index_dir).expanduser()))
    return cfg
