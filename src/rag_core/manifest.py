"""Bookkeeping for the vector store: which files are indexed, and with what settings.

The manifest lets the indexer re-embed only new/changed files, and lets the
dashboard show library statistics instantly without querying the vector DB.
It lives next to the store as ``manifest.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MANIFEST_NAME = "manifest.json"
VERSION = 1


def manifest_path(config: dict) -> Path:
    return Path(config["rag_core"]["database"]["persist_directory"]) / MANIFEST_NAME


def index_settings(config: dict) -> dict:
    """Settings that change what the vectors mean; a change forces a full rebuild."""
    chunking = config["rag_core"]["chunking"]
    return {
        "chunk_size": chunking["chunk_size"],
        "chunk_overlap": chunking["chunk_overlap"],
        "embedding_model": config["rag_core"]["embedding"]["model_name"],
        "space": "cosine",
    }


def empty_manifest(config: dict) -> dict:
    return {"version": VERSION, "settings": index_settings(config), "files": {}}


def load_manifest(config: dict) -> dict:
    path = manifest_path(config)
    if not path.exists():
        return empty_manifest(config)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("files", {})
        return data
    except (json.JSONDecodeError, OSError):
        return empty_manifest(config)


def save_manifest(config: dict, manifest: dict) -> None:
    """Atomic write, so an interrupted run never leaves a half-written manifest."""
    path = manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def manifest_mtime(config: dict) -> float:
    """Modification time of the manifest (0 if absent) - handy as a cache key."""
    path = manifest_path(config)
    return path.stat().st_mtime if path.exists() else 0.0
