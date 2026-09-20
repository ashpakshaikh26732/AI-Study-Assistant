"""Tiny JSON file cache for generated study material (summaries, flashcards).

Generating these takes a while on CPU, so results are keyed by (kind, course,
model, the exact passages used) and reused until the notes or model change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def make_key(*parts: Any) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class JsonCache:
    def __init__(self, directory):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def cache_from_config(config: dict) -> JsonCache:
    return JsonCache(config["data"]["cache_path"])
