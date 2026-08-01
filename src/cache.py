"""A tiny disk cache for API lookups.

Exists because NewsData enforces a short-window burst limit of 60 requests
separately from its 200/day credit budget. A once-daily production run never
approaches either, but iterating on the scorer means running the pipeline
repeatedly within minutes, which trips the burst limit and silently empties
every saturation reading.

Caching also means re-running to inspect a ranking change costs nothing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


class JsonCache:
    def __init__(self, name: str, ttl_seconds: int = 6 * 3600) -> None:
        self.path = CACHE_DIR / f"{name}.json"
        self.ttl = ttl_seconds
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def get(self, key: str):
        entry = self._data.get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > self.ttl:
            return None
        return entry.get("value")

    def set(self, key: str, value) -> None:
        self._data[key] = {"value": value, "ts": time.time()}

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        tmp.replace(self.path)
