"""Configuration loading and the keyword filter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class Config:
    raw: dict

    @property
    def lookback_hours(self) -> int:
        return int(self.raw.get("lookback_hours", 48))

    @property
    def keywords(self) -> list[str]:
        return [k.lower() for k in self.raw.get("keywords", [])]

    def source(self, name: str) -> dict:
        return self.raw.get("sources", {}).get(name, {}) or {}

    def enabled(self, name: str) -> bool:
        return bool(self.source(name).get("enabled", False))

    def lookback_for(self, name: str) -> int:
        """Per-source lookback, falling back to the global window.

        Sources publish on different rhythms. arXiv does not announce at
        weekends, so a 48-hour window returns nothing at all on a Monday —
        which is exactly the bug this override exists to prevent.
        """
        return int(self.source(name).get("lookback_hours", self.lookback_hours))


def load_config(path: Path | None = None) -> Config:
    target = path or CONFIG_PATH
    with open(target, "r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle) or {})


def build_keyword_matcher(keywords: list[str]):
    """Return a predicate testing whether text mentions any keyword.

    Word-boundary matching, so "ai" does not fire on "chain" or "said" — a
    substring check here produces enormous false-positive volume.
    """
    if not keywords:
        return lambda _text: True

    pattern = re.compile(
        r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in keywords) + r")(?![a-z0-9])",
        re.IGNORECASE,
    )

    def matches(text: str) -> bool:
        return bool(text) and bool(pattern.search(text))

    return matches
