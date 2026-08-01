"""Helpers shared by collectors."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def cutoff(lookback_hours: int) -> datetime:
    return utcnow() - timedelta(hours=lookback_hours)


def from_unix(value: int | float | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def from_struct_time(parsed: time.struct_time | None) -> datetime | None:
    """Convert feedparser's parsed date, which is always already UTC."""
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def clean_text(value: str | None, limit: int = 400) -> str | None:
    """Flatten whitespace and truncate. Summaries are context, not content."""
    if not value:
        return None
    flat = " ".join(value.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"
