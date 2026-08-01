"""Orchestration: run every enabled collector concurrently, in isolation.

The guarantee this module provides is that one failing source degrades the
briefing rather than breaking the run. A briefing missing one source is useful;
no briefing at all is not.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .collectors import REGISTRY
from .config import Config
from .http import make_client
from .models import CollectorResult, Item

log = logging.getLogger(__name__)

# A collector exceeding this is treated as failed. Without it, one hung source
# would hold the entire run open until the workflow timeout.
COLLECTOR_TIMEOUT_S = 90.0


async def run_collectors(config: Config) -> list[CollectorResult]:
    enabled = [name for name in REGISTRY if config.enabled(name)]

    if not enabled:
        log.warning("no collectors enabled in config")
        return []

    async with make_client() as client:

        async def run_one(name: str) -> CollectorResult:
            started = time.perf_counter()
            try:
                items = await asyncio.wait_for(
                    REGISTRY[name](client, config), timeout=COLLECTOR_TIMEOUT_S
                )
                return CollectorResult(
                    name=name,
                    items=items,
                    duration_s=time.perf_counter() - started,
                )
            except asyncio.TimeoutError:
                return CollectorResult(
                    name=name,
                    error=f"timed out after {COLLECTOR_TIMEOUT_S:.0f}s",
                    duration_s=time.perf_counter() - started,
                )
            except Exception as exc:  # noqa: BLE001 — isolation is the point
                return CollectorResult(
                    name=name,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_s=time.perf_counter() - started,
                )

        return await asyncio.gather(*(run_one(name) for name in enabled))


def deduplicate(results: list[CollectorResult]) -> list[Item]:
    """Merge collector output, collapsing items that share a canonical URL.

    When two sources carry the same story we keep the richer record but count
    the corroboration, which Sprint 2's scorer depends on.
    """
    merged: dict[str, Item] = {}

    for result in results:
        for item in result.items:
            key = item.dedup_key
            existing = merged.get(key)

            if existing is None:
                item.extra.setdefault("seen_in", [item.source])
                merged[key] = item
                continue

            seen = existing.extra.setdefault("seen_in", [existing.source])
            if item.source not in seen:
                seen.append(item.source)

            # Prefer whichever copy carries engagement data.
            if existing.score is None and item.score is not None:
                existing.score = item.score
                existing.comments = item.comments
                existing.discussion_url = existing.discussion_url or item.discussion_url
            if not existing.summary and item.summary:
                existing.summary = item.summary

    return list(merged.values())


async def collect_all(config: Config) -> tuple[list[Item], list[CollectorResult]]:
    results = await run_collectors(config)
    return deduplicate(results), results
