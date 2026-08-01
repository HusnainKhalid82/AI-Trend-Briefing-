"""Collector registry.

Each collector exposes `collect(client, config) -> list[Item]` and is
responsible only for fetching and normalising. Filtering, scoring and
clustering happen downstream, so a collector staying dumb is a feature.
"""

from __future__ import annotations

from . import (
    arxiv,
    guardian,
    hackernews,
    huggingface,
    lobsters,
    newsdata,
    openrouter,
    rss,
)

REGISTRY = {
    "hackernews": hackernews.collect,
    "rss": rss.collect,
    "huggingface": huggingface.collect,
    "openrouter": openrouter.collect,
    "lobsters": lobsters.collect,
    "arxiv": arxiv.collect,
    "guardian": guardian.collect,
    "newsdata": newsdata.collect,
}

__all__ = ["REGISTRY"]
