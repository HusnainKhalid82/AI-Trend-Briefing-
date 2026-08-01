"""Hacker News via the Algolia search API.

Keyless, documented at 10,000 requests/hour per IP. The strongest single early
indicator we have: points gained per hour in the first few hours predicts what
the wider internet discusses tomorrow.
"""

from __future__ import annotations

import asyncio

import httpx

from ..config import Config, build_keyword_matcher
from ..models import Item
from .base import cutoff, from_unix

ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"
ITEM_URL = "https://news.ycombinator.com/item?id={}"


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    settings = config.source("hackernews")
    kind = settings.get("kind", "signal")
    queries: list[str] = settings.get("queries", ["ai"])
    min_points = int(settings.get("min_points", 5))
    per_query = int(settings.get("limit_per_query", 40))
    since = int(cutoff(config.lookback_for("hackernews")).timestamp())

    # Algolia's matching is loose and prefix-based: a query of "ai" returns
    # "Airbus" and "Airport Simulator". Without this the top of the report
    # fills with high-velocity stories that have nothing to do with AI.
    matches = (
        build_keyword_matcher(config.keywords)
        if settings.get("filter_by_keywords", True)
        else lambda _text: True
    )

    async def run_query(term: str) -> list[dict]:
        response = await get(
            client,
            ENDPOINT,
            params={
                "query": term,
                "tags": "story",
                "numericFilters": f"created_at_i>{since},points>={min_points}",
                "hitsPerPage": per_query,
            },
        )
        return response.json().get("hits", [])

    # Queries run concurrently; Algolia's limit is far above our volume.
    batches = await asyncio.gather(
        *(run_query(term) for term in queries), return_exceptions=True
    )

    # One failing query must not lose the others. If every query failed, raise
    # so the collector is reported as down rather than silently empty.
    hits: list[dict] = []
    failures = 0
    for batch in batches:
        if isinstance(batch, BaseException):
            failures += 1
            continue
        hits.extend(batch)

    if failures == len(queries):
        raise RuntimeError(f"all {failures} Hacker News queries failed")

    items: dict[str, Item] = {}
    for hit in hits:
        object_id = str(hit.get("objectID", ""))
        title = hit.get("title") or hit.get("story_title")
        published = from_unix(hit.get("created_at_i"))
        if not object_id or not title or not published:
            continue
        if not matches(title):
            continue

        discussion = ITEM_URL.format(object_id)
        items[object_id] = Item(
            source="hackernews",
            kind=kind,
            title=title,
            # Ask HN and Show HN posts carry no external link; the thread is
            # the artefact in that case.
            url=hit.get("url") or discussion,
            published=published,
            discussion_url=discussion,
            score=hit.get("points"),
            comments=hit.get("num_comments"),
            author=hit.get("author"),
            raw_id=object_id,
        )

    return list(items.values())
