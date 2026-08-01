"""arXiv — new papers in the AI categories.

Terms of use: one request every three seconds, one connection at a time, and
that limit applies across all machines under our control. We therefore issue a
single query covering every category rather than one per category.

Uses GET deliberately: arXiv serves GET from its Fastly cache, and POST has
been reported to draw 429s even at the documented rate.
"""

from __future__ import annotations

import asyncio

import feedparser
import httpx

from ..config import Config
from ..models import Item
from .base import clean_text, cutoff, from_struct_time

ENDPOINT = "https://export.arxiv.org/api/query"


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    settings = config.source("arxiv")
    kind = settings.get("kind", "signal")
    categories: list[str] = settings.get("categories", ["cs.AI"])
    max_results = int(settings.get("max_results", 60))
    since = cutoff(config.lookback_for("arxiv"))

    # One combined query keeps us to a single request against the 3s limit.
    search_query = " OR ".join(f"cat:{category}" for category in categories)

    response = await get(
        client,
        ENDPOINT,
        params={
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        },
    )

    parsed = await asyncio.to_thread(feedparser.parse, response.content)

    if not parsed.entries:
        raise RuntimeError(
            "no entries returned — arXiv tightened query validation in Nov 2025, "
            f"check search_query syntax: {search_query!r}"
        )

    items: list[Item] = []
    for entry in parsed.entries:
        title = entry.get("title")
        link = entry.get("link")
        published = from_struct_time(entry.get("published_parsed"))
        if not title or not link or not published or published < since:
            continue

        authors = [a.get("name") for a in entry.get("authors", []) if a.get("name")]

        items.append(
            Item(
                source="arxiv",
                kind=kind,
                title=title,
                url=link,
                published=published,
                summary=clean_text(entry.get("summary")),
                author=", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                raw_id=entry.get("id"),
                extra={
                    "categories": [t.get("term") for t in entry.get("tags", [])],
                    "author_count": len(authors),
                },
            )
        )

    return items
