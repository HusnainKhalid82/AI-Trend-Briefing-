"""NewsData.io — breadth of coverage across outlets.

Two distinct jobs, and they want opposite query strategies:

  Collection  — `qInTitle`, because a plain `q` search matches any article that
                mentions AI in passing. Measured live: `q` returned 3,159
                results dominated by stock reports; `qInTitle` returned 50 that
                were actually about AI.

  Saturation  — plain `q`, because there we *want* loose matching. The question
                is "how many outlets have touched this story", and recall
                matters more than precision. Lives in saturation.py.

Free tier is 200 credits/day, so collection is deliberately a handful of calls.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from ..config import Config
from ..models import Item
from .base import clean_text, cutoff, from_iso

ENDPOINT = "https://newsdata.io/api/1/latest"


def api_key() -> str | None:
    key = (os.getenv("NEWSDATA_API_KEY") or "").strip()
    return key or None


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    key = api_key()
    if not key:
        raise RuntimeError("NEWSDATA_API_KEY is not set")

    settings = config.source("newsdata")
    kind = settings.get("kind", "record")
    queries: list[str] = settings.get("queries", ["artificial intelligence"])
    since = cutoff(config.lookback_for("newsdata"))

    async def run_query(term: str) -> list[dict]:
        response = await get(
            client,
            ENDPOINT,
            params={
                "apikey": key,
                "qInTitle": term,
                "language": "en",
            },
        )
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"newsdata error: {str(payload)[:160]}")
        return payload.get("results") or []

    batches = await asyncio.gather(
        *(run_query(term) for term in queries), return_exceptions=True
    )

    articles: list[dict] = []
    failures = 0
    for batch in batches:
        if isinstance(batch, BaseException):
            failures += 1
            continue
        articles.extend(batch)

    if failures == len(queries):
        raise RuntimeError(f"all {failures} NewsData queries failed")

    items: dict[str, Item] = {}
    for article in articles:
        title = article.get("title")
        link = article.get("link")
        published = from_iso((article.get("pubDate") or "").replace(" ", "T"))
        if not title or not link or not published or published < since:
            continue

        article_id = article.get("article_id") or link
        items[article_id] = Item(
            source="newsdata",
            kind=kind,
            title=title,
            url=link,
            published=published,
            summary=clean_text(article.get("description")),
            author=article.get("source_id"),
            raw_id=article_id,
            extra={
                "outlet": article.get("source_id"),
                "country": article.get("country"),
            },
        )

    return list(items.values())
