"""RSS and Atom feeds — official company announcements and curated aggregation.

Fetched with httpx rather than letting feedparser do its own networking, so
timeouts, retries and the User-Agent behave like every other collector.
"""

from __future__ import annotations

import asyncio
import logging

import feedparser
import httpx

from ..config import Config, build_keyword_matcher
from ..models import Item
from .base import clean_text, cutoff, from_struct_time

log = logging.getLogger(__name__)


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    settings = config.source("rss")
    feeds: list[dict] = settings.get("feeds", [])
    since = cutoff(config.lookback_for("rss"))
    matches = build_keyword_matcher(config.keywords)

    async def fetch_one(feed: dict) -> list[Item]:
        name = feed.get("name", "rss")
        response = await get(client, feed["url"])

        # feedparser is CPU-bound and synchronous; keep it off the event loop.
        parsed = await asyncio.to_thread(feedparser.parse, response.content)

        if not parsed.entries:
            # Several AI company "feeds" return 200 with an HTML page. Without
            # this check they would look like a healthy but quiet feed forever.
            raise RuntimeError(
                f"no entries — served {response.headers.get('content-type', 'unknown')}, "
                "feed may have moved or now returns HTML"
            )

        topical = bool(feed.get("topical", False))
        kind = feed.get("kind", "record")
        out: list[Item] = []

        for entry in parsed.entries:
            title = entry.get("title")
            link = entry.get("link")
            if not title or not link:
                continue

            published = from_struct_time(
                entry.get("published_parsed") or entry.get("updated_parsed")
            )
            # A feed with no dates at all is still useful; treat undated entries
            # as current rather than discarding them.
            if published is None:
                published = since
            elif published < since:
                continue

            summary = clean_text(entry.get("summary") or entry.get("description"))

            # AI-only feeds skip filtering. General feeds must mention something
            # relevant in the title or summary.
            if not topical and not matches(f"{title} {summary or ''}"):
                continue

            out.append(
                Item(
                    source=f"rss:{name}",
                    kind=kind,
                    title=title,
                    url=link,
                    published=published,
                    summary=summary,
                    author=entry.get("author"),
                    raw_id=entry.get("id") or entry.get("guid"),
                )
            )

        return out

    results = await asyncio.gather(
        *(fetch_one(feed) for feed in feeds), return_exceptions=True
    )

    items: list[Item] = []
    broken: list[str] = []
    for feed, result in zip(feeds, results):
        if isinstance(result, BaseException):
            broken.append(f"{feed.get('name')}: {type(result).__name__}: {result}")
            continue
        items.extend(result)

    for failure in broken:
        log.warning("feed failed — %s", failure)

    if broken and not items:
        raise RuntimeError(f"all {len(feeds)} feeds failed; first: {broken[0]}")

    return items
