"""Lobste.rs — lower volume and far lower noise than Hacker News.

Keyless JSON. Useful as corroboration: a story on both Lobste.rs and Hacker
News has cleared two independent technical audiences.
"""

from __future__ import annotations

import asyncio

import httpx

from ..config import Config, build_keyword_matcher
from ..models import Item
from .base import cutoff, from_iso

TAG_ENDPOINT = "https://lobste.rs/t/{tag}.json"


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    settings = config.source("lobsters")
    kind = settings.get("kind", "signal")
    tags: list[str] = settings.get("tags", ["ai"])
    since = cutoff(config.lookback_for("lobsters"))

    # Lobsters was the only collector without a keyword filter, and it needs
    # one: its "ml" tag is the OCaml / Standard ML language family, not machine
    # learning, so functional-programming posts were reaching the briefing.
    matches = (
        build_keyword_matcher(config.keywords)
        if settings.get("filter_by_keywords", True)
        else lambda _text: True
    )

    async def fetch_tag(tag: str) -> list[dict]:
        response = await get(client, TAG_ENDPOINT.format(tag=tag))
        payload = response.json()
        return payload if isinstance(payload, list) else []

    batches = await asyncio.gather(
        *(fetch_tag(tag) for tag in tags), return_exceptions=True
    )

    stories: list[dict] = []
    failures = 0
    for batch in batches:
        if isinstance(batch, BaseException):
            failures += 1
            continue
        stories.extend(batch)

    if failures == len(tags):
        raise RuntimeError(f"all {failures} Lobsters tag requests failed")

    items: dict[str, Item] = {}
    for story in stories:
        short_id = story.get("short_id")
        title = story.get("title")
        published = from_iso(story.get("created_at"))
        if not short_id or not title or not published or published < since:
            continue
        if not matches(title):
            continue

        comments_url = story.get("comments_url") or ""
        items[short_id] = Item(
            source="lobsters",
            kind=kind,
            title=title,
            # Text-only submissions have an empty url field.
            url=story.get("url") or comments_url,
            published=published,
            discussion_url=comments_url,
            score=story.get("score"),
            comments=story.get("comment_count"),
            author=(story.get("submitter_user") or {}).get("username")
            if isinstance(story.get("submitter_user"), dict)
            else story.get("submitter_user"),
            raw_id=short_id,
            extra={"tags": story.get("tags")},
        )

    return list(items.values())
