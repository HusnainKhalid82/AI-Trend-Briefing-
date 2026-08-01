"""The Guardian Open Platform.

Low volume as a news source — roughly ten AI stories in two days — so this is
not where breadth comes from. Its value is `bodyText`: full article bodies of
20,000-plus characters, which no other free news API returns. Sprint 3 uses
that for context; Sprint 2 uses the articles as corroboration.

Restricted to the technology section deliberately. Plain boolean search across
the whole paper returns gilt-yield reports and obituaries that happen to
contain the words "artificial intelligence".
"""

from __future__ import annotations

import os

import httpx

from ..config import Config
from ..models import Item
from .base import clean_text, cutoff, from_iso

ENDPOINT = "https://content.guardianapis.com/search"


def api_key() -> str | None:
    key = (os.getenv("GUARDIAN_API_KEY") or "").strip()
    return key or None


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    key = api_key()
    if not key:
        raise RuntimeError("GUARDIAN_API_KEY is not set")

    settings = config.source("guardian")
    kind = settings.get("kind", "record")
    since = cutoff(config.lookback_for("guardian"))

    params = {
        "q": (
            '"artificial intelligence" OR "machine learning" OR '
            '"large language model" OR OpenAI OR Anthropic OR DeepMind OR chatbot'
        ),
        "from-date": since.date().isoformat(),
        "order-by": "newest",
        "page-size": int(settings.get("page_size", 30)),
        "show-fields": "trailText,bodyText",
        "api-key": key,
    }
    if settings.get("section"):
        params["section"] = settings["section"]

    response = await get(client, ENDPOINT, params=params)
    payload = response.json().get("response", {})

    if payload.get("status") != "ok":
        raise RuntimeError(f"guardian error: {str(payload)[:160]}")

    items: list[Item] = []
    for result in payload.get("results", []):
        title = result.get("webTitle")
        url = result.get("webUrl")
        published = from_iso(result.get("webPublicationDate"))
        if not title or not url or not published or published < since:
            continue

        fields = result.get("fields") or {}
        body = fields.get("bodyText") or ""

        items.append(
            Item(
                source="guardian",
                kind=kind,
                title=title,
                url=url,
                published=published,
                summary=clean_text(fields.get("trailText") or body),
                raw_id=result.get("id"),
                extra={
                    # Carried through for Sprint 3; the summariser reads this
                    # rather than re-fetching the article.
                    "body_text": body,
                    "body_chars": len(body),
                    "section": result.get("sectionName"),
                },
            )
        )

    return items
