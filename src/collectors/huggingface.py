"""Hugging Face — trending models and the daily papers listing.

Works anonymously (500 requests per 5-minute window, per IP). Model weights
appear here at the moment of shipping, frequently ahead of the announcement.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx

from ..config import Config
from ..models import Item
from .base import clean_text, from_iso, utcnow

MODELS_ENDPOINT = "https://huggingface.co/api/models"
PAPERS_ENDPOINT = "https://huggingface.co/api/daily_papers"


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    settings = config.source("huggingface")
    kind = settings.get("kind", "signal")
    limit = int(settings.get("trending_limit", 30))
    # Trending is a present-tense signal about models that may be weeks old,
    # so the global 48-hour window is the wrong filter here.
    since = utcnow() - timedelta(days=int(settings.get("new_within_days", 21)))

    async def trending_models() -> list[Item]:
        response = await get(
            client,
            MODELS_ENDPOINT,
            params={
                # Must be "trendingScore" in camelCase. The value the web UI
                # uses, "trending", is rejected with a 400.
                "sort": "trendingScore",
                "direction": "-1",
                "limit": limit,
            },
        )

        out: list[Item] = []
        for model in response.json():
            model_id = model.get("id") or model.get("modelId")
            created = from_iso(model.get("createdAt"))
            if not model_id or not created:
                continue

            # Trending includes long-standing popular models. Only genuinely
            # new ones are news.
            if created < since:
                continue

            out.append(
                Item(
                    source="huggingface:models",
                    kind=kind,
                    title=f"New model trending on Hugging Face: {model_id}",
                    url=f"https://huggingface.co/{model_id}",
                    published=created,
                    score=model.get("likes"),
                    author=model_id.split("/")[0] if "/" in model_id else None,
                    raw_id=model_id,
                    extra={
                        "downloads": model.get("downloads"),
                        "pipeline_tag": model.get("pipeline_tag"),
                        "trending_score": model.get("trendingScore"),
                    },
                )
            )
        return out

    async def daily_papers() -> list[Item]:
        if not settings.get("include_daily_papers", True):
            return []

        response = await get(client, PAPERS_ENDPOINT)

        out: list[Item] = []
        for entry in response.json():
            paper = entry.get("paper") or {}
            paper_id = paper.get("id")
            title = paper.get("title")
            # `publishedAt` is the paper's own publication date, not the date it
            # was featured — it is routinely weeks old. Date-filtering on it
            # empties this collector. Appearing on the curated daily list is
            # itself the signal, so we keep everything the endpoint returns and
            # let Sprint 5's history file prevent repeats.
            published = from_iso(entry.get("publishedAt") or paper.get("publishedAt"))
            if not paper_id or not title or not published:
                continue

            out.append(
                Item(
                    source="huggingface:papers",
                    kind=kind,
                    title=title,
                    url=f"https://huggingface.co/papers/{paper_id}",
                    published=published,
                    # Upvotes here are a genuine community-interest signal, not
                    # just a view count.
                    score=paper.get("upvotes"),
                    comments=entry.get("numComments"),
                    summary=clean_text(paper.get("summary")),
                    raw_id=paper_id,
                    extra={"arxiv_id": paper_id},
                )
            )
        return out

    models, papers = await asyncio.gather(
        trending_models(), daily_papers(), return_exceptions=True
    )

    items: list[Item] = []
    errors: list[str] = []
    for label, result in (("models", models), ("daily_papers", papers)):
        if isinstance(result, BaseException):
            errors.append(f"{label}: {result}")
            continue
        items.extend(result)

    if errors and not items:
        raise RuntimeError("; ".join(errors))

    return items
