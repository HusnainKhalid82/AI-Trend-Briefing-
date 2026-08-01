"""OpenRouter model catalogue — a purpose-built model-release tracker.

Keyless. OpenRouter lists new models very quickly, often before the lab's own
announcement post goes live, which makes this one of the earliest signals of a
launch available at zero cost.
"""

from __future__ import annotations

from datetime import timedelta

import httpx

from ..config import Config
from ..models import Item
from .base import clean_text, from_unix, utcnow

ENDPOINT = "https://openrouter.ai/api/v1/models"


async def collect(client: httpx.AsyncClient, config: Config) -> list[Item]:
    from ..http import get

    settings = config.source("openrouter")
    kind = settings.get("kind", "signal")
    window = timedelta(days=int(settings.get("new_within_days", 14)))
    since = utcnow() - window

    response = await get(client, ENDPOINT)
    payload = response.json()

    models = payload.get("data")
    if not isinstance(models, list):
        raise RuntimeError("unexpected response shape: no 'data' list")

    items: list[Item] = []
    for model in models:
        model_id = model.get("id")
        created = from_unix(model.get("created"))
        if not model_id or not created or created < since:
            continue

        name = model.get("name") or model_id
        pricing = model.get("pricing") or {}

        # A zero prompt price means the model is free to run, which is itself a
        # newsworthy detail and worth carrying into scoring later.
        is_free = str(pricing.get("prompt", "")).strip() in {"0", "0.0", "-1"}

        items.append(
            Item(
                source="openrouter",
                kind=kind,
                title=f"New model available: {name}",
                url=f"https://openrouter.ai/models/{model_id}",
                published=created,
                summary=clean_text(model.get("description")),
                author=model_id.split("/")[0] if "/" in model_id else None,
                raw_id=model_id,
                extra={
                    "context_length": model.get("context_length"),
                    "pricing_prompt": pricing.get("prompt"),
                    "is_free": is_free,
                },
            )
        )

    return items
