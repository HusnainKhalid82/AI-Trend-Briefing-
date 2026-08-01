"""Measure how much coverage a story already has.

This is the denominator of the ranking formula and the reason the system
surfaces early stories rather than popular ones. A story every outlet has
already run is worthless to a content producer no matter how large it is.

Uses a broad `q` search on purpose. Collection wants precision, so it uses
qInTitle; saturation wants recall — the question is "how many outlets have
touched this", and undercounting produces false "this is still fresh" signals.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx

from .cache import JsonCache
from .cluster import Cluster
from .collectors.newsdata import ENDPOINT, api_key
from .config import Config
from .entities import top_terms

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]")

# Words too generic to identify a story. Querying "AI model" measures the whole
# field's coverage, not this story's.
_GENERIC = {
    "ai", "new", "model", "models", "llm", "the", "a", "an", "and", "or", "to",
    "of", "in", "on", "for", "with", "is", "are", "how", "why", "what", "says",
    "could", "will", "can", "from", "at", "by", "its", "it", "this", "that",
    "artificial", "intelligence", "open", "source", "release", "released",
}

# Ordinary English that appears capitalised in Title Case headlines. Without
# this, "OpenAI Stages Investor Comeback" yields the query "OpenAI Stages" —
# "Stages" is a verb, and capitalisation carries no signal in a Title Case
# headline. Entity detection has to work on word shape, not case alone.
_COMMON = {
    "stages", "investor", "investors", "comeback", "surge", "fuel", "makes",
    "made", "make", "takes", "take", "gets", "get", "sets", "set", "puts",
    "adds", "adding", "brings", "gives", "goes", "comes", "looks", "shows",
    "reveals", "reports", "claims", "plans", "aims", "seeks", "faces", "hits",
    "wins", "loses", "cuts", "raises", "raised", "drops", "rises", "falls",
    "advice", "people", "users", "company", "companies", "startup", "startups",
    "firm", "firms", "group", "team", "teams", "week", "year", "years", "day",
    "days", "month", "months", "time", "times", "world", "global", "market",
    "markets", "industry", "business", "tech", "technology", "data", "study",
    "research", "report", "news", "update", "updates", "first", "next", "last",
    "best", "top", "big", "small", "more", "most", "less", "than", "after",
    "before", "about", "into", "over", "under", "between", "through", "during",
    "against", "without", "billion", "million", "percent", "here", "there",
    "when", "where", "who", "which", "while", "still", "just", "even", "also",
    "now", "then", "back", "down", "out", "off", "up", "but", "not", "was",
    "were", "been", "has", "had", "have", "may", "might", "would", "should",
    "their", "they", "them", "his", "her", "our", "your", "you", "we", "he",
    "she", "some", "many", "much", "every", "all", "any", "both", "each",
    "other", "another", "such", "own", "same", "why", "because", "since",
}


def _distinctiveness(token: str) -> int:
    """Score how well a token identifies a specific story.

    Word shape beats capitalisation. "OpenAI" and "CuspAI" have internal
    capitals; "GPT-5.6" and "K3" carry digits. Both are unmistakable entity
    markers that survive Title Case, which plain `istitle()` does not.
    """
    core = token.strip(".,:;!?\"'()[]—–-")
    if len(core) < 3:
        return 0

    low = core.lower()
    if low in _GENERIC or low in _COMMON:
        return 0

    score = 0
    if any(ch.isupper() for ch in core[1:]):
        score += 10          # OpenAI, DeepMind, CuspAI, GPT
    if any(ch.isdigit() for ch in core):
        score += 6           # K3, GPT-5.6, 450M
    if core[0].isupper():
        score += 3           # weak on its own, useful as a tiebreak
    if len(core) > 7:
        score += 1
    return score


def search_phrase(cluster: Cluster, max_terms: int = 2) -> str:
    """Build a query identifying this story specifically.

    Two terms, not more. NewsData ANDs the terms, and measurement showed the
    cliff is sharp: "Moonshot Kimi" returns 195 articles, "Moonshot Kimi
    suspends" returns 0. Over-specifying reports every story as uncovered,
    which silently disables the saturation mechanism entirely — the failure
    mode this whole function exists to prevent.

    Proper nouns first: they are what distinguish one story from another.
    """
    terms = top_terms(cluster.title, limit=max_terms)
    return " ".join(terms) if terms else cluster.title[:40]


@dataclass
class SaturationReport:
    """What actually happened, so degradation is visible rather than silent."""

    measured: int = 0
    from_cache: int = 0
    requests: int = 0
    rate_limited: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.rate_limited


async def measure(
    client: httpx.AsyncClient,
    clusters: list[Cluster],
    config: Config,
) -> SaturationReport:
    """Attach `saturation_count` to each cluster.

    Clusters are expected pre-sorted by preliminary score: only the top
    candidates are worth a lookup. Everything else keeps
    `saturation_count = None` and is treated as unmeasured rather than
    uncovered — assuming zero coverage would rocket unknown stories to the top.
    """
    report = SaturationReport()

    key = api_key()
    if not key:
        report.error = "NEWSDATA_API_KEY unset"
        return report

    settings = config.source("newsdata")
    budget = int(settings.get("max_saturation_lookups", 25))
    targets = clusters[:budget]

    cache = JsonCache("saturation", ttl_seconds=int(settings.get("cache_ttl_s", 21600)))

    # Conservative concurrency. NewsData enforces a 60-request burst window
    # separately from the daily credit budget, and tripping it returns 429s
    # that leave every story unmeasured.
    semaphore = asyncio.Semaphore(2)

    async def lookup(cluster: Cluster) -> None:
        phrase = search_phrase(cluster)
        cluster.components["saturation_query"] = phrase

        cached = cache.get(phrase)
        if cached is not None:
            cluster.saturation_count = int(cached)
            report.from_cache += 1
            report.measured += 1
            return

        # Once rate-limited, further requests only burn budget for nothing.
        if report.rate_limited:
            return

        async with semaphore:
            if report.rate_limited:
                return
            try:
                response = await client.get(
                    ENDPOINT,
                    params={"apikey": key, "q": phrase, "language": "en"},
                )
                report.requests += 1

                if response.status_code == 429:
                    report.rate_limited = True
                    remaining = response.headers.get("x-api-limit-remaining", "?")
                    report.error = (
                        f"burst limit hit (60/window); daily credits remaining: {remaining}"
                    )
                    return
                if response.status_code != 200:
                    return

                payload = response.json()
                if payload.get("status") != "success":
                    return

                count = int(payload.get("totalResults") or 0)
                cluster.saturation_count = count
                cache.set(phrase, count)
                report.measured += 1
            except (httpx.HTTPError, ValueError) as exc:
                log.debug("saturation lookup failed for %r: %s", phrase, exc)

    await asyncio.gather(*(lookup(c) for c in targets))
    cache.save()
    return report
