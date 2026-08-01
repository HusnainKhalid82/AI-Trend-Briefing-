"""Fetch article text for the stories that will be summarised.

Feed snippets are often a single sentence — the top story one day carried 386
characters total, which is not enough to write a real "context" paragraph from.
Fetching the article itself typically yields 1,500-3,000 characters instead.

The hard part is not fetching, it is knowing when the fetch produced rubbish.
Extractors return text from 404 pages, paywall interstitials and cookie
notices just as happily as from articles, and passing that to the summariser is
worse than passing nothing: the model will faithfully summarise the cookie
notice. Everything below exists to catch that.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
import trafilatura

from .cache import JsonCache
from .cluster import Cluster
from .config import Config

log = logging.getLogger(__name__)

# Sites whose pages are application UI rather than prose. GitHub pull requests
# extract to "Add this suggestion to a batch that can be applied as a single
# commit" — 900 characters of interface text that reads as content.
SKIP_DOMAINS = {
    "github.com", "gitlab.com", "huggingface.co", "openrouter.ai",
    "twitter.com", "x.com", "reddit.com", "news.ycombinator.com",
    "lobste.rs", "arxiv.org",
}

# Phrases that indicate an error page, paywall or consent wall rather than an
# article. Checked against the opening of the extraction, where they live.
BOILERPLATE_MARKERS = (
    "404", "page not found", "access denied", "subscribe to continue",
    "sign up for our newsletter", "enable javascript", "accept cookies",
    "you have unearthed", "add this suggestion", "this suggestion is invalid",
    "create an account", "log in to continue", "are you a robot",
    "verify you are human", "something went wrong",
)

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class EnrichReport:
    attempted: int = 0
    fetched: int = 0
    from_cache: int = 0
    rejected: int = 0
    reasons: list[str] = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []


def _domain(url: str) -> str:
    try:
        host = httpx.URL(url).host or ""
    except Exception:  # noqa: BLE001 — malformed URLs are just skipped
        return ""
    return host[4:] if host.startswith("www.") else host


def is_usable(text: str, min_chars: int) -> tuple[bool, str]:
    """Does this look like article prose rather than interface furniture?"""
    if not text:
        return False, "empty"
    if len(text) < min_chars:
        return False, f"too short ({len(text)}c)"

    opening = text[:400].lower()
    for marker in BOILERPLATE_MARKERS:
        if marker in opening:
            return False, f"boilerplate ({marker!r})"

    # Prose has sentences. A wall of navigation links or headings does not.
    if text.count(". ") < 3:
        return False, "no sentence structure"

    return True, "ok"


async def enrich(
    client: httpx.AsyncClient,
    clusters: list[Cluster],
    config: Config,
) -> EnrichReport:
    """Attach fetched article text to the top clusters, in place.

    Writes into `item.extra["body_text"]`, which is where the summariser
    already looks for Guardian full text — so nothing downstream changes.
    """
    settings = config.raw.get("enrich", {}) or {}
    report = EnrichReport()

    if not settings.get("enabled", True):
        return report

    limit = int(settings.get("max_stories", 4))
    min_chars = int(settings.get("min_chars", 600))
    max_chars = int(settings.get("max_chars", 8000))
    cache = JsonCache("articles", ttl_seconds=int(settings.get("cache_ttl_s", 172800)))

    semaphore = asyncio.Semaphore(3)

    async def fetch(cluster: Cluster) -> None:
        # Guardian already supplies full body text; nothing to gain.
        if any((i.extra or {}).get("body_text") for i in cluster.items):
            return

        # Prefer a source we can actually extract from. The highest-scoring
        # item is often a GitHub or HN link, while a sibling points at prose.
        candidates = [
            i for i in cluster.items
            if i.url.startswith("http") and _domain(i.url) not in SKIP_DOMAINS
        ]
        if not candidates:
            report.rejected += 1
            report.reasons.append(f"no extractable url: {cluster.title[:34]}")
            return

        target = candidates[0]
        report.attempted += 1

        cached = cache.get(target.url)
        if cached is not None:
            if cached:
                target.extra["body_text"] = cached
                report.from_cache += 1
                report.fetched += 1
            return

        async with semaphore:
            try:
                response = await client.get(
                    target.url,
                    headers={"User-Agent": BROWSER_UA},
                    follow_redirects=True,
                    timeout=15.0,
                )
                if response.status_code != 200:
                    report.rejected += 1
                    report.reasons.append(f"HTTP {response.status_code}: {_domain(target.url)}")
                    cache.set(target.url, "")
                    return

                text = await asyncio.to_thread(
                    trafilatura.extract,
                    response.text,
                    include_comments=False,
                    include_tables=False,
                )

                usable, reason = is_usable(text or "", min_chars)
                if not usable:
                    report.rejected += 1
                    report.reasons.append(f"{reason}: {_domain(target.url)}")
                    # Cache the rejection so tomorrow does not refetch it.
                    cache.set(target.url, "")
                    return

                body = text[:max_chars]
                target.extra["body_text"] = body
                cache.set(target.url, body)
                report.fetched += 1

            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                report.rejected += 1
                report.reasons.append(f"{type(exc).__name__}: {_domain(target.url)}")

    await asyncio.gather(*(fetch(c) for c in clusters[:limit]))
    cache.save()
    return report
