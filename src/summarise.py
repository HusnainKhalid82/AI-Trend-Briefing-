"""Turn ranked clusters into readable intelligence, using Gemini.

The model's job is deliberately narrow: read the source material we already
collected and restate it. It does not choose stories, rank them, or decide what
matters — that is done by inspectable code in score.py. Keeping the model out
of ranking is what makes the ordering debuggable.

Everything it writes must trace back to text we supplied. The prompt forbids
outside knowledge, and `confidence` lets a story flag its own thinness rather
than inventing detail to fill the fields.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass

import httpx

from .cache import JsonCache
from .cluster import Cluster
from .config import Config

log = logging.getLogger(__name__)

BASE = "https://generativelanguage.googleapis.com"

# Structured output. Without a schema the model drifts into prose wrappers and
# markdown fences that then need brittle parsing.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "what_happened": {
            "type": "string",
            "description": "2-3 sentences of plain fact. No adjectives of significance.",
        },
        "context": {
            "type": "string",
            "description": (
                "3-4 sentences of background needed to understand why this matters, "
                "drawn only from the supplied material."
            ),
        },
        "why_it_spreads": {
            "type": "string",
            "description": "1-2 sentences on the angle, tension or surprise.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "low when the supplied material is too thin to summarise",
        },
    },
    "required": ["what_happened", "context", "why_it_spreads", "confidence"],
}

PROMPT = """You are preparing a factual intelligence briefing about AI news for \
a single reader who writes their own commentary. You are NOT writing a script, \
caption, headline, or social media post.

Rules, in order of importance:
1. Use ONLY the source material below. Do not add facts from your own knowledge.
2. If the material does not support a claim, do not make it. Set confidence to \
"low" rather than filling space.
3. No hype. Do not call anything revolutionary, game-changing, or a breakthrough. \
State what happened and let the reader judge.
4. Do not speculate about consequences that the material does not raise.
5. Plain language. Explain jargon the first time it appears.

For "context": explain the background a reader needs to understand why this is \
significant — what came before, who the players are, what it competes with. This \
is the most valuable field. If the material gives you nothing beyond the headline, \
say so plainly and set confidence to "low".

For "why_it_spreads": identify the specific angle, tension, disagreement or \
surprise that would make people discuss this. If there isn't one, say so.

--- SOURCE MATERIAL ---
{material}
--- END SOURCE MATERIAL ---
"""


@dataclass
class Summary:
    what_happened: str
    context: str
    why_it_spreads: str
    confidence: str
    cached: bool = False

    @property
    def usable(self) -> bool:
        return self.confidence in {"high", "medium"} and bool(self.what_happened)


def build_material(cluster: Cluster, max_body_chars: int = 6000) -> str:
    """Assemble everything known about a cluster into the prompt payload."""
    lines = [
        f"HEADLINE: {cluster.title}",
        f"AGE: {cluster.age_hours:.1f} hours old",
        f"CARRIED BY: {', '.join(cluster.sources)}",
    ]

    if cluster.saturation_count is not None:
        lines.append(f"EXISTING COVERAGE: ~{cluster.saturation_count} articles")

    lines.append("\nREPORTS:")
    for item in cluster.items[:8]:
        lines.append(f"\n- [{item.source}] {item.title}")
        if item.score is not None:
            lines.append(f"  engagement: {item.score} points, {item.comments or 0} comments")
        if item.summary:
            lines.append(f"  summary: {item.summary}")

    # The Guardian is the only free source returning full article bodies, which
    # is what makes a real "context" paragraph possible rather than a restated
    # headline. Use it when the cluster happens to include one.
    for item in cluster.items:
        body = (item.extra or {}).get("body_text")
        if body:
            lines.append(f"\nFULL ARTICLE TEXT ({item.source}):\n{body[:max_body_chars]}")
            break

    return "\n".join(lines)


class Summariser:
    def __init__(self, api_key: str, config: Config) -> None:
        settings = config.raw.get("summarise", {}) or {}
        self.key = api_key
        self.model = settings.get("model", "gemini-flash-lite-latest")
        self.temperature = float(settings.get("temperature", 0.2))
        self.max_output_tokens = int(settings.get("max_output_tokens", 2048))
        self.cache = JsonCache("summaries", ttl_seconds=int(settings.get("cache_ttl_s", 86400)))
        self.calls = 0
        self.errors: list[str] = []

    def _cache_key(self, material: str) -> str:
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return f"{self.model}:{digest}"

    async def summarise(
        self, client: httpx.AsyncClient, cluster: Cluster
    ) -> Summary | None:
        material = build_material(cluster)
        key = self._cache_key(material)

        cached = self.cache.get(key)
        if cached:
            return Summary(**cached, cached=True)

        url = f"{BASE}/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": PROMPT.format(material=material)}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }

        try:
            response = await client.post(url, params={"key": self.key}, json=payload)
            self.calls += 1

            if response.status_code == 429:
                self.errors.append("rate limited — check quota in AI Studio")
                return None
            if response.status_code != 200:
                self.errors.append(f"HTTP {response.status_code}: {response.text[:120]}")
                return None

            data = response.json()
            candidate = (data.get("candidates") or [{}])[0]

            # A truncated response yields invalid JSON. Surface it rather than
            # letting json.loads raise something opaque.
            if candidate.get("finishReason") == "MAX_TOKENS":
                self.errors.append(f"truncated: {cluster.title[:40]}")
                return None

            text = "".join(
                part.get("text", "")
                for part in (candidate.get("content") or {}).get("parts") or []
            )
            if not text.strip():
                self.errors.append(f"empty response: {cluster.title[:40]}")
                return None

            parsed = json.loads(text)
            self.cache.set(key, parsed)
            return Summary(**parsed)

        except (httpx.HTTPError, json.JSONDecodeError, TypeError) as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return None


async def summarise_top(
    client: httpx.AsyncClient,
    clusters: list[Cluster],
    config: Config,
    api_key: str,
) -> tuple[dict[int, Summary], Summariser]:
    """Summarise the top N clusters. Returns {cluster_index: Summary}."""
    settings = config.raw.get("summarise", {}) or {}
    limit = int(settings.get("max_stories", 3))

    summariser = Summariser(api_key, config)
    targets = clusters[:limit]

    # Small concurrency: free-tier per-minute limits are not published, and a
    # burst of parallel calls is the fastest way to discover them the hard way.
    semaphore = asyncio.Semaphore(2)

    async def one(index: int, cluster: Cluster):
        async with semaphore:
            return index, await summariser.summarise(client, cluster)

    results = await asyncio.gather(*(one(i, c) for i, c in enumerate(targets)))
    summariser.cache.save()

    return {i: s for i, s in results if s is not None}, summariser
