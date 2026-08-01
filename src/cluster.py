"""Group items that describe the same event.

Two items belong together when they point at the same URL, or when their
headlines are similar enough after normalisation. Deliberately not embeddings:
fuzzy title matching is fast, has no dependencies worth the name, and its
mistakes are legible — you can read the cluster and see why it merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from rapidfuzz import fuzz

from .config import Config
from .entities import extract as extract_entities
from .models import Item

# Words that carry no distinguishing information in a tech headline. Removing
# them stops "OpenAI announces the new model" and "Google announces a new
# model" from looking similar purely because of their scaffolding.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "with", "at", "by", "from", "as", "its", "it", "that", "this", "new", "now",
    "has", "have", "how", "why", "what", "you", "your", "we", "our", "be",
}

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalise_title(title: str, strip_prefixes: list[str]) -> str:
    text = title.lower().strip()
    for prefix in strip_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = _PUNCT.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens) if tokens else text


@dataclass
class Cluster:
    """One event, as reported by one or more sources."""

    items: list[Item] = field(default_factory=list)
    key: str = ""

    # Populated by score.py
    topic: str = "uncategorised"
    topic_weight: float = 1.0
    saturation_count: int | None = None
    score: float = 0.0
    components: dict = field(default_factory=dict)

    @property
    def lead(self) -> Item:
        """The item that best represents the cluster.

        Prefer whichever copy has engagement data, then the most recent — an
        HN thread with 300 points describes the event better than a syndicated
        wire-service rewrite of it.
        """
        return max(
            self.items,
            key=lambda i: (i.score is not None, i.score or 0, i.published),
        )

    @property
    def title(self) -> str:
        return self.lead.title

    @property
    def sources(self) -> list[str]:
        # Collapse "rss:techmeme" and "rss:openai" to distinct entries but treat
        # repeated hits from one collector as one source.
        return sorted({i.source for i in self.items})

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def published(self) -> datetime:
        """Earliest sighting — when the story broke, not when it was echoed."""
        return min(i.published for i in self.items)

    @property
    def age_hours(self) -> float:
        return min(i.age_hours for i in self.items)

    @property
    def max_velocity(self) -> float:
        return max((i.velocity or 0.0) for i in self.items)

    @property
    def total_engagement(self) -> int:
        return sum((i.score or 0) for i in self.items)

    @property
    def has_official_source(self) -> bool:
        """True when a company blog or model registry carries this.

        These rarely generate forum engagement immediately but are the most
        reliable indicator that something actually shipped.
        """
        return any(
            i.source.startswith(("rss:openai", "rss:deepmind", "rss:google-ai",
                                 "rss:mistral", "rss:huggingface", "openrouter",
                                 "huggingface:"))
            for i in self.items
        )

    @property
    def best_url(self) -> str:
        return self.lead.url

    @property
    def discussion_urls(self) -> list[str]:
        return sorted({i.discussion_url for i in self.items if i.discussion_url})


def build_clusters(items: list[Item], config: Config) -> list[Cluster]:
    settings = config.raw.get("clustering", {}) or {}
    threshold = int(settings.get("title_similarity", 82))
    prefixes = [p.lower() for p in settings.get("strip_prefixes", [])]

    # Newest first, so the earliest-published item does not become the anchor
    # for everything that follows it.
    min_shared = int(settings.get("min_shared_entities", 2))
    overlap_ratio = float(settings.get("entity_overlap_ratio", 0.6))

    ordered = sorted(items, key=lambda i: i.published, reverse=True)

    clusters: list[Cluster] = []
    normalised: list[str] = []
    entity_sets: list[set[str]] = []
    by_url: dict[str, int] = {}

    for item in ordered:
        norm = normalise_title(item.title, prefixes)
        ents = extract_entities(item.title)

        # Same destination URL is a certainty, not a similarity judgement.
        index = by_url.get(item.url) if item.url else None

        if index is None:
            best_index, best_score = None, 0.0
            for i, existing in enumerate(normalised):
                # token_set_ratio ignores word order and tolerates one headline
                # carrying extra words, which is exactly how outlets rewrite.
                similarity = fuzz.token_set_ratio(norm, existing)
                if similarity > best_score:
                    best_index, best_score = i, similarity
            if best_score >= threshold:
                index = best_index

        # Fuzzy matching fails on genuine rewrites: "Moonshot Launches Kimi K3"
        # and "Chinese AI model takes US industry by surprise" share no words at
        # all. Shared named entities catch some of those.
        #
        # Two constraints keep this from running away. Entity sets are NOT
        # accumulated as items join — comparing against a growing union lets one
        # cluster widen its net until it swallows everything (measured: 22
        # unrelated models merged with a security breach). And overlap must be a
        # majority of the smaller set, so sharing two incidental names is not
        # enough on its own.
        if index is None and len(ents) >= min_shared:
            for i, existing in enumerate(entity_sets):
                if len(existing) < min_shared:
                    continue
                shared = ents & existing
                if len(shared) < min_shared:
                    continue
                if len(shared) / min(len(ents), len(existing)) >= overlap_ratio:
                    index = i
                    break

        if index is None:
            clusters.append(Cluster(items=[item], key=norm))
            normalised.append(norm)
            entity_sets.append(set(ents))
            index = len(clusters) - 1
        else:
            clusters[index].items.append(item)

        if item.url:
            by_url.setdefault(item.url, index)

    return clusters
