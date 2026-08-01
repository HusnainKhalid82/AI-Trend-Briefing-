"""The common record every collector normalises into."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# Query parameters that identify a campaign rather than a document. Stripping
# them means the same article arriving from two feeds resolves to one URL.
_TRACKING_PREFIXES = ("utm_", "ga_", "_hs", "mc_")
_TRACKING_EXACT = {
    "fbclid", "gclid", "msclkid", "igshid", "ref", "referrer",
    "source", "cmpid", "spm", "at_medium", "at_campaign",
}


def canonical_url(url: str) -> str:
    """Strip tracking parameters and fragments so equal articles compare equal."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()

    keep = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower() in _TRACKING_EXACT or k.lower().startswith(_TRACKING_PREFIXES))
    ]

    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parts.path.rstrip("/") or "/"

    return urlunsplit((parts.scheme.lower(), netloc, path, urlencode(keep), ""))


@dataclass
class Item:
    """One thing that happened, from one source.

    `kind` is "signal" (something is gaining attention) or "record" (something
    officially occurred). The scorer in Sprint 2 treats the two differently.
    """

    source: str
    kind: str
    title: str
    url: str
    published: datetime
    discussion_url: str | None = None
    score: int | None = None
    comments: int | None = None
    author: str | None = None
    summary: str | None = None
    raw_id: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.title = " ".join((self.title or "").split())
        self.url = canonical_url(self.url)
        if self.published.tzinfo is None:
            self.published = self.published.replace(tzinfo=timezone.utc)
        else:
            self.published = self.published.astimezone(timezone.utc)

    @property
    def age_hours(self) -> float:
        delta = datetime.now(timezone.utc) - self.published
        return max(delta.total_seconds() / 3600.0, 0.0)

    @property
    def velocity(self) -> float | None:
        """Engagement gained per hour. None when the source exposes no score."""
        if self.score is None:
            return None
        # Floor the divisor so a 10-minute-old post does not report absurd velocity.
        return self.score / max(self.age_hours, 1.0)

    @property
    def dedup_key(self) -> str:
        return self.url or f"{self.source}:{self.raw_id or self.title.lower()}"


@dataclass
class CollectorResult:
    """Outcome of one collector. A failure here must never stop the run."""

    name: str
    items: list[Item] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None
