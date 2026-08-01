"""The ranking formula.

    score = signal × corroboration × topic_weight × recency ÷ saturation

Every term is exposed in `cluster.components` so any ordering can be explained
rather than guessed at. That matters more than it sounds: this is the file that
gets tuned, and untraceable scores cannot be tuned.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .cluster import Cluster
from .config import Config


@lru_cache(maxsize=64)
def _topic_pattern(keywords: tuple[str, ...]) -> re.Pattern:
    """Word-boundary matcher for one topic's keywords.

    Substring matching is wrong here and quietly so: "api" fires on "rapid"
    and "capital", "ban" fires on "urban" and "banking". That silently
    mislabels stories and applies the wrong weight, which is worse than
    leaving them uncategorised because it looks like it worked.
    """
    return re.compile(
        r"(?<![a-z0-9])(?:"
        + "|".join(re.escape(k.lower()) for k in keywords)
        + r")(?![a-z0-9])",
        re.IGNORECASE,
    )


def classify(cluster: Cluster, config: Config) -> tuple[str, float]:
    """Assign a topic by keyword match against title and summary.

    Sprint 2 is deliberately keyword-based: it is inspectable and costs nothing.
    Topics are checked in config order, so put the ones you care about first.
    """
    topics = config.raw.get("topics", {}) or {}

    # Weight the title over body text — a story is about what its headline
    # says, and summaries mention all sorts of incidental things.
    title = cluster.title.lower()
    body = " ".join(i.summary or "" for i in cluster.items).lower()

    for haystack in (title, f"{title} {body}"):
        for name, spec in topics.items():
            keywords = tuple(spec.get("keywords", []))
            if not keywords:
                continue
            if _topic_pattern(keywords).search(haystack):
                return name, float(spec.get("weight", 1.0))

    # Unmatched stories are not penalised into invisibility — a genuinely novel
    # story may use none of these words.
    return "uncategorised", 0.6


def signal_strength(cluster: Cluster, floor: float, news_floor: float) -> float:
    """How much attention this is actually getting, per hour.

    Engagement velocity where it exists. Where it does not, we fall back to
    what the source implies:

    - An official announcement with no forum traction yet is among the most
      valuable things this system can find. Scoring it zero would bury it.
    - A reported news story (Techmeme, NewsData, Guardian) has passed an
      editor, which is weaker evidence than an official post but far stronger
      than nothing. A $450M funding round should not score 0.5 purely because
      the outlet exposes no engagement metric.
    """
    velocity = cluster.max_velocity
    if velocity > 0:
        return velocity
    if cluster.has_official_source:
        return floor
    if any(i.kind == "record" for i in cluster.items):
        return news_floor
    return 0.5


def score_cluster(cluster: Cluster, config: Config) -> float:
    settings = config.raw.get("scoring", {}) or {}
    half_life = float(settings.get("recency_half_life_h", 24))
    corroboration_exp = float(settings.get("corroboration_exponent", 0.6))
    floor = float(settings.get("official_announcement_floor", 8.0))
    news_floor = float(settings.get("news_report_floor", 2.0))
    saturation_exp = float(settings.get("saturation_exponent", 0.45))

    topic, weight = classify(cluster, config)
    cluster.topic, cluster.topic_weight = topic, weight

    signal = signal_strength(cluster, floor, news_floor)

    # Independent sources carrying the same story is the strongest evidence
    # that it is real and spreading. Sub-linear so a wire-service echo across
    # ten outlets does not dominate a genuine scoop on two.
    corroboration = cluster.source_count ** corroboration_exp

    # Exponential decay: at one half-life a story is worth half as much.
    recency = 0.5 ** (cluster.age_hours / half_life)

    # Unmeasured saturation is treated as average rather than zero. Treating it
    # as zero would send every unmeasured cluster straight to the top.
    if cluster.saturation_count is None:
        saturation = 12.0 ** saturation_exp
        measured = False
    else:
        saturation = (1.0 + cluster.saturation_count) ** saturation_exp
        measured = True

    score = (signal * corroboration * weight * recency) / saturation

    cluster.components = {
        **cluster.components,
        "signal": round(signal, 2),
        "corroboration": round(corroboration, 2),
        "source_count": cluster.source_count,
        "topic": topic,
        "topic_weight": weight,
        "recency": round(recency, 3),
        "age_hours": round(cluster.age_hours, 1),
        "saturation": round(saturation, 2),
        "saturation_count": cluster.saturation_count,
        "saturation_measured": measured,
    }
    cluster.score = score
    return score


def preliminary_rank(clusters: list[Cluster], config: Config) -> list[Cluster]:
    """Rank before saturation is known, to decide which lookups to spend on.

    Saturation costs an API credit per cluster, so it is only worth measuring
    for stories that could plausibly reach the briefing.
    """
    for cluster in clusters:
        score_cluster(cluster, config)
    return sorted(clusters, key=lambda c: c.score, reverse=True)


def final_rank(clusters: list[Cluster], config: Config) -> list[Cluster]:
    """Re-score once saturation counts are attached."""
    for cluster in clusters:
        score_cluster(cluster, config)
    return sorted(clusters, key=lambda c: c.score, reverse=True)


def saturation_label(count: int | None) -> str:
    if count is None:
        return "unmeasured"
    if count <= 5:
        return "LOW"
    if count <= 40:
        return "MEDIUM"
    return "HIGH"


def format_components(cluster: Cluster) -> str:
    c = cluster.components
    return (
        f"signal {c.get('signal', 0):>6.1f} × corrob {c.get('corroboration', 0):>4.2f}"
        f" × topic {c.get('topic_weight', 0):.1f} × recency {c.get('recency', 0):.3f}"
        f" ÷ satur {c.get('saturation', 0):>5.2f}"
    )
