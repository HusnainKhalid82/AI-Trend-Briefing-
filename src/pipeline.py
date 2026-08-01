"""End-to-end: collect, cluster, measure saturation, rank."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .cluster import Cluster, build_clusters
from .collect import collect_all
from .config import Config
from .enrich import EnrichReport, enrich
from .history import History, filter_seen
from .http import make_client
from .models import CollectorResult, Item
from .saturation import SaturationReport, measure
from .score import final_rank, preliminary_rank
from .summarise import Summariser, Summary, summarise_top

log = logging.getLogger(__name__)


@dataclass
class Briefing:
    clusters: list[Cluster] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    results: list[CollectorResult] = field(default_factory=list)
    saturation: SaturationReport | None = None
    summaries: dict[int, Summary] = field(default_factory=dict)
    summariser: Summariser | None = None
    enrichment: EnrichReport | None = None
    dropped_seen: int = 0

    @property
    def failed_sources(self) -> list[CollectorResult]:
        return [r for r in self.results if not r.ok]


async def build(
    config: Config,
    measure_saturation: bool = True,
    summarise: bool = False,
    gemini_key: str | None = None,
    history: History | None = None,
) -> Briefing:
    items, results = await collect_all(config)

    clusters = build_clusters(items, config)
    log.info("clustered %d items into %d stories", len(items), len(clusters))

    # Rank first, then spend credits only on clusters that could realistically
    # make the briefing.
    ranked = preliminary_rank(clusters, config)

    # Drop stories already covered on a previous day — before saturation and
    # summarisation, so no API budget is spent on anything we cannot use.
    dropped = 0
    if history is not None:
        ranked, dropped = filter_seen(ranked, history)

    report = None
    if measure_saturation:
        async with make_client() as client:
            report = await measure(client, ranked, config)

    final = final_rank(ranked, config)

    summaries, summariser, enrichment = {}, None, None
    if summarise and gemini_key:
        async with make_client() as client:
            # Fetch article text first — the summariser reads whatever is in
            # extra["body_text"], so enrichment must land before it runs.
            enrichment = await enrich(client, final, config)
            summaries, summariser = await summarise_top(client, final, config, gemini_key)

    return Briefing(
        clusters=final,
        items=items,
        results=results,
        saturation=report,
        summaries=summaries,
        summariser=summariser,
        enrichment=enrichment,
        dropped_seen=dropped,
    )
