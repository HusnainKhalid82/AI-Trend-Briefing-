#!/usr/bin/env python3
"""AI Trend Briefing — collect, cluster and rank AI news.

Usage:
    python run.py                    # ranked briefing
    python run.py --sources          # per-source collection report
    python run.py --top 15           # show more stories
    python run.py --explain          # show the score breakdown per story
    python run.py --no-saturation    # skip NewsData lookups (saves credits)
    python run.py --json out.json    # write ranked clusters to a file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import sys
from datetime import timezone

from dotenv import load_dotenv

# Explicit path: find_dotenv() inspects the call stack and fails in some
# execution contexts.
load_dotenv(".env", override=False)

from src.cluster import Cluster  # noqa: E402
from src.config import load_config  # noqa: E402
from src.pipeline import Briefing, build  # noqa: E402
from src.score import format_components, saturation_label  # noqa: E402

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, CYAN, MAGENTA = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[35m",
)

RULE = "─" * 78


def _c(text: str, colour: str) -> str:
    return text if not sys.stdout.isatty() else f"{colour}{text}{RESET}"


def _saturation_colour(label: str) -> str:
    return {"LOW": GREEN, "MEDIUM": YELLOW, "HIGH": RED}.get(label, DIM)


def print_sources(briefing: Briefing) -> None:
    print(f"\n{_c('SOURCES', BOLD)}")
    print(RULE)
    for result in sorted(briefing.results, key=lambda r: (-len(r.items), r.name)):
        status = _c("ok", GREEN) if result.ok else _c("FAILED", RED)
        pad = 10 + (len(status) - (2 if result.ok else 6))
        print(f"{result.name:<22}{status:<{pad}}{len(result.items):>7}{result.duration_s:>8.1f}s")
    print(RULE)
    ok = sum(1 for r in briefing.results if r.ok)
    print(f"{len(briefing.items)} unique items from {ok}/{len(briefing.results)} sources")
    for result in briefing.failed_sources:
        print(f"  {_c(result.name, RED)}: {result.error}")


def print_ranked(briefing: Briefing, top: int, explain: bool) -> None:
    print(f"\n{_c('RANKED STORIES', BOLD)}   "
          f"{_c('velocity × corroboration × topic × recency ÷ saturation', DIM)}")
    print(RULE)

    shown = [c for c in briefing.clusters if c.score > 0][:top]

    if not shown:
        print("  Nothing scored above zero — check that collectors returned data.")
        return

    for rank, cluster in enumerate(shown, 1):
        label = saturation_label(cluster.saturation_count)
        count = cluster.saturation_count
        satur = f"{label}" + (f" ({count})" if count is not None else "")

        print(f"\n{_c(f'{rank:>2}.', BOLD)} {_c(f'{cluster.score:6.2f}', BOLD)}  "
              f"{cluster.title[:66]}")
        print(f"     {_c(cluster.topic, MAGENTA):<28}"
              f"  {_c(f'{cluster.age_hours:.1f}h old', DIM)}"
              f"  {_c(f'{cluster.source_count} source(s)', CYAN)}"
              f"  saturation {_c(satur, _saturation_colour(label))}")
        print(f"     {_c(cluster.best_url[:70], DIM)}")

        if cluster.source_count > 1:
            print(f"     {_c('seen in: ' + ', '.join(cluster.sources[:6]), DIM)}")
        if explain:
            print(f"     {_c(format_components(cluster), DIM)}")
            if q := cluster.components.get("saturation_query"):
                print(f"     {_c(f'saturation query: {q!r}', DIM)}")


def print_briefing(briefing: Briefing, config) -> None:
    """The fully worked stories — what the email will carry."""
    limit = int((config.raw.get("summarise") or {}).get("max_stories", 3))
    min_conf = (config.raw.get("summarise") or {}).get("min_confidence", "medium")
    rank = {"low": 0, "medium": 1, "high": 2}

    usable = [
        (i, briefing.clusters[i], s)
        for i, s in sorted(briefing.summaries.items())
        if rank.get(s.confidence, 0) >= rank.get(min_conf, 1)
    ]

    print(f"\n{_c('TODAY\'S BRIEFING', BOLD)}")
    print(RULE)

    if not usable:
        print(f"  {_c('No story cleared the confidence threshold today.', YELLOW)}")
        print("  The material was too thin to summarise honestly.")
        print("  Better to report a quiet day than manufacture significance.\n")
        return

    for n, (idx, cluster, s) in enumerate(usable[:limit], 1):
        label = saturation_label(cluster.saturation_count)
        print(f"\n{_c(f'{n}.', BOLD)} {_c(cluster.title[:68], BOLD)}")
        print(f"   {_c(cluster.topic, MAGENTA)}  {_c(f'{cluster.age_hours:.1f}h old', DIM)}"
              f"  saturation {_c(label, _saturation_colour(label))}"
              f"  {_c('confidence: ' + s.confidence, DIM)}"
              f"{_c('  [cached]', DIM) if s.cached else ''}")
        print(f"\n   {_c('WHAT HAPPENED', BOLD)}")
        print(f"   {s.what_happened}")
        print(f"\n   {_c('CONTEXT', BOLD)}")
        print(f"   {s.context}")
        print(f"\n   {_c('WHY IT CAN SPREAD', BOLD)}")
        print(f"   {s.why_it_spreads}")
        print(f"\n   {_c('SOURCES', BOLD)}")
        print(f"   {cluster.best_url}")
        for u in cluster.discussion_urls[:2]:
            print(f"   {u}")
        print()


def print_watchlist(briefing: Briefing, top: int, skip: int) -> None:
    """Stories with traction but not yet enough to lead the briefing."""
    tail = [c for c in briefing.clusters if c.score > 0][skip:skip + top]
    if not tail:
        return
    print(f"\n\n{_c('WATCHLIST', BOLD)}   {_c('gaining traction, not yet major', DIM)}")
    print(RULE)
    for cluster in tail:
        label = saturation_label(cluster.saturation_count)
        print(f"  {_c(f'{cluster.score:5.2f}', DIM)}  {cluster.title[:60]}")
        print(f"         {_c(cluster.topic, DIM)}  {_c(f'{cluster.age_hours:.0f}h', DIM)}"
              f"  {_c(label, _saturation_colour(label))}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Collect, cluster and rank AI news")
    parser.add_argument("--top", type=int, default=10, help="stories to show")
    parser.add_argument("--sources", action="store_true", help="show per-source report")
    parser.add_argument("--explain", action="store_true", help="show score breakdown")
    parser.add_argument("--no-saturation", action="store_true", help="skip NewsData lookups")
    parser.add_argument("--json", metavar="PATH", help="write ranked clusters to JSON")
    parser.add_argument("--brief", action="store_true",
                        help="summarise top stories with Gemini")
    parser.add_argument("--send", action="store_true",
                        help="render and email the briefing (implies --brief)")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --send, render and save locally without sending")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--alert-on-failure", action="store_true",
                        help="email an alert if the run crashes (for scheduled runs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="  %(levelname)s %(name)s: %(message)s",
    )

    try:
        return await run(args)
    except Exception as exc:
        # A scheduled run that crashes is invisible until you notice the missing
        # briefing. Turn it into an email, then re-raise so the workflow still
        # shows a red X in the run history.
        if args.alert_on_failure:
            import traceback
            from src.mailer import send_failure_alert
            send_failure_alert(f"{type(exc).__name__}: {exc}"[:100],
                               traceback.format_exc()[-1500:])
        raise


async def run(args) -> int:
    config = load_config()

    # Deduplication memory is only meaningful when we are actually sending; a
    # local inspection run should see every story, seen before or not.
    from src.history import History
    history = History.load() if args.send and not args.dry_run else None

    briefing = await build(
        config,
        measure_saturation=not args.no_saturation,
        summarise=args.brief or args.send,
        gemini_key=(os.getenv("GEMINI_API_KEY") or "").strip() or None,
        history=history,
    )

    if args.sources:
        print_sources(briefing)

    if args.brief or args.send:
        print_briefing(briefing, config)
    print_ranked(briefing, args.top, args.explain)
    print_watchlist(briefing, top=8, skip=args.top)

    scored = sum(1 for c in briefing.clusters if c.score > 0)
    print(f"\n{RULE}")
    print(f"{len(briefing.items)} items → {len(briefing.clusters)} stories ({scored} scored)")

    sat = briefing.saturation
    if sat is not None:
        detail = (f"{sat.measured} measured ({sat.from_cache} cached, "
                  f"{sat.requests} requests)")
        if sat.ok:
            print(f"saturation: {_c(detail, GREEN)}")
        else:
            print(f"saturation: {_c('DEGRADED', RED)} — {detail}")
            print(f"            {_c(str(sat.error), YELLOW)}")
            print(f"            {_c('rankings below are unreliable until this clears', YELLOW)}")
    if briefing.enrichment is not None:
        e = briefing.enrichment
        print(f"articles:  {e.fetched} fetched ({e.from_cache} cached), "
              f"{e.rejected} rejected of {e.attempted + e.rejected} attempted")
        for r in e.reasons[:3]:
            print(f"           {_c(r, DIM)}")
    if briefing.summariser is not None:
        sm = briefing.summariser
        cached = sum(1 for s in briefing.summaries.values() if s.cached)
        print(f"summaries: {len(briefing.summaries)} ({cached} cached, {sm.calls} API calls)")
        for e in sm.errors[:3]:
            print(f"           {_c(e, YELLOW)}")
    if briefing.failed_sources:
        names = ", ".join(r.name for r in briefing.failed_sources)
        print(f"{_c('degraded:', YELLOW)} {names}")
    print()

    if args.send:
        from src.mailer import render, send
        rendered = render(briefing, config)
        preview = pathlib.Path("briefing-preview.html")
        preview.write_text(rendered.html, encoding="utf-8")
        print(f"rendered {rendered.story_count} stories -> {preview}")
        try:
            print(f"{_c(send(rendered, dry_run=args.dry_run), GREEN)}")
        except Exception as exc:
            print(f"{_c('SEND FAILED', RED)}: {type(exc).__name__}: {exc}")
            return 1

        # Record only what actually went out, and only on a real send. Stories
        # that led the email are marked seen so tomorrow does not repeat them;
        # the watchlist is left unrecorded so it can still be promoted later.
        if history is not None and not args.dry_run:
            limit = int((config.raw.get("summarise") or {}).get("max_stories", 3))
            min_conf = (config.raw.get("summarise") or {}).get("min_confidence", "medium")
            rank = {"low": 0, "medium": 1, "high": 2}
            sent = [
                briefing.clusters[i]
                for i, s in sorted(briefing.summaries.items())
                if rank.get(s.confidence, 0) >= rank.get(min_conf, 1)
            ][:limit]
            history.record(sent)
            pruned = history.prune()
            history.save()
            print(f"recorded {len(sent)} stories to history "
                  f"({len(history.seen)} tracked, {pruned} pruned)")
        print()

    if args.json:
        payload = [
            {
                "rank": i,
                "score": round(c.score, 3),
                "title": c.title,
                "url": c.best_url,
                "topic": c.topic,
                "age_hours": round(c.age_hours, 2),
                "sources": c.sources,
                "saturation_count": c.saturation_count,
                "saturation_label": saturation_label(c.saturation_count),
                "discussion_urls": c.discussion_urls,
                "components": c.components,
                "published": c.published.astimezone(timezone.utc).isoformat(),
                "items": [
                    {"source": it.source, "title": it.title, "url": it.url,
                     "score": it.score, "comments": it.comments}
                    for it in c.items
                ],
            }
            for i, c in enumerate(briefing.clusters[: args.top * 3], 1)
        ]
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {len(payload)} stories to {args.json}\n")

    return 0 if any(r.ok for r in briefing.results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
