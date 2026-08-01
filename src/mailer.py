"""Render the briefing to HTML and send it.

Named `mailer` rather than `email` on purpose — a module called `email.py`
shadows the standard library package that smtplib itself imports, and the
resulting failure is deeply confusing.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import Config
from .pipeline import Briefing
from .score import saturation_label

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_SATURATION_COLOUR = {
    "LOW": "#245c44",
    "MEDIUM": "#9a6206",
    "HIGH": "#8f3527",
    "unmeasured": "#8794a1",
}


def _age_label(hours: float) -> str:
    if hours < 1:
        return f"{int(hours * 60)}m old"
    if hours < 24:
        return f"{hours:.0f}h old"
    return f"{hours / 24:.0f}d old"


def _topic_label(topic: str) -> str:
    return topic.replace("_", " ")


@dataclass
class Rendered:
    subject: str
    html: str
    story_count: int


def build_context(briefing: Briefing, config: Config) -> dict:
    settings = config.raw.get("summarise", {}) or {}
    min_confidence = settings.get("min_confidence", "medium")
    threshold = _CONFIDENCE_RANK.get(min_confidence, 1)

    stories = []
    used_indices = set()

    for index, summary in sorted(briefing.summaries.items()):
        if _CONFIDENCE_RANK.get(summary.confidence, 0) < threshold:
            # Too thin to summarise honestly. Dropping it here is what makes
            # the quiet-day message truthful rather than decorative.
            continue

        cluster = briefing.clusters[index]
        used_indices.add(index)
        label = saturation_label(cluster.saturation_count)

        links = [{"url": cluster.best_url, "label": "Source"}]
        for n, url in enumerate(cluster.discussion_urls[:2], 1):
            links.append({"url": url, "label": f"Discussion {n}" if n > 1 else "Discussion"})

        seen = [s for s in cluster.sources if s != cluster.lead.source]

        stories.append({
            "title": cluster.title,
            "topic_label": _topic_label(cluster.topic),
            "age_label": _age_label(cluster.age_hours),
            "saturation_label": label,
            "saturation_colour": _SATURATION_COLOUR.get(label, "#8794a1"),
            "what_happened": summary.what_happened,
            "context": summary.context,
            "why_it_spreads": summary.why_it_spreads,
            "links": links,
            "seen_in": ", ".join(seen[:5]) if seen else "",
        })

    watch_limit = int(settings.get("watchlist_size", 6))
    watchlist = []
    for index, cluster in enumerate(briefing.clusters):
        if index in used_indices or cluster.score <= 0:
            continue
        label = saturation_label(cluster.saturation_count)
        watchlist.append({
            "title": cluster.title,
            "url": cluster.best_url,
            "topic_label": _topic_label(cluster.topic),
            "age_label": _age_label(cluster.age_hours),
            "saturation_label": label,
            "saturation_colour": _SATURATION_COLOUR.get(label, "#8794a1"),
        })
        if len(watchlist) >= watch_limit:
            break

    sat = briefing.saturation
    if sat is None:
        saturation_note = ""
    elif sat.ok:
        saturation_note = f"{sat.measured} coverage checks"
    else:
        saturation_note = "coverage checks degraded"

    degraded = ", ".join(r.name for r in briefing.failed_sources)
    if sat is not None and not sat.ok:
        degraded = ", ".join(filter(None, [degraded, "saturation"]))

    return {
        "date_label": datetime.now(timezone.utc).strftime("%A %d %B %Y"),
        "stories": stories,
        "watchlist": watchlist,
        "total_items": len(briefing.items),
        "total_stories": len(briefing.clusters),
        "sources_ok": sum(1 for r in briefing.results if r.ok),
        "sources_total": len(briefing.results),
        "saturation_note": saturation_note,
        "degraded": degraded,
    }


def render(briefing: Briefing, config: Config) -> Rendered:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    context = build_context(briefing, config)
    html = env.get_template("briefing.html").render(**context)

    stories = context["stories"]
    if stories:
        lead = stories[0]["title"]
        subject = f"AI Briefing · {lead[:64]}{'…' if len(lead) > 64 else ''}"
    else:
        subject = "AI Briefing · quiet day, nothing worth covering"

    return Rendered(subject=subject, html=html, story_count=len(stories))


def send(rendered: Rendered, *, dry_run: bool = False) -> str:
    """Send via Gmail SMTP. Returns a human-readable outcome."""
    address = (os.getenv("GMAIL_ADDRESS") or "").strip()
    # Gmail displays app passwords in four-character groups; the spaces are
    # presentation only and must be stripped before authenticating.
    password = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "").strip()
    recipient = (os.getenv("BRIEFING_RECIPIENT") or "").strip() or address

    if not (address and password):
        raise RuntimeError("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set")

    message = EmailMessage()
    message["Subject"] = rendered.subject
    message["From"] = f"AI Trend Briefing <{address}>"
    message["To"] = recipient
    message.set_content(
        "This briefing is formatted as HTML. "
        "Your mail client is showing the plain-text fallback."
    )
    message.add_alternative(rendered.html, subtype="html")

    if dry_run:
        return f"dry run — would send '{rendered.subject}' to {recipient}"

    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.starttls(context=context)
        server.login(address, password)
        server.send_message(message)

    return f"sent to {recipient}"


def send_failure_alert(summary: str, detail: str) -> None:
    """Email a short alert when a run fails.

    A scheduled job that breaks in the cloud is invisible — you only notice
    the absence of a briefing, days later. This turns silent failure into a
    message. Best-effort: if even this cannot send, swallow it, since it runs
    inside an already-failing path.
    """
    address = (os.getenv("GMAIL_ADDRESS") or "").strip()
    password = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "").strip()
    recipient = (os.getenv("BRIEFING_RECIPIENT") or "").strip() or address
    if not (address and password):
        return

    message = EmailMessage()
    message["Subject"] = f"⚠️ AI Briefing failed — {summary}"[:120]
    message["From"] = f"AI Trend Briefing <{address}>"
    message["To"] = recipient
    message.set_content(
        f"The daily briefing run did not complete.\n\n{summary}\n\n{detail}\n\n"
        "Check the GitHub Actions logs for the full traceback."
    )

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls(context=ctx)
            server.login(address, password)
            server.send_message(message)
    except Exception as exc:  # noqa: BLE001 — already in a failure path
        log.error("could not send failure alert: %s", exc)
