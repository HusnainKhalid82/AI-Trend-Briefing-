"""Deduplication memory — the repo *is* the database.

After each briefing is sent, the keys of the stories it carried are written to
history.json, and the GitHub Actions workflow commits that file back to the
repository. The next run loads it and drops anything already covered, so no
story is ever emailed twice.

For a once-daily job writing one small file, this beats a hosted database on
every axis: zero cost, zero credentials, zero network dependency, and a full
git history of every briefing ever sent as a free by-product.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .cluster import Cluster
from .entities import extract as extract_entities

log = logging.getLogger(__name__)

HISTORY_PATH = Path(__file__).resolve().parent.parent / "history.json"


def story_key(cluster: Cluster) -> str:
    """A stable identity for a story, robust to headline rewrites.

    A URL alone is too strict — the same event appears under many URLs. So the
    key is the two strongest entities; matching marks the story as seen, which
    is what stops a story returning tomorrow under a different outlet's rewrite.

    Two entities are required, not one. A single generic token like "ten" or
    "mit" would collide with unrelated future stories and wrongly suppress
    them. When a headline yields fewer than two entities, fall back to the URL,
    accepting that a rewrite under a new URL may slip through — a duplicate is a
    far smaller harm than silently dropping a real story.
    """
    entities = sorted(extract_entities(cluster.title))
    if len(entities) >= 2:
        return "ent:" + "+".join(entities[:2])
    return "url:" + cluster.best_url


@dataclass
class History:
    """Seen stories, keyed by story_key, valued by ISO timestamp first seen."""

    seen: dict[str, str]
    path: Path

    @classmethod
    def load(cls, path: Path = HISTORY_PATH) -> "History":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            seen = data.get("seen", {})
        except (FileNotFoundError, json.JSONDecodeError):
            seen = {}
        return cls(seen=seen, path=path)

    def is_seen(self, cluster: Cluster) -> bool:
        # A story is also "seen" if it shares a lead URL with anything recorded.
        if story_key(cluster) in self.seen:
            return True
        url_key = "url:" + cluster.best_url
        return url_key in self.seen

    def record(self, clusters: list[Cluster]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for cluster in clusters:
            self.seen.setdefault(story_key(cluster), now)
            # Also stamp the URL form, so a later rewrite is caught even if the
            # entities shift.
            self.seen.setdefault("url:" + cluster.best_url, now)

    def prune(self, keep_days: int = 14) -> int:
        """Forget stories older than keep_days so the file cannot grow forever.

        Two weeks is well beyond any story's shelf life, so pruning never
        resurrects something genuinely stale, and it keeps the committed file
        small enough that the daily diff stays reviewable.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        before = len(self.seen)
        kept = {}
        for key, stamp in self.seen.items():
            try:
                when = datetime.fromisoformat(stamp)
            except ValueError:
                when = datetime.now(timezone.utc)
            if when >= cutoff:
                kept[key] = stamp
        self.seen = kept
        return before - len(kept)

    def save(self) -> None:
        payload = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "count": len(self.seen),
            "seen": dict(sorted(self.seen.items())),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)


def filter_seen(clusters: list[Cluster], history: History) -> tuple[list[Cluster], int]:
    """Drop clusters already covered. Returns (fresh, dropped_count)."""
    fresh = [c for c in clusters if not history.is_seen(c)]
    return fresh, len(clusters) - len(fresh)
