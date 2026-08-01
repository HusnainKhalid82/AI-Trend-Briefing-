"""Extract the named entities that identify a story.

Shared by clustering (do two headlines concern the same thing?) and saturation
(how do I search for this story?). Lives in its own module because both need it
and neither should import the other.

Deliberately not an NLP model. Word shape carries most of the signal in tech
headlines — "OpenAI", "CuspAI", "DeepMind" have internal capitals; "GPT-5.6",
"K3", "450M" carry digits — and both survive Title Case, which defeats naive
capitalisation checks.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s.-]")

# Too generic to identify anything. Querying or clustering on these matches the
# entire field rather than one story.
GENERIC = {
    "ai", "new", "model", "models", "llm", "llms", "the", "a", "an", "and",
    "or", "to", "of", "in", "on", "for", "with", "is", "are", "how", "why",
    "what", "says", "could", "will", "can", "from", "at", "by", "its", "it",
    "this", "that", "artificial", "intelligence", "open", "source", "release",
    "released", "chatbot", "startup", "company",
    # Platform names appear in titles this project constructs itself — the
    # Hugging Face collector emits "New model trending on Hugging Face: X" —
    # so treating them as entities merges every model on a platform with any
    # story that merely mentions the platform. Measured: a security breach
    # merged with 12 unrelated model listings.
    "hugging", "face", "huggingface", "openrouter", "arxiv", "github",
    "lobsters", "reddit", "twitter",
}

# Ordinary English that appears capitalised in Title Case headlines. Without
# this, "OpenAI Stages Investor Comeback" treats "Stages" as an entity.
COMMON = {
    "stages", "investor", "investors", "comeback", "surge", "fuel", "makes",
    "made", "make", "takes", "take", "gets", "get", "sets", "set", "puts",
    "adds", "adding", "brings", "gives", "goes", "comes", "looks", "shows",
    "reveals", "reports", "claims", "plans", "aims", "seeks", "faces", "hits",
    "wins", "loses", "cuts", "raises", "raised", "drops", "rises", "falls",
    "launches", "launch", "launched", "announces", "announced", "unveils",
    "advice", "people", "users", "companies", "startups", "firm", "firms",
    "group", "team", "teams", "week", "year", "years", "day", "days", "month",
    "months", "time", "times", "world", "global", "market", "markets",
    "industry", "business", "tech", "technology", "data", "study", "research",
    "report", "news", "update", "updates", "first", "next", "last", "best",
    "top", "big", "small", "more", "most", "less", "than", "after", "before",
    "about", "into", "over", "under", "between", "through", "during",
    "against", "without", "billion", "million", "trillion", "percent", "here",
    "there", "when", "where", "who", "which", "while", "still", "just", "even",
    "also", "now", "then", "back", "down", "out", "off", "but", "not", "was",
    "were", "been", "has", "had", "have", "may", "might", "would", "should",
    "their", "they", "them", "his", "her", "our", "your", "you", "we",
    "some", "many", "much", "every", "all", "any", "both", "each", "other",
    "another", "such", "own", "same", "because", "since", "surprise",
    "ability", "shares", "industry", "landing", "demand", "parameter",
}


def distinctiveness(token: str) -> int:
    """How well a token identifies a specific story. Zero means useless."""
    core = token.strip(".,:;!?\"'()[]—–-")
    if len(core) < 3:
        return 0

    low = core.lower()
    if low in GENERIC or low in COMMON:
        return 0

    score = 0
    if any(ch.isupper() for ch in core[1:]):
        score += 10          # OpenAI, DeepMind, CuspAI, GPT
    if any(ch.isdigit() for ch in core):
        score += 6           # K3, GPT-5.6, 450M
    if core[0].isupper():
        score += 3           # weak alone, useful as a tiebreak
    if len(core) > 7:
        score += 1
    return score


def extract(text: str, min_score: int = 3) -> set[str]:
    """Entity tokens in `text`, lowercased.

    Trailing punctuation is stripped but internal hyphens and dots are kept so
    "GPT-5.6" survives as a single token.
    """
    tokens = _PUNCT.sub(" ", text).split()
    found = set()
    for token in tokens:
        if distinctiveness(token) >= min_score:
            found.add(token.strip(".,:;!?\"'()[]—–-").lower())
    return found


def top_terms(text: str, limit: int = 2) -> list[str]:
    """The `limit` most distinctive tokens, original casing, order preserved."""
    tokens = [t for t in _PUNCT.sub(" ", text).split() if t]
    ranked = sorted(
        ((t, distinctiveness(t)) for t in tokens),
        key=lambda pair: pair[1],
        reverse=True,
    )

    out, seen = [], set()
    for token, score in ranked:
        if score <= 0:
            break
        core = token.strip(".,:;!?\"'()[]—–-")
        low = core.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(core)
        if len(out) == limit:
            break

    if not out:
        fallback = sorted(
            (t for t in tokens if len(t) > 4 and t.lower() not in GENERIC),
            key=len,
            reverse=True,
        )
        out = [t.strip(".,:;!?\"'()[]—–-") for t in fallback[:limit]]

    return out
