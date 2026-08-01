# AI Trend Briefing

An automated daily email that finds AI news which is **climbing but not yet
saturated**, and delivers it with the context needed to act on it the same day.

Built to answer one question every morning — *what AI story is worth covering
today, and why* — without an hour of manual source-checking. Runs on a schedule,
costs nothing, needs no server.

---

## What it does

Every morning a scheduled job:

1. **Collects** ~250 items from 8 public sources in parallel
2. **Clusters** items describing the same event into single stories
3. **Ranks** them by a formula that rewards *early* stories over *popular* ones
4. **Summarises** the top stories with an LLM, constrained to fact only
5. **Emails** a briefing — what happened, the context, why it may spread
6. **Remembers** what it sent, so no story is ever repeated

Start to finish in about 30 seconds. Running cost: **$0/month**.

## The core idea

Most aggregators rank by popularity, which surfaces news that is already
everywhere — useless if you want to be first. This ranks by:

```
score = velocity × corroboration × topic_weight × recency ÷ saturation
```

Dividing by **saturation** — how many outlets already cover it — is what
promotes a story on the way up over one that has already peaked.

## Sources

| Signal (early)                          | Record (official)              |
| --------------------------------------- | ------------------------------ |
| Hacker News, Hugging Face, OpenRouter,  | Company RSS (OpenAI, DeepMind, |
| Lobste.rs, arXiv                        | Google, Mistral…), Guardian,   |
|                                         | NewsData.io                    |

## Architecture

```
8 sources ──> collect ──> cluster ──> rank ──> [saturation] ──> summarise ──> email
   (async)     (dedup)   (entity +    (score      (NewsData)      (Gemini)     (SMTP)
                          fuzzy)       formula)
                                          │
                                     history.json  ◄── committed back each run
                                     (dedup memory, the repo is the database)
```

Every stage isolates failure: one dead source degrades the briefing rather than
breaking the run.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys
python run.py                 # ranked report to the terminal
python run.py --brief         # + LLM summaries of the top stories
python run.py --send          # + email it
python run.py --explain       # show the score breakdown per story
```

## Automated delivery

A GitHub Actions workflow (`.github/workflows/briefing.yml`) runs it daily,
reads keys from repository secrets, and commits the updated `history.json` back
to the repo. See the workflow file for the schedule and the required secrets.

## Configuration

Everything tunable lives in `config.yaml` — source toggles, topic weights,
clustering thresholds, and scoring parameters — so behaviour changes without
touching code.

## Stack

Python · httpx · feedparser · rapidfuzz · trafilatura · Jinja2 · Gemini
(free tier) · Gmail SMTP · GitHub Actions. No database, no server, no paid tier.
