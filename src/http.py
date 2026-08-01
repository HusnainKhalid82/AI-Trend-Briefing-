"""Shared HTTP behaviour: one client, sane timeouts, polite retries."""

from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

USER_AGENT = (
    "ai-trend-briefing/0.1 (personal news aggregator; "
    "contact via repository owner)"
)

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)

# Retried because they are usually transient. 4xx other than 429 is not retried:
# a 403 from Reddit will still be a 403 in two seconds.
_RETRY_STATUS = {429, 500, 502, 503, 504}


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


async def get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    attempts: int = 3,
) -> httpx.Response:
    """GET with bounded exponential backoff.

    GET specifically, never POST — arXiv only serves its Fastly cache to GET,
    and several sources rate-limit POST far more aggressively.
    """
    last: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code in _RETRY_STATUS and attempt < attempts:
                wait = _backoff(response, attempt)
                log.debug("%s -> %s, retrying in %.1fs", url, response.status_code, wait)
                await asyncio.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            if attempt < attempts and _worth_retrying(exc):
                await asyncio.sleep(2.0 ** (attempt - 1))
                continue
            raise

    raise last if last else RuntimeError(f"GET failed with no exception: {url}")


def _worth_retrying(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return False


def _backoff(response: httpx.Response, attempt: int) -> float:
    """Honour Retry-After when the server sends it, otherwise back off."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(2.0 ** (attempt - 1), 8.0)
