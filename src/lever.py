"""Collector for public Lever job sites.

Uses only the Python standard library. The public Postings API
(https://github.com/lever/postings-api) requires no authentication:

    GET https://api.lever.co/v0/postings/{site}?mode=json
    -> [{"id", "text", "categories": {"location", "team", "department",
         "commitment", "allLocations"}, "description", "descriptionPlain",
         "lists": [{"text", "content"}], "additional", "additionalPlain",
         "hostedUrl", "applyUrl", "createdAt", "workplaceType", ...}, ...]

Unlike Greenhouse and Ashby the response is a top-level JSON **list**, and an
unknown site returns HTTP 404. ``createdAt`` is an epoch timestamp in
milliseconds. The public API exposes no update timestamp, so ``updated_at`` is
``None`` unless a genuine ``updatedAt`` field is present.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone
from typing import Any

from src import ats_common
from src.ats_common import DEFAULT_TIMEOUT_SECONDS, clean_str

ATS_PROVIDER = "lever"
API_URL_TEMPLATE = "https://api.lever.co/v0/postings/{site}?mode=json"

# Sanity bounds for epoch-millisecond timestamps (2000-01-01 .. 2100-01-01 UTC).
_MIN_EPOCH_MS = 946_684_800_000
_MAX_EPOCH_MS = 4_102_444_800_000


class LeverError(Exception):
    """Raised when jobs cannot be fetched or the API response is unusable."""


def build_url(site: str) -> str:
    site = (site or "").strip()
    if not site:
        raise LeverError("Lever site identifier must be a non-empty string.")
    return API_URL_TEMPLATE.format(site=urllib.parse.quote(site, safe=""))


def fetch_postings_json(site: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[Any]:
    """GET the site's postings endpoint and return the decoded JSON list."""
    url = build_url(site)
    data = ats_common.get_json(url, timeout, LeverError)
    if not isinstance(data, list):
        raise LeverError(f"Malformed response from {url}: expected a JSON list of postings.")
    return data


def epoch_ms_to_iso(value: Any) -> str | None:
    """Convert an epoch-millisecond timestamp to an ISO-8601 UTC string, or None.

    Booleans, non-numbers and values outside 2000..2100 are rejected rather than
    guessed at, so a bad timestamp is treated as an unknown date downstream.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not (_MIN_EPOCH_MS <= value < _MAX_EPOCH_MS):
        return None
    instant = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return instant.isoformat(timespec="milliseconds")


def _description(job: dict[str, Any]) -> str:
    """Body HTML, then each titled list (responsibilities, requirements...), then the closing text."""
    parts: list[str] = []

    body = job.get("description")
    if isinstance(body, str) and body.strip():
        parts.append(ats_common.html_to_text(body))
    else:
        plain = clean_str(job.get("descriptionPlain"))
        if plain:
            parts.append(plain)

    lists = job.get("lists")
    if isinstance(lists, list):
        for section in lists:
            if not isinstance(section, dict):
                continue
            heading = clean_str(section.get("text"))
            content = section.get("content")
            text = ats_common.html_to_text(content) if isinstance(content, str) else ""
            block = "\n".join(p for p in (heading, text) if p)
            if block:
                parts.append(block)

    additional = job.get("additional")
    if isinstance(additional, str) and additional.strip():
        parts.append(ats_common.html_to_text(additional))
    else:
        plain = clean_str(job.get("additionalPlain"))
        if plain:
            parts.append(plain)

    return "\n\n".join(p for p in parts if p)


def normalize_job(job: dict[str, Any], company: str) -> dict[str, Any]:
    """Map one raw Lever posting to the normalized schema.

    Raises LeverError if the posting lacks a string ``id`` or a ``text`` (title).
    ``published_at`` comes from ``createdAt``, the employer-side creation time of
    the posting, which is the only posting date the public API exposes. It is
    never later than the actual publication, so the report's age cutoff can only
    err towards excluding a job, never towards showing a stale one.
    """
    if not isinstance(job, dict):
        raise LeverError(f"Job entry is not an object: {job!r:.80}")

    job_id = clean_str(job.get("id"))
    if job_id is None:
        raise LeverError(f"Job entry missing a valid 'id': {job!r:.80}")

    title = clean_str(job.get("text"))
    if title is None:
        raise LeverError(f"Job {job_id} missing a 'text' (title).")

    categories = job.get("categories")
    location = clean_str(categories.get("location")) if isinstance(categories, dict) else None

    updated = job.get("updatedAt")
    updated_at = epoch_ms_to_iso(updated) if not isinstance(updated, str) else clean_str(updated)

    return {
        "company": company,
        "ats_provider": ATS_PROVIDER,
        "ats_job_id": job_id,
        "title": title,
        "location": location,
        "description": _description(job),
        "application_url": clean_str(job.get("applyUrl")) or clean_str(job.get("hostedUrl")),
        "updated_at": updated_at,
        # Employer-side posting date (createdAt); kept separate from updated_at.
        "published_at": epoch_ms_to_iso(job.get("createdAt")),
    }


def fetch_lever_jobs(
    site: str,
    company: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch and normalize all postings for a Lever site.

    ``site`` is the config ``board_token`` (``acme`` in ``jobs.lever.co/acme``).
    ``company`` defaults to the site identifier. Any malformed posting fails the
    whole fetch so that a partial result is never mistaken for a complete snapshot.
    """
    data = fetch_postings_json(site, timeout=timeout)
    name = company or site
    return [normalize_job(raw, name) for raw in data]
