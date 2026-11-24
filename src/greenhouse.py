"""Collector for public Greenhouse job boards.

Uses only the Python standard library. The public Job Board API is documented at
https://developers.greenhouse.io/job-board.html and requires no authentication.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from src import ats_common
from src.ats_common import DEFAULT_TIMEOUT_SECONDS, clean_str

ATS_PROVIDER = "greenhouse"
API_URL_TEMPLATE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"


class GreenhouseError(Exception):
    """Raised when jobs cannot be fetched or the API response is unusable."""


def html_to_text(raw: str | None) -> str:
    """Convert Greenhouse ``content`` (entity-escaped HTML) to plain text."""
    return ats_common.html_to_text(raw, entity_escaped=True)


def build_url(board_token: str) -> str:
    token = (board_token or "").strip()
    if not token:
        raise GreenhouseError("Board token must be a non-empty string.")
    return API_URL_TEMPLATE.format(token=urllib.parse.quote(token, safe=""))


def fetch_board_json(board_token: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """GET the board's jobs endpoint and return the decoded JSON object."""
    url = build_url(board_token)
    data = ats_common.get_json(url, timeout, GreenhouseError)
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise GreenhouseError(f"Malformed response from {url}: expected an object with a 'jobs' list.")
    return data


def normalize_job(job: dict[str, Any], company: str) -> dict[str, Any]:
    """Map one raw Greenhouse job to the normalized schema.

    Raises GreenhouseError if the job lacks the fields needed to identify it.
    Missing optional fields become ``None`` (or "" for description); dates are
    never invented.
    """
    if not isinstance(job, dict):
        raise GreenhouseError(f"Job entry is not an object: {job!r:.80}")

    job_id = job.get("id")
    if isinstance(job_id, bool) or not isinstance(job_id, (int, str)) or str(job_id).strip() == "":
        raise GreenhouseError(f"Job entry missing a valid 'id': {job!r:.80}")

    title = clean_str(job.get("title"))
    if title is None:
        raise GreenhouseError(f"Job {job_id} missing a 'title'.")

    location = job.get("location")
    location_name = clean_str(location.get("name")) if isinstance(location, dict) else None

    return {
        "company": company,
        "ats_provider": ATS_PROVIDER,
        "ats_job_id": str(job_id).strip(),
        "title": title,
        "location": location_name,
        "description": html_to_text(job.get("content") if isinstance(job.get("content"), str) else None),
        "application_url": clean_str(job.get("absolute_url")),
        "updated_at": clean_str(job.get("updated_at")),
        # Employer publication date; never derived from updated_at.
        "published_at": clean_str(job.get("first_published")),
    }


def fetch_greenhouse_jobs(
    board_token: str,
    company: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch and normalize all jobs for a Greenhouse board token.

    ``company`` defaults to the API's ``company_name`` field, falling back to
    the board token. Any malformed job entry fails the whole fetch so that a
    partial or corrupted result is never mistaken for a complete snapshot.
    """
    data = fetch_board_json(board_token, timeout=timeout)
    jobs: list[dict[str, Any]] = []
    for raw in data["jobs"]:
        name = company or (clean_str(raw.get("company_name")) if isinstance(raw, dict) else None) or board_token
        jobs.append(normalize_job(raw, name))
    return jobs
