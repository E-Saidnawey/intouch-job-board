"""Collector for public Ashby job boards.

Uses only the Python standard library. The public Posting API
(https://developers.ashbyhq.com/docs/public-job-posting-api) requires no
authentication:

    GET https://api.ashbyhq.com/posting-api/job-board/{board_token}
    -> {"jobs": [{"id", "title", "location", "jobUrl", "applyUrl",
                  "publishedAt", "descriptionHtml", ...}, ...]}
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from src import ats_common
from src.ats_common import DEFAULT_TIMEOUT_SECONDS, clean_str

ATS_PROVIDER = "ashby"
API_URL_TEMPLATE = "https://api.ashbyhq.com/posting-api/job-board/{token}"


class AshbyError(Exception):
    """Raised when jobs cannot be fetched or the API response is unusable."""


def build_url(board_token: str) -> str:
    token = (board_token or "").strip()
    if not token:
        raise AshbyError("Board token must be a non-empty string.")
    return API_URL_TEMPLATE.format(token=urllib.parse.quote(token, safe=""))


def fetch_board_json(board_token: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """GET the board's postings endpoint and return the decoded JSON object."""
    url = build_url(board_token)
    data = ats_common.get_json(url, timeout, AshbyError)
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise AshbyError(f"Malformed response from {url}: expected an object with a 'jobs' list.")
    return data


def normalize_job(job: dict[str, Any], company: str) -> dict[str, Any]:
    """Map one raw Ashby posting to the normalized schema.

    Raises AshbyError if the posting lacks a string ``id`` or a ``title``.
    ``updated_at`` is taken only from a genuine ``updatedAt`` field; ``publishedAt``
    is a publication date, not an update timestamp, so it is never used.
    """
    if not isinstance(job, dict):
        raise AshbyError(f"Job entry is not an object: {job!r:.80}")

    job_id = clean_str(job.get("id"))
    if job_id is None:
        raise AshbyError(f"Job entry missing a valid 'id': {job!r:.80}")

    title = clean_str(job.get("title"))
    if title is None:
        raise AshbyError(f"Job {job_id} missing a 'title'.")

    description_html = job.get("descriptionHtml")
    if isinstance(description_html, str) and description_html.strip():
        description = ats_common.html_to_text(description_html)
    else:
        description = clean_str(job.get("descriptionPlain")) or ""

    return {
        "company": company,
        "ats_provider": ATS_PROVIDER,
        "ats_job_id": job_id,
        "title": title,
        "location": clean_str(job.get("location")),
        "description": description,
        "application_url": clean_str(job.get("applyUrl")) or clean_str(job.get("jobUrl")),
        "updated_at": clean_str(job.get("updatedAt")),
    }


def fetch_ashby_jobs(
    board_token: str,
    company: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch and normalize all postings for an Ashby job board.

    ``company`` defaults to the board token (the Ashby response carries no
    company name). Any malformed posting fails the whole fetch so that a
    partial or corrupted result is never mistaken for a complete snapshot.
    """
    data = fetch_board_json(board_token, timeout=timeout)
    name = company or board_token
    return [normalize_job(raw, name) for raw in data["jobs"]]
