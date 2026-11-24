"""Employer publication dates and the report's posting-age cutoff.

Only a genuine employer publication date counts: Greenhouse ``first_published`` and
Ashby ``publishedAt`` (normalized to ``published_at`` by the provider modules and
stored as ``jobs.source_published_at``). ``updated_at``, ``first_seen_at`` and
``last_seen_at`` are never substituted for a missing publication date.

A date is *verified* only when it parses as ISO-8601 **with** a UTC offset; it is
then converted to timezone-aware UTC. Anything else (missing, unparseable, no
offset) is *unknown* and the job is excluded from the report.

Standard library only; reading snapshots never touches the network or the database.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MAX_AGE_DAYS = 30

RECENT = "recent"        # published within the cutoff window (boundary inclusive)
TOO_OLD = "too_old"      # published more than max_age_days ago
UNKNOWN = "unknown"      # publication date missing or unverified

SnapshotKey = tuple[str, str, str]  # (ats_provider, company name casefolded, ats_job_id)


def parse_published_at(value: Any) -> datetime | None:
    """Return a timezone-aware UTC datetime, or None if the value is not a verified date."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None  # no offset: the instant is ambiguous, so it is not verified
    return parsed.astimezone(timezone.utc)


def cutoff_instant(now: datetime, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> datetime:
    """Earliest publication instant still inside the window (inclusive)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc) - timedelta(days=max_age_days)


def classify_age(published: datetime | None, now: datetime, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> str:
    """RECENT if published at or after ``now - max_age_days``; TOO_OLD if earlier; UNKNOWN if None."""
    if published is None:
        return UNKNOWN
    return RECENT if published >= cutoff_instant(now, max_age_days) else TOO_OLD


def snapshot_key(ats_provider: Any, company: Any, ats_job_id: Any) -> SnapshotKey:
    return (str(ats_provider or "").strip().lower(), str(company or "").strip().casefold(),
            str(ats_job_id or "").strip())


def resolve_job_published_at(job: dict[str, Any],
                             snapshot_dates: dict[SnapshotKey, str] | None) -> datetime | None:
    """A job's verified employer publication date: latest snapshot first, then the database.

    ``job`` needs ``ats_provider``, ``company``, ``ats_job_id`` and (optionally)
    ``source_published_at``. Only publication dates are consulted; update and
    first/last-seen timestamps are never used as a substitute. Shared by the report
    and the batch evaluator so both apply exactly the same cutoff.
    """
    candidates = []
    if snapshot_dates:
        candidates.append(snapshot_dates.get(
            snapshot_key(job.get("ats_provider"), job.get("company"), job.get("ats_job_id"))))
    candidates.append(job.get("source_published_at"))
    for raw in candidates:
        parsed = parse_published_at(raw)
        if parsed is not None:
            return parsed
    return None


def load_snapshot_published_dates(data_dir: str | Path,
                                  warnings: list[str] | None = None) -> dict[SnapshotKey, str]:
    """Map (provider, company, posting id) -> raw ``published_at`` from ``data_dir/*_jobs.json``.

    Snapshots are written by ``src.collect_jobs`` on every run (with or without ``--db``),
    so they carry the freshest employer dates. Unreadable or malformed files are
    skipped (a note is appended to ``warnings``); jobs without a date are omitted.
    """
    dates: dict[SnapshotKey, str] = {}
    directory = Path(data_dir)
    if not directory.is_dir():
        return dates
    for path in sorted(directory.glob("*_jobs.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            jobs: Iterable[Any] = payload["jobs"]
            if not isinstance(jobs, list):
                raise TypeError("'jobs' is not a list")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if warnings is not None:
                warnings.append(f"skipped snapshot {path.name}: {exc.__class__.__name__}")
            continue
        for job in jobs:
            if not isinstance(job, dict):
                continue
            published = job.get("published_at")
            if isinstance(published, str) and published.strip():
                key = snapshot_key(job.get("ats_provider"), job.get("company"), job.get("ats_job_id"))
                dates[key] = published.strip()
    return dates
