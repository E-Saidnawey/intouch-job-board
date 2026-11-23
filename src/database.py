"""SQLite persistence for collected jobs (standard-library sqlite3 only)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    ats_provider TEXT NOT NULL,
    board_token TEXT NOT NULL,
    UNIQUE(ats_provider, board_token)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL,
    ats_job_id TEXT NOT NULL,
    title TEXT NOT NULL,
    location TEXT,
    description TEXT,
    application_url TEXT,
    source_updated_at TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_changed_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(company_id) REFERENCES companies(id),
    UNIQUE(company_id, ats_job_id)
);
"""

# Fields that come from the ATS; a difference in any of them is a "change".
SOURCE_FIELDS = ("title", "location", "description", "application_url", "source_updated_at")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with foreign keys on and manual transaction control."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)  # we issue BEGIN/COMMIT ourselves
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database(db_path: str | Path) -> sqlite3.Connection:
    """Create the database file and schema if needed; return an open connection."""
    path = Path(db_path)
    if str(db_path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    conn.executescript(SCHEMA)
    return conn


class _Transaction:
    """BEGIN IMMEDIATE ... COMMIT, rolling back on any exception."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def upsert_company(conn: sqlite3.Connection, name: str, ats_provider: str, board_token: str) -> int:
    """Insert or update a company keyed on (ats_provider, board_token); return its id."""
    with _Transaction(conn):
        conn.execute(
            """
            INSERT INTO companies (name, ats_provider, board_token) VALUES (?, ?, ?)
            ON CONFLICT(ats_provider, board_token) DO UPDATE SET name = excluded.name
            """,
            (name, ats_provider, board_token),
        )
        row = conn.execute(
            "SELECT id FROM companies WHERE ats_provider = ? AND board_token = ?",
            (ats_provider, board_token),
        ).fetchone()
    return int(row["id"])


def _source_values(job: dict[str, Any]) -> tuple[Any, ...]:
    """Extract source fields from a normalized job (``updated_at`` -> ``source_updated_at``)."""
    values = dict(job)
    if "source_updated_at" not in values:
        values["source_updated_at"] = job.get("updated_at")
    return tuple(values.get(f) for f in SOURCE_FIELDS)


def upsert_jobs(
    conn: sqlite3.Connection,
    company_id: int,
    jobs: Iterable[dict[str, Any]],
    seen_at: str | None = None,
) -> dict[str, int]:
    """Insert new jobs and update changed ones in a single transaction.

    - New job: inserted with first_seen_at = last_seen_at = last_changed_at = seen_at.
    - Existing job, source fields identical: only last_seen_at is updated.
    - Existing job, any source field differs: source fields, last_seen_at and
      last_changed_at are updated. first_seen_at is never modified.

    Jobs absent from ``jobs`` are left untouched (not marked inactive).
    Any error rolls back the whole batch. Returns inserted/updated/unchanged counts.
    """
    seen_at = seen_at or utc_now()
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}

    with _Transaction(conn):
        for job in jobs:
            ats_job_id = job.get("ats_job_id")
            if ats_job_id is None or str(ats_job_id).strip() == "":
                raise ValueError(f"Job missing ats_job_id: {job!r:.80}")
            ats_job_id = str(ats_job_id)
            new_values = _source_values(job)

            existing = conn.execute(
                f"SELECT id, {', '.join(SOURCE_FIELDS)} FROM jobs WHERE company_id = ? AND ats_job_id = ?",
                (company_id, ats_job_id),
            ).fetchone()

            if existing is None:
                conn.execute(
                    f"""
                    INSERT INTO jobs (company_id, ats_job_id, {', '.join(SOURCE_FIELDS)},
                                      first_seen_at, last_seen_at, last_changed_at)
                    VALUES (?, ?, {', '.join('?' * len(SOURCE_FIELDS))}, ?, ?, ?)
                    """,
                    (company_id, ats_job_id, *new_values, seen_at, seen_at, seen_at),
                )
                counts["inserted"] += 1
            elif tuple(existing[f] for f in SOURCE_FIELDS) != new_values:
                assignments = ", ".join(f"{f} = ?" for f in SOURCE_FIELDS)
                conn.execute(
                    f"UPDATE jobs SET {assignments}, last_seen_at = ?, last_changed_at = ? WHERE id = ?",
                    (*new_values, seen_at, seen_at, existing["id"]),
                )
                counts["updated"] += 1
            else:
                conn.execute("UPDATE jobs SET last_seen_at = ? WHERE id = ?", (seen_at, existing["id"]))
                counts["unchanged"] += 1

    return counts
