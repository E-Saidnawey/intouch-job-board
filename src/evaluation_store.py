"""SQLite persistence and caching for Claude job evaluations.

Adds one table, ``evaluations``, to the existing jobs database. The migration is
idempotent and runs in one transaction. It never touches the ``companies`` or
``jobs`` tables. For an existing ``evaluations`` table it only ever *adds* things:

* v1 -> v2: ``ALTER TABLE ... ADD COLUMN location_key`` (nullable). Rows written
  before v2 keep ``location_key = NULL``; they stay in the table as history but
  can never match a cache lookup, because their location is unknown.
* The v2 cache-key index is created and the v1 index (no data) is dropped.

A pre-existing table missing any v1 column is rejected and left untouched.

Each row is one successful, validated evaluation. Rows are only ever inserted,
never updated or deleted, so history is preserved when the posting, profile,
model or schema version changes (and when ``--force`` re-evaluates).

Cache key: (job_id, description_sha256, profile_sha256, model, schema_version,
location_key), where ``location_key`` is the job's normalized location string
(see ``normalize_location``). Effort is deliberately not part of the key.
The most recent matching row wins.

Privacy: only a SHA-256 hash of the candidate profile is stored, never its contents.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from src.database import _Transaction, utc_now

EVALUATIONS_TABLE = "evaluations"

_CREATE_TABLE = (
    """
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY,
        job_id INTEGER NOT NULL REFERENCES jobs(id),
        description_sha256 TEXT NOT NULL CHECK (length(description_sha256) = 64),
        profile_sha256 TEXT NOT NULL CHECK (length(profile_sha256) = 64),
        model TEXT NOT NULL CHECK (length(model) > 0),
        response_model TEXT,
        schema_version INTEGER NOT NULL,
        evaluation_json TEXT NOT NULL,
        recommendation TEXT NOT NULL CHECK (recommendation IN ('Apply', 'Review', 'Skip')),
        overall_fit_score INTEGER NOT NULL CHECK (overall_fit_score BETWEEN 0 AND 100),
        evaluated_at TEXT NOT NULL,
        location_key TEXT
    )
    """
)
_ADD_LOCATION_COLUMN = "ALTER TABLE evaluations ADD COLUMN location_key TEXT"
_OLD_INDEX = "idx_evaluations_cache_key"
_CREATE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_evaluations_cache_key_v2
        ON evaluations (job_id, description_sha256, profile_sha256, model, schema_version, location_key,
                        evaluated_at)
"""

# Columns every evaluations table must have (the original v1 shape).
BASE_COLUMNS = frozenset({
    "id", "job_id", "description_sha256", "profile_sha256", "model", "response_model",
    "schema_version", "evaluation_json", "recommendation", "overall_fit_score", "evaluated_at",
})
LOCATION_COLUMN = "location_key"
EXPECTED_COLUMNS = BASE_COLUMNS | {LOCATION_COLUMN}


class EvaluationStoreError(Exception):
    """The evaluations table is unusable (e.g. an incompatible pre-existing table)."""


# --------------------------------------------------------------------------- #
# Hashing (local only)
# --------------------------------------------------------------------------- #

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_location(location: str | None) -> str:
    """Cache-key form of a job location: whitespace collapsed, trimmed, case-folded.

    ``None`` / blank become ``""``. Any other textual change (e.g. "Remote" ->
    "New York, NY") produces a different key.
    """
    if not isinstance(location, str):
        return ""
    return " ".join(location.split()).casefold()


def profile_sha256(profile: dict[str, Any]) -> str:
    """Hash a profile's canonical JSON, so formatting/key-order edits don't change it."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(canonical)


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #

def table_exists(conn: sqlite3.Connection, name: str = EVALUATIONS_TABLE) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({EVALUATIONS_TABLE})")}


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or upgrade the evaluations table. Safe to run repeatedly.

    Requires the ``jobs`` table to exist. Raises EvaluationStoreError if an
    ``evaluations`` table already exists without the v1 columns (it is left untouched).
    Existing rows are never modified or deleted.
    """
    if not table_exists(conn, "jobs"):
        raise EvaluationStoreError("jobs table not found; this is not a collector database")
    with _Transaction(conn):
        conn.execute(_CREATE_TABLE)
        _check_columns(conn)  # before any change, so an incompatible table is never altered
        if LOCATION_COLUMN not in _columns(conn):
            conn.execute(_ADD_LOCATION_COLUMN)
        conn.execute(_CREATE_INDEX)
        conn.execute(f"DROP INDEX IF EXISTS {_OLD_INDEX}")


def _check_columns(conn: sqlite3.Connection) -> None:
    missing = BASE_COLUMNS - _columns(conn)
    if missing:
        raise EvaluationStoreError(
            f"existing {EVALUATIONS_TABLE} table is incompatible (missing columns: {sorted(missing)}); "
            "it was not modified"
        )


# --------------------------------------------------------------------------- #
# Cache read / write
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CacheKey:
    job_id: int
    description_sha256: str
    profile_sha256: str
    model: str
    schema_version: int
    location_key: str  # normalize_location(job location)


@dataclass(frozen=True)
class StoredEvaluation:
    id: int
    evaluation: dict[str, Any]
    response_model: str | None
    evaluated_at: str


def find_cached(conn: sqlite3.Connection, key: CacheKey) -> StoredEvaluation | None:
    """Return the most recent evaluation matching every part of ``key``, or None.

    Works on read-only connections and never migrates. Returns None if the table
    does not exist yet or predates ``location_key`` (no row there can match).
    """
    if not table_exists(conn):
        return None
    _check_columns(conn)
    if LOCATION_COLUMN not in _columns(conn):
        return None
    row = conn.execute(
        """
        SELECT id, evaluation_json, response_model, evaluated_at FROM evaluations
        WHERE job_id = ? AND description_sha256 = ? AND profile_sha256 = ?
              AND model = ? AND schema_version = ? AND location_key = ?
        ORDER BY evaluated_at DESC, id DESC
        LIMIT 1
        """,
        (key.job_id, key.description_sha256, key.profile_sha256, key.model, key.schema_version,
         key.location_key),
    ).fetchone()
    if row is None:
        return None
    try:
        evaluation = json.loads(row[1])
    except json.JSONDecodeError:
        return None  # unreadable row: treat as a miss rather than trusting it
    return StoredEvaluation(id=int(row[0]), evaluation=evaluation, response_model=row[2], evaluated_at=row[3])


def save_evaluation(
    conn: sqlite3.Connection,
    key: CacheKey,
    evaluation: dict[str, Any],
    response_model: str | None,
    evaluated_at: str | None = None,
) -> StoredEvaluation:
    """Insert one validated evaluation (never updates or replaces earlier rows)."""
    evaluated_at = evaluated_at or utc_now()
    payload = json.dumps(evaluation, sort_keys=True, ensure_ascii=False)
    with _Transaction(conn):
        cursor = conn.execute(
            """
            INSERT INTO evaluations (job_id, description_sha256, profile_sha256, model, response_model,
                                     schema_version, evaluation_json, recommendation, overall_fit_score,
                                     evaluated_at, location_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key.job_id, key.description_sha256, key.profile_sha256, key.model, response_model,
             key.schema_version, payload, evaluation["recommendation"], evaluation["overall_fit_score"],
             evaluated_at, key.location_key),
        )
    return StoredEvaluation(id=int(cursor.lastrowid), evaluation=evaluation,
                            response_model=response_model, evaluated_at=evaluated_at)


def count_evaluations(conn: sqlite3.Connection, job_id: int | None = None) -> int:
    if not table_exists(conn):
        return 0
    if job_id is None:
        return conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM evaluations WHERE job_id = ?", (job_id,)).fetchone()[0]
