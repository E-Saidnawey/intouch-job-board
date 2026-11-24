"""Collect jobs for configured companies and save per-company JSON snapshots.

Run from the project root:
    python -m src.collect_jobs --all --db data/jobs.db
    python -m src.collect_jobs --company "Muon Space"
    python -m src.collect_jobs                       # legacy default: Muon Space only

Companies are defined in config/companies.json (override with --config). That file is
git-ignored; create it from config/companies.example.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):  # allow `python src/collect_jobs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import database, providers  # noqa: E402
from src.company_config import CompanyConfig, ConfigError, load_config, select_companies  # noqa: E402
from src.providers import CollectionError  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "companies.json"        # personal, git-ignored
EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "companies.example.json"  # public template
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_COMPANY = "Muon Space"  # used when neither --all nor --company is given

EXIT_OK = 0
EXIT_COLLECTION_FAILED = 1
EXIT_CONFIG_ERROR = 2


def dedupe_jobs(jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicates by (ats_provider, company, ats_job_id), keeping the first."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for job in jobs:
        key = (job["ats_provider"], job["company"], job["ats_job_id"])
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def build_snapshot(
    company: CompanyConfig, jobs: list[dict[str, Any]], collected_at: datetime | None = None
) -> dict[str, Any]:
    collected_at = collected_at or datetime.now(timezone.utc)
    return {
        "collected_at": collected_at.isoformat(timespec="seconds"),
        "source": {
            "ats_provider": company.ats_provider,
            "board_token": company.board_token,
            "company": company.name,
        },
        "job_count": len(jobs),
        "jobs": jobs,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to a temp file in the target directory, then os.replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def collect_company(
    company: CompanyConfig,
    output: Path,
    conn: sqlite3.Connection | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Fetch, dedupe, snapshot and (optionally) persist one company.

    The snapshot is only written after a fully valid fetch, so a failure never
    overwrites an existing snapshot. Database writes run in their own
    transaction per company.
    """
    jobs = providers.fetch_jobs(company.ats_provider, company.board_token, company=company.name)
    unique = dedupe_jobs(jobs)
    if not unique and not allow_empty:
        raise CollectionError(
            "API returned zero jobs; refusing to overwrite snapshot (use --allow-empty to override)."
        )
    snapshot = build_snapshot(company, unique)
    write_json_atomic(output, snapshot)
    dupes = len(jobs) - len(unique)
    print(
        f"Collected {snapshot['job_count']} {company.name} jobs"
        + (f" ({dupes} duplicates removed)" if dupes else "")
        + f" at {snapshot['collected_at']} -> {output}"
    )

    result: dict[str, Any] = {"job_count": snapshot["job_count"], "db": None}
    if conn is not None:
        try:
            company_id = database.upsert_company(conn, company.name, company.ats_provider, company.board_token)
            counts = database.upsert_jobs(conn, company_id, unique, seen_at=snapshot["collected_at"])
        except (sqlite3.Error, ValueError) as exc:
            raise CollectionError(f"database update failed and was rolled back: {exc}") from exc
        print(
            f"  Database: {counts['inserted']} inserted, "
            f"{counts['updated']} updated, {counts['unchanged']} unchanged"
        )
        result["db"] = counts
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect aerospace jobs from configured ATS job boards.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--all", action="store_true", help="Collect from all enabled companies.")
    target.add_argument(
        "--company", action="append", metavar="NAME",
        help="Collect one configured company by name or board token (repeatable).",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Company config JSON.")
    parser.add_argument("--db", type=Path, default=None, help="Optional SQLite database to upsert jobs into.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory for JSON snapshots.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Snapshot path override (only when collecting a single company).")
    parser.add_argument("--allow-empty", action="store_true", help="Allow saving a snapshot with zero jobs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        companies = select_companies(
            load_config(args.config), names=args.company or [DEFAULT_COMPANY], all_enabled=args.all
        )
    except ConfigError as exc:
        print(f"ERROR: configuration: {exc}", file=sys.stderr)
        if not args.config.exists() and EXAMPLE_CONFIG.exists():
            print(f"Hint: create it from the template: copy {EXAMPLE_CONFIG.relative_to(PROJECT_ROOT)} "
                  f"to {args.config} and fill in your companies.", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if args.output is not None and len(companies) != 1:
        parser.error("--output can only be used when collecting exactly one company.")

    conn = None
    if args.db is not None:
        try:
            conn = database.initialize_database(args.db)
        except (sqlite3.Error, OSError) as exc:
            print(f"ERROR: cannot open database {args.db}: {exc}", file=sys.stderr)
            return EXIT_COLLECTION_FAILED

    failures: list[str] = []
    try:
        for company in companies:
            output = args.output or args.data_dir / company.snapshot_filename
            try:
                collect_company(company, output, conn=conn, allow_empty=args.allow_empty)
            except (CollectionError, OSError) as exc:
                failures.append(company.name)
                print(f"ERROR: {company.name}: {exc} (existing snapshot left unchanged)", file=sys.stderr)
    finally:
        if conn is not None:
            conn.close()

    if len(companies) > 1:
        print(f"Summary: {len(companies) - len(failures)}/{len(companies)} companies succeeded"
              + (f"; failed: {', '.join(failures)}" if failures else ""))
    return EXIT_COLLECTION_FAILED if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
