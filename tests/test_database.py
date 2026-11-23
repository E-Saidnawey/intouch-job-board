"""Unit tests for SQLite persistence. Uses temporary database files; no network."""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src import collect_jobs, database

T0 = "2026-11-01T00:00:00+00:00"
T1 = "2026-11-02T00:00:00+00:00"
T2 = "2026-11-03T00:00:00+00:00"


def make_job(job_id: str, **overrides) -> dict:
    job = {
        "company": "Muon Space",
        "ats_provider": "greenhouse",
        "ats_job_id": job_id,
        "title": f"Engineer {job_id}",
        "location": "Mountain View, CA",
        "description": "Build satellites.",
        "application_url": f"https://job-boards.greenhouse.io/muonspace/jobs/{job_id}",
        "updated_at": "2026-10-30T12:00:00-04:00",
    }
    job.update(overrides)
    return job


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "nested" / "jobs.db"
        self.conn = database.initialize_database(self.db_path)
        self.company_id = database.upsert_company(self.conn, "Muon Space", "greenhouse", "muonspace")

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def row(self, ats_job_id: str) -> sqlite3.Row:
        return self.conn.execute(
            "SELECT * FROM jobs WHERE company_id = ? AND ats_job_id = ?", (self.company_id, ats_job_id)
        ).fetchone()

    def job_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


class SchemaAndCompanyTests(DatabaseTestCase):
    def test_initialize_is_idempotent_and_creates_tables(self):
        database.initialize_database(self.db_path).close()
        tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"companies", "jobs"} <= tables)

    def test_upsert_company_returns_same_id_and_updates_name(self):
        again = database.upsert_company(self.conn, "Muon Space Inc.", "greenhouse", "muonspace")
        self.assertEqual(again, self.company_id)
        rows = self.conn.execute("SELECT name FROM companies").fetchall()
        self.assertEqual([r["name"] for r in rows], ["Muon Space Inc."])
        other = database.upsert_company(self.conn, "Muon Space", "lever", "muonspace")
        self.assertNotEqual(other, self.company_id)

    def test_foreign_keys_enforced(self):
        self.assertEqual(self.conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        with self.assertRaises(sqlite3.IntegrityError):
            database.upsert_jobs(self.conn, 9999, [make_job("1")], seen_at=T0)
        self.assertEqual(self.job_count(), 0)


class UpsertJobsTests(DatabaseTestCase):
    def test_initial_inserts(self):
        counts = database.upsert_jobs(self.conn, self.company_id, [make_job("1"), make_job("2")], seen_at=T0)
        self.assertEqual(counts, {"inserted": 2, "updated": 0, "unchanged": 0})
        r = self.row("1")
        self.assertEqual(r["title"], "Engineer 1")
        self.assertEqual(r["source_updated_at"], "2026-10-30T12:00:00-04:00")
        self.assertEqual((r["first_seen_at"], r["last_seen_at"], r["last_changed_at"]), (T0, T0, T0))
        self.assertEqual(r["is_active"], 1)

    def test_missing_optional_fields_stored_as_null(self):
        job = {"ats_job_id": "7", "title": "Technician"}
        database.upsert_jobs(self.conn, self.company_id, [job], seen_at=T0)
        r = self.row("7")
        for field in ("location", "description", "application_url", "source_updated_at"):
            self.assertIsNone(r[field])

    def test_duplicate_prevention(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T0)
        counts = database.upsert_jobs(self.conn, self.company_id, [make_job("1"), make_job("1")], seen_at=T1)
        self.assertEqual(counts, {"inserted": 0, "updated": 0, "unchanged": 2})
        self.assertEqual(self.job_count(), 1)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO jobs (company_id, ats_job_id, title, first_seen_at, last_seen_at, last_changed_at)"
                " VALUES (?, '1', 't', ?, ?, ?)",
                (self.company_id, T0, T0, T0),
            )

    def test_unchanged_jobs_only_bump_last_seen(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T0)
        counts = database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T1)
        self.assertEqual(counts, {"inserted": 0, "updated": 0, "unchanged": 1})
        r = self.row("1")
        self.assertEqual((r["first_seen_at"], r["last_seen_at"], r["last_changed_at"]), (T0, T1, T0))

    def test_changed_description_updates_row(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T0)
        changed = make_job("1", description="Build MORE satellites.")
        counts = database.upsert_jobs(self.conn, self.company_id, [changed], seen_at=T1)
        self.assertEqual(counts, {"inserted": 0, "updated": 1, "unchanged": 0})
        r = self.row("1")
        self.assertEqual(r["description"], "Build MORE satellites.")
        self.assertEqual((r["first_seen_at"], r["last_seen_at"], r["last_changed_at"]), (T0, T1, T1))

    def test_each_source_field_counts_as_change(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T0)
        changes = {"title": "New", "location": None, "application_url": "https://x",
                   "updated_at": "2026-11-05T00:00:00Z"}
        for field, value in changes.items():
            with self.subTest(field=field):
                counts = database.upsert_jobs(
                    self.conn, self.company_id, [make_job("1", **{field: value})], seen_at=T1
                )
                self.assertEqual(counts["updated"], 1)
                database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T0)  # reset

    def test_timestamp_preservation_across_runs(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T0)
        database.upsert_jobs(self.conn, self.company_id, [make_job("1", title="Senior")], seen_at=T1)
        database.upsert_jobs(self.conn, self.company_id, [make_job("1", title="Senior")], seen_at=T2)
        r = self.row("1")
        self.assertEqual(r["first_seen_at"], T0)
        self.assertEqual(r["last_changed_at"], T1)
        self.assertEqual(r["last_seen_at"], T2)

    def test_missing_jobs_not_marked_inactive(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1"), make_job("2")], seen_at=T0)
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T1)
        r = self.row("2")
        self.assertEqual(r["is_active"], 1)
        self.assertEqual(r["last_seen_at"], T0)

    def test_default_seen_at_is_utc(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")])
        self.assertTrue(self.row("1")["first_seen_at"].endswith("+00:00"))

    def test_rollback_on_error_mid_batch(self):
        database.upsert_jobs(self.conn, self.company_id, [make_job("1")], seen_at=T0)
        batch = [make_job("1", description="changed"), make_job("2"), make_job("3", title=None)]
        with self.assertRaises(sqlite3.IntegrityError):  # title NOT NULL
            database.upsert_jobs(self.conn, self.company_id, batch, seen_at=T1)
        self.assertEqual(self.job_count(), 1)
        r = self.row("1")
        self.assertEqual(r["description"], "Build satellites.")
        self.assertEqual(r["last_seen_at"], T0)
        self.assertFalse(self.conn.in_transaction)

    def test_rollback_on_missing_job_id(self):
        with self.assertRaises(ValueError):
            database.upsert_jobs(self.conn, self.company_id, [make_job("1"), make_job("")], seen_at=T0)
        self.assertEqual(self.job_count(), 0)

    def test_parameterized_values_are_stored_literally(self):
        nasty = make_job("1'; DROP TABLE jobs; --", title="O'Brien \"quotes\"")
        database.upsert_jobs(self.conn, self.company_id, [nasty], seen_at=T0)
        self.assertEqual(self.row("1'; DROP TABLE jobs; --")["title"], "O'Brien \"quotes\"")


class CollectJobsDbFlagTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.out, self.db = tmp / "snap.json", tmp / "jobs.db"

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, jobs, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("src.greenhouse.fetch_greenhouse_jobs", return_value=jobs), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = collect_jobs.main(["--output", str(self.out), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_db_flag_persists_and_keeps_json_snapshot(self):
        code, out, _ = self.run_main([make_job("1"), make_job("2")], "--db", str(self.db))
        self.assertEqual(code, 0)
        self.assertIn("2 inserted, 0 updated, 0 unchanged", out)
        snap = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(snap["job_count"], 2)

        code, out, _ = self.run_main([make_job("1"), make_job("2", description="new")], "--db", str(self.db))
        self.assertEqual(code, 0)
        self.assertIn("0 inserted, 1 updated, 1 unchanged", out)

        conn = database.connect(self.db)
        try:
            row = conn.execute("SELECT c.name, j.first_seen_at, j.last_seen_at FROM jobs j "
                               "JOIN companies c ON c.id = j.company_id WHERE j.ats_job_id = '1'").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["name"], "Muon Space")
        self.assertLessEqual(row["first_seen_at"], row["last_seen_at"])

    def test_without_db_flag_no_database_created(self):
        code, _, _ = self.run_main([make_job("1")])
        self.assertEqual(code, 0)
        self.assertFalse(self.db.exists())

    def test_db_failure_exits_nonzero(self):
        with mock.patch.object(collect_jobs.database, "upsert_jobs", side_effect=sqlite3.OperationalError("locked")):
            code, _, err = self.run_main([make_job("1")], "--db", str(self.db))
        self.assertEqual(code, 1)
        self.assertIn("database update failed", err)


if __name__ == "__main__":
    unittest.main()
