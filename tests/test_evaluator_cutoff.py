"""Offline tests for the batch evaluator's posting-date cutoff.

No network and no Claude API: the client is the mock from ``BatchTestCase`` (which also
makes constructing a real ``anthropic.Anthropic`` fail) and the API key is fake.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import batch_evaluator as be
from src import claude_evaluator as ce
from src import database, generate_report, posting_dates
from src import evaluation_store as store
from tests.test_batch_evaluator import BatchTestCase
from tests.test_claude_evaluator import valid_evaluation

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat()


class EvaluatorCutoffTestCase(BatchTestCase):
    """Eligible fixture jobs a1, a2, b1, b2 (see tests.test_batch_evaluator.JOBS) with chosen dates."""

    def set_published(self, ats_job_id: str, value: str | None) -> None:
        self.db().execute("UPDATE jobs SET source_published_at = ? WHERE ats_job_id = ?", (value, ats_job_id))

    def set_dates(self, now: datetime) -> None:
        self.set_published("a1", iso(now - timedelta(days=2)))                 # recent
        self.set_published("a2", iso(now - timedelta(days=45)))                # too old
        self.set_published("b1", None)                                         # unknown
        self.db().execute("UPDATE jobs SET source_updated_at = ? WHERE ats_job_id = 'b1'",
                          (iso(now - timedelta(hours=1)),))                    # recent update is NOT a substitute
        self.set_published("b2", iso(now - timedelta(days=29)))                # recent

    def pending_ids(self, plan: be.BatchPlan) -> list[str]:
        return [c.job["ats_job_id"] for c in plan.pending]


class PlanCutoffTests(EvaluatorCutoffTestCase):
    def test_old_and_undated_jobs_are_skipped(self):
        self.set_dates(NOW)
        plan = self.plan(max_age_days=30, now=NOW)
        self.assertEqual(sorted(self.pending_ids(plan)), ["a1", "b2"])
        self.assertEqual(plan.skipped[be.SKIP_TOO_OLD], 1)
        self.assertEqual(plan.skipped[be.SKIP_UNKNOWN_DATE], 1)
        self.assertEqual(plan.cutoff, NOW - timedelta(days=30))
        self.assertEqual(plan.active_jobs, 6)

    def test_exact_boundary_is_kept_and_one_second_older_is_skipped(self):
        self.set_published("a1", iso(NOW - timedelta(days=30)))
        self.set_published("a2", iso(NOW - timedelta(days=30, seconds=1)))
        plan = self.plan(max_age_days=30, now=NOW)
        self.assertIn("a1", self.pending_ids(plan))
        self.assertNotIn("a2", self.pending_ids(plan))

    def test_non_utc_offsets_are_converted(self):
        # 2026-10-24T05:00-08:00 == 13:00Z, one hour after the cutoff (12:00Z) -> kept.
        self.set_published("a1", "2026-10-24T05:00:00-08:00")
        # 2026-10-24T03:00-08:00 == 11:00Z, before the cutoff -> skipped.
        self.set_published("a2", "2026-10-24T03:00:00-08:00")
        ids = self.pending_ids(self.plan(max_age_days=30, now=NOW))
        self.assertIn("a1", ids)
        self.assertNotIn("a2", ids)

    def test_naive_or_garbage_dates_are_unknown(self):
        self.set_published("a1", "2026-11-20T10:00:00")      # no offset: not verified
        self.set_published("a2", "last Tuesday")
        plan = self.plan(max_age_days=30, now=NOW)
        self.assertNotIn("a1", self.pending_ids(plan))
        self.assertNotIn("a2", self.pending_ids(plan))
        self.assertEqual(plan.skipped[be.SKIP_UNKNOWN_DATE], 2)

    def test_no_cutoff_by_default_for_programmatic_callers(self):
        self.set_dates(NOW)
        plan = self.plan()
        self.assertEqual(sorted(self.pending_ids(plan)), ["a1", "a2", "b1", "b2"])
        self.assertIsNone(plan.cutoff)
        self.assertNotIn(be.SKIP_TOO_OLD, plan.skipped)

    def test_snapshot_date_takes_precedence_over_database(self):
        self.set_dates(NOW)
        snapshots = {
            # DB says unknown, the latest snapshot has a recent date -> evaluated.
            posting_dates.snapshot_key("greenhouse", "Beta Space", "b1"): iso(NOW - timedelta(days=1)),
            # DB says recent, the latest snapshot says old -> skipped.
            posting_dates.snapshot_key("greenhouse", "Alpha Orbital", "a1"): iso(NOW - timedelta(days=90)),
        }
        ids = self.pending_ids(self.plan(max_age_days=30, now=NOW, snapshot_dates=snapshots))
        self.assertEqual(sorted(ids), ["b1", "b2"])

    def test_cutoff_applies_before_location_and_relevance(self):
        for job_id in ("a1", "a2", "b1", "b2", "b3", "g1"):
            self.set_published(job_id, iso(NOW - timedelta(days=60)))
        plan = self.plan(max_age_days=30, now=NOW)
        self.assertEqual(plan.pending, [])
        self.assertEqual(dict(plan.skipped), {be.SKIP_TOO_OLD: 6})

    def test_old_job_with_cached_evaluation_is_neither_cached_nor_requested(self):
        self.run_cli("--call-api")                      # all four are recent: evaluate and cache them
        self.assertEqual(len(self.stored()), 4)
        self.set_published("a2", iso(datetime.now(timezone.utc) - timedelta(days=45)))
        plan = self.plan(max_age_days=30)
        self.assertEqual(len(plan.cached), 3)
        self.assertEqual(plan.pending, [])
        self.assertEqual(plan.skipped[be.SKIP_TOO_OLD], 1)
        self.assertEqual(len(self.stored()), 4)         # evaluations are never deleted

    def test_cutoff_matches_the_report(self):
        self.set_dates(NOW)
        conn = database.connect(self.db_path)
        try:
            jobs = be.load_active_jobs(conn)
            report_jobs = {j["job_id"]: j for j in generate_report.load_active_jobs(conn)}
        finally:
            conn.close()
        for job in jobs:
            with self.subTest(job=job["ats_job_id"]):
                self.assertEqual(posting_dates.resolve_job_published_at(job, None),
                                 generate_report.resolve_published_at(report_jobs[job["job_id"]], None))


class CacheCompatibilityTests(EvaluatorCutoffTestCase):
    def test_publication_date_does_not_change_cache_key_or_prompt(self):
        conn = database.connect(self.db_path)
        try:
            job = next(j for j in be.load_active_jobs(conn) if j["ats_job_id"] == "a1")
        finally:
            conn.close()
        self.assertIn("source_published_at", job)
        without = {k: v for k, v in job.items() if k != "source_published_at"}
        self.assertEqual(ce.make_cache_key(job, self.profile, self.config),
                         ce.make_cache_key(without, self.profile, self.config))
        self.assertEqual(ce.build_user_message(job, self.profile), ce.build_user_message(without, self.profile))

    def test_evaluation_cached_before_the_change_is_still_reused(self):
        conn = database.connect(self.db_path)
        try:
            store.ensure_schema(conn)
            job = next(j for j in be.load_active_jobs(conn) if j["ats_job_id"] == "a1")
            legacy_job = {k: v for k, v in job.items() if k != "source_published_at"}
            store.save_evaluation(conn, ce.make_cache_key(legacy_job, self.profile, self.config),
                                  valid_evaluation(), "claude-opus-5-5")
        finally:
            conn.close()
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("already evaluated (cached): 1", out)


class CliCutoffTests(EvaluatorCutoffTestCase):
    def setUp(self):
        super().setUp()
        self.set_dates(datetime.now(timezone.utc))

    def test_default_cli_applies_30_day_cutoff_and_reports_it(self):
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.factory.assert_not_called()
        self.assertIn("Posting-date cutoff: published on/after ", out)
        self.assertIn("(last 30 days); older or undated jobs are skipped, not deleted", out)
        self.assertIn("Eligible (location + HIGH/MEDIUM relevance): 2", out)
        self.assertIn(f"  {be.SKIP_TOO_OLD}: 1", out)
        self.assertIn(f"  {be.SKIP_UNKNOWN_DATE}: 1", out)
        self.assertIn("Re-run with --call-api to make 2 API request(s).", out)
        self.assertNotIn("Mechanical Design Engineer", out)    # a2, too old
        self.assertNotIn("Spacecraft GNC Engineer", out)       # b1, undated

    def test_call_api_requests_only_recent_jobs(self):
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(self.requested_titles), ["Payload Systems Engineer", "Thermal Engineer"])
        self.assertEqual(len(self.stored()), 2)
        self.assertIn("evaluated=2 cached=0 failed=0 skipped=4", out)

    def test_max_age_days_widens_the_window(self):
        code, out, _ = self.run_cli("--max-age-days", "60")
        self.assertEqual(code, 0)
        self.assertIn("(last 60 days)", out)
        self.assertIn("Mechanical Design Engineer", out)       # 45 days old: inside 60
        self.assertNotIn("Spacecraft GNC Engineer", out)       # still undated

    def test_invalid_max_age_days_rejected(self):
        for value in ("0", "-3", "abc"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.run_cli("--max-age-days", value)
        self.factory.assert_not_called()

    def test_data_dir_snapshots_supply_dates(self):
        snap_dir = Path(self._tmp.name) / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "beta_space_jobs.json").write_text(json.dumps({"jobs": [{
            "ats_provider": "greenhouse", "company": "Beta Space", "ats_job_id": "b1",
            "published_at": iso(datetime.now(timezone.utc) - timedelta(days=3))}]}), encoding="utf-8")
        (snap_dir / "broken_jobs.json").write_text("{not json", encoding="utf-8")
        code, out, _ = self.run_cli("--data-dir", str(snap_dir))
        self.assertEqual(code, 0)
        self.assertIn("Spacecraft GNC Engineer", out)          # b1 now dated via the snapshot
        self.assertIn("Note: skipped snapshot broken_jobs.json", out)

    def test_default_data_dir_is_database_directory(self):
        (self.db_path.parent / "beta_space_jobs.json").write_text(json.dumps({"jobs": [{
            "ats_provider": "greenhouse", "company": "Beta Space", "ats_job_id": "b1",
            "published_at": iso(datetime.now(timezone.utc) - timedelta(days=3))}]}), encoding="utf-8")
        _, out, _ = self.run_cli()
        self.assertIn("Spacecraft GNC Engineer", out)

    def test_existing_options_unchanged(self):
        parser = be.build_parser()
        args = parser.parse_args(["--profile", "p.json"])
        self.assertEqual(args.db, ce.DEFAULT_DB)
        self.assertFalse(args.call_api)
        self.assertIsNone(args.limit)
        self.assertEqual(args.max_consecutive_failures, be.DEFAULT_MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(args.max_age_days, 30)
        self.assertIsNone(args.data_dir)


class LegacyDatabaseTests(BatchTestCase):
    def test_database_without_publication_column_is_read_only_and_all_undated(self):
        legacy = Path(self._tmp.name) / "legacy.db"
        conn = sqlite3.connect(legacy)
        src = database.connect(self.db_path)
        try:
            cols = [r[1] for r in src.execute("PRAGMA table_info(jobs)") if r[1] != database.PUBLISHED_COLUMN]
            src.execute(f"ATTACH DATABASE ? AS legacy", (str(legacy),))
            src.execute("CREATE TABLE legacy.companies AS SELECT * FROM main.companies")
            src.execute(f"CREATE TABLE legacy.jobs AS SELECT {', '.join(cols)} FROM main.jobs")
            src.commit()
            src.execute("DETACH DATABASE legacy")
        finally:
            src.close()
            conn.close()
        before = legacy.read_bytes()
        self.db_path = legacy
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.assertIn(f"  {be.SKIP_UNKNOWN_DATE}: 6", out)      # cutoff is checked first, for every active job
        self.assertEqual(legacy.read_bytes(), before)          # no migration, no write
