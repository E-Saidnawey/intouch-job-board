"""Offline tests for batch Claude evaluation (src.batch_evaluator).

No network: the Anthropic client is a mock injected through ``client_factory``,
``anthropic.Anthropic`` is patched to fail if constructed, and the API key is a fake.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import anthropic
import httpx2 as httpx

from src import batch_evaluator as be
from src import claude_evaluator as ce
from src import database, evaluation_store as store
from src.candidate_profile import DEFAULT_PROFILE, load_profile
from tests.test_claude_evaluator import SYNTHETIC_PROFILE, fake_response, valid_evaluation

FAKE_KEY = "sk-ant-test-NOT-REAL"

# (company, ats_job_id, title, location, active) — expected outcome in the comment.
JOBS = [
    ("Alpha Orbital", "a1", "Payload Systems Engineer", "New York, NY", True),     # high, compatible -> eval
    ("Alpha Orbital", "a2", "Mechanical Design Engineer", "New York, NY", True),   # medium -> eval
    ("Beta Space", "b1", "Spacecraft GNC Engineer", "Remote", True),               # high, review -> eval
    ("Beta Space", "b2", "Thermal Engineer", "Boston, MA (Hybrid)", True),          # medium, boston hybrid -> eval
    ("Beta Space", "b3", "Payload Systems Engineer", "London, UK", True),           # incompatible location
    ("Gamma Launch", "g1", "Senior Accountant", "New York, NY", True),              # low relevance
    ("Gamma Launch", "g2", "Spacecraft GNC Engineer", "New York, NY", False),       # inactive: ignored
]
# Expected request order: HIGH before MEDIUM, then by relevance score.
EVALUATED_TITLES = ["Spacecraft GNC Engineer", "Payload Systems Engineer", "Mechanical Design Engineer",
                    "Thermal Engineer"]


def _http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.invalid/v1/messages"))


class BatchTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("anthropic.Anthropic", side_effect=AssertionError("real Anthropic client constructed"))
        patcher.start()
        self.addCleanup(patcher.stop)
        key_patch = mock.patch.object(be, "load_api_key", return_value=FAKE_KEY)
        self.load_key = key_patch.start()
        self.addCleanup(key_patch.stop)

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "jobs.db"
        self.add_jobs(JOBS)
        self.profile = load_profile(SYNTHETIC_PROFILE)
        self.config = ce.EvaluatorConfig()

        self.failing: dict[str, Exception | str] = {}   # title fragment -> exception or bad response text
        self.requested_titles: list[str] = []
        self.clients: list[mock.MagicMock] = []
        self.factory = mock.MagicMock(side_effect=self._make_client)

    # -- fixtures ------------------------------------------------------------

    def add_jobs(self, jobs, description: str = "Responsibilities: integrate optical payloads.") -> None:
        conn = database.initialize_database(self.db_path)
        try:
            by_company: dict[str, list[dict]] = {}
            for company, ats_id, title, location, active in jobs:
                by_company.setdefault(company, []).append(dict(
                    ats_job_id=ats_id, title=title, location=location, description=description,
                    application_url=f"https://example.invalid/{ats_id}", updated_at=None, _active=active))
            for company, rows in by_company.items():
                cid = database.upsert_company(conn, company, "greenhouse", company.lower().replace(" ", ""))
                database.upsert_jobs(conn, cid, rows)
                for row in rows:
                    if not row["_active"]:
                        conn.execute("UPDATE jobs SET is_active = 0 WHERE ats_job_id = ?", (row["ats_job_id"],))
        finally:
            conn.close()

    def _make_client(self, api_key, config):
        self.assertEqual(api_key, FAKE_KEY)

        def create(**kwargs):
            text = kwargs["messages"][0]["content"]
            posting = text.split("<job_posting>", 1)[1].split("</job_posting>", 1)[0]
            title = json.loads(posting)["title"]
            self.requested_titles.append(title)
            for failing_title, outcome in self.failing.items():
                if title == failing_title:
                    if isinstance(outcome, Exception):
                        raise outcome
                    return fake_response(outcome)
            return fake_response(json.dumps(valid_evaluation()))

        client = mock.MagicMock()
        client.messages.create.side_effect = create
        client.beta.messages.create.side_effect = create
        self.clients.append(client)
        return client

    def run_cli(self, *extra: str, profile: Path = SYNTHETIC_PROFILE):
        argv = ["--profile", str(profile), "--db", str(self.db_path), *extra]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = be.main(argv, client_factory=self.factory)
        return code, out.getvalue(), err.getvalue()

    def db(self) -> sqlite3.Connection:
        conn = database.connect(self.db_path)
        self.addCleanup(conn.close)
        return conn

    def stored(self) -> list[sqlite3.Row]:
        conn = self.db()
        if not store.table_exists(conn):
            return []
        return conn.execute(
            "SELECT e.*, j.title FROM evaluations e JOIN jobs j ON j.id = e.job_id ORDER BY e.id").fetchall()

    def plan(self, **kw) -> be.BatchPlan:
        conn = database.connect(self.db_path)
        try:
            return be.plan_batch(conn, self.profile, self.config, **kw)
        finally:
            conn.close()

    def file_digest(self) -> str:
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

class SelectionTests(BatchTestCase):
    def test_selects_eligible_high_and_medium_jobs_only(self):
        plan = self.plan()
        self.assertEqual([c.job["title"] for c in plan.pending], EVALUATED_TITLES)
        self.assertEqual([c.relevance for c in plan.pending], ["high", "high", "medium", "medium"])
        self.assertEqual(plan.cached, [])
        self.assertEqual(plan.skipped, {be.SKIP_LOCATION: 1, be.SKIP_RELEVANCE: 1})
        self.assertEqual(plan.active_jobs, 6)          # the inactive job is never considered

    def test_location_classes_match_shared_filter(self):
        classes = {c.job["ats_job_id"]: c.location_class for c in self.plan().pending}
        self.assertEqual(classes, {"a1": "compatible", "a2": "compatible", "b1": "review", "b2": "boston_hybrid"})

    def test_incomplete_record_is_skipped_not_requested(self):
        self.db().execute("UPDATE jobs SET description = '' WHERE ats_job_id = 'a2'")
        plan = self.plan()
        self.assertNotIn("a2", [c.job["ats_job_id"] for c in plan.pending])
        self.assertEqual(plan.skipped[be.SKIP_INVALID], 1)

    def test_no_five_job_limit(self):
        extra = [("Delta Orbit", f"d{i}", f"Payload Systems Engineer {i}", "New York, NY", True) for i in range(8)]
        self.add_jobs(extra)
        self.assertEqual(len(self.plan().pending), 12)
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.stored()), 12)
        self.assertIn("evaluated=12", out)

    def test_optional_limit(self):
        code, out, _ = self.run_cli("--call-api", "--limit", "2")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.stored()), 2)
        self.assertEqual(len(self.requested_titles), 2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            be.main(["--profile", str(SYNTHETIC_PROFILE), "--db", str(self.db_path), "--limit", "0"])


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #

class DryRunTests(BatchTestCase):
    def test_dry_run_lists_jobs_and_makes_no_request_or_write(self):
        before = self.file_digest()
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.factory.assert_not_called()
        self.assertEqual(self.file_digest(), before)
        self.assertFalse(store.table_exists(self.db()))          # dry run never creates the table
        self.assertIn("Would evaluate (one API request each):", out)
        for title in EVALUATED_TITLES:
            self.assertIn(title, out)
        self.assertNotIn("Senior Accountant", out)
        self.assertNotIn("London", out)
        self.assertIn("Dry run: no request was sent. Re-run with --call-api to make 4 API request(s).", out)

    def test_dry_run_without_key_reports_not_ready(self):
        self.load_key.return_value = None
        code, out, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("ANTHROPIC_API_KEY is not set", out)
        self.factory.assert_not_called()

    def test_dry_run_counts_cached_jobs(self):
        self.run_cli("--call-api", "--limit", "1")
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("already evaluated (cached): 1", out)
        self.assertIn("need evaluation:            3", out)


# --------------------------------------------------------------------------- #
# Cache reuse and history
# --------------------------------------------------------------------------- #

class CacheReuseTests(BatchTestCase):
    def test_second_run_reuses_cache_without_requests(self):
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.requested_titles), 4)
        self.assertIn("Summary: evaluated=4 cached=0 failed=0 skipped=2 not_attempted=0", out)

        self.requested_titles.clear()
        self.factory.reset_mock()
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 0)
        self.assertEqual(self.requested_titles, [])
        self.factory.assert_not_called()                 # no client is even constructed
        self.assertIn("Summary: evaluated=0 cached=4 failed=0 skipped=2 not_attempted=0", out)
        self.assertEqual(len(self.stored()), 4)

    def test_cache_hit_needs_no_api_key(self):
        self.run_cli("--call-api")
        self.load_key.return_value = None
        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0, err)
        self.assertIn("cached=4", out)

    def test_rows_match_single_job_cache_key(self):
        self.run_cli("--call-api")
        conn = self.db()
        for cand in be.plan_batch(conn, self.profile, self.config).cached:
            key = ce.make_cache_key(cand.job, self.profile, self.config)
            self.assertIsNotNone(store.find_cached(conn, key))

    def test_only_changed_postings_are_re_evaluated_and_history_kept(self):
        self.run_cli("--call-api")
        first_rows = [tuple(r) for r in self.stored()]
        self.db().execute("UPDATE jobs SET description = 'New responsibilities.' WHERE ats_job_id = 'b1'")
        self.requested_titles.clear()
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 0)
        self.assertEqual(self.requested_titles, ["Spacecraft GNC Engineer"])
        self.assertIn("evaluated=1 cached=3", out)
        rows = self.stored()
        self.assertEqual(len(rows), 5)
        self.assertEqual([tuple(r) for r in rows[:4]], first_rows)      # earlier rows untouched
        self.assertEqual(sum(r["title"] == "Spacecraft GNC Engineer" for r in rows), 2)

    def test_single_job_evaluation_is_reused_by_batch(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                mock.patch.object(ce, "load_api_key", return_value=FAKE_KEY):
            code = ce.main(["--provider", "greenhouse", "--posting-id", "a1", "--profile", str(SYNTHETIC_PROFILE),
                            "--db", str(self.db_path), "--call-api"], client_factory=self.factory)
        self.assertEqual(code, 0)
        self.requested_titles.clear()
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.requested_titles), 3)
        self.assertIn("evaluated=3 cached=1", out)

    def test_profile_or_model_change_invalidates(self):
        self.run_cli("--call-api")
        self.requested_titles.clear()
        code, out, _ = self.run_cli("--call-api", "--model", "claude-sonnet-5-5")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.requested_titles), 4)
        self.assertEqual(len(self.stored()), 8)


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #

class FailureTests(BatchTestCase):
    def test_individual_failure_does_not_stop_batch_and_is_not_cached(self):
        self.failing["Spacecraft GNC Engineer"] = "not json"
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 1)
        self.assertEqual(len(self.requested_titles), 4)          # every job was still attempted
        self.assertIn("FAILED", out)
        self.assertIn("Summary: evaluated=3 cached=0 failed=1 skipped=2 not_attempted=0", out)
        self.assertNotIn("Spacecraft GNC Engineer", [r["title"] for r in self.stored()])

        # The next run retries only the failed job.
        self.failing.clear()
        self.requested_titles.clear()
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 0)
        self.assertEqual(self.requested_titles, ["Spacecraft GNC Engineer"])
        self.assertIn("evaluated=1 cached=3 failed=0", out)

    def test_api_errors_refusals_and_invalid_schema_are_isolated(self):
        self.failing["Payload Systems Engineer"] = anthropic.RateLimitError(
            "rate limited", response=_http_response(429), body=None)
        self.failing["Mechanical Design Engineer"] = json.dumps(valid_evaluation(overall_fit_score=900))
        code, out, _ = self.run_cli("--call-api", "--max-consecutive-failures", "5")
        self.assertEqual(code, 1)
        self.assertIn("rate limit", out)
        self.assertIn("evaluated=2 cached=0 failed=2", out)
        self.assertEqual(sorted(r["title"] for r in self.stored()), ["Spacecraft GNC Engineer", "Thermal Engineer"])

    def test_authentication_error_stops_batch(self):
        self.failing[EVALUATED_TITLES[0]] = anthropic.AuthenticationError(
            "bad key", response=_http_response(401), body=None)
        code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 1)
        self.assertEqual(len(self.requested_titles), 1)
        self.assertIn("evaluated=0 cached=0 failed=1 skipped=2 not_attempted=3", out)
        self.assertIn("Batch stopped: Claude API authentication failed", out)
        self.assertEqual(self.stored(), [])

    def test_consecutive_failures_stop_batch(self):
        for title in EVALUATED_TITLES:
            self.failing[title] = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.invalid"))
        code, out, _ = self.run_cli("--call-api", "--max-consecutive-failures", "2")
        self.assertEqual(code, 1)
        self.assertEqual(len(self.requested_titles), 2)
        self.assertIn("failed=2 skipped=2 not_attempted=2", out)
        self.assertIn("stopped after 2 consecutive failures", out)

    def test_save_failure_counted_and_batch_continues(self):
        real_save = be.save_evaluation
        calls = {"n": 0}

        def flaky_save(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("disk I/O error")
            return real_save(*args, **kwargs)

        with mock.patch.object(be, "save_evaluation", side_effect=flaky_save):
            code, out, _ = self.run_cli("--call-api")
        self.assertEqual(code, 1)
        self.assertIn("could not be saved", out)
        self.assertIn("evaluated=3 cached=0 failed=1", out)
        self.assertEqual(len(self.stored()), 3)

    def test_missing_key_with_pending_jobs_makes_no_request(self):
        self.load_key.return_value = None
        code, _, err = self.run_cli("--call-api")
        self.assertEqual(code, 2)
        self.assertIn("ANTHROPIC_API_KEY is not set", err)
        self.factory.assert_not_called()
        self.assertEqual(self.stored(), [])


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #

class SafetyTests(BatchTestCase):
    def test_real_profile_refused(self):
        code, _, err = self.run_cli("--call-api", profile=DEFAULT_PROFILE)
        self.assertEqual(code, 2)
        self.assertIn("refusing", err)
        self.factory.assert_not_called()

    def test_missing_database_is_not_created(self):
        missing = Path(self._tmp.name) / "nope.db"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            code = be.main(["--profile", str(SYNTHETIC_PROFILE), "--db", str(missing), "--call-api"],
                           client_factory=self.factory)
        self.assertEqual(code, 2)
        self.assertFalse(missing.exists())
        self.factory.assert_not_called()

    def test_output_never_contains_key_profile_or_description(self):
        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0)
        text = out + err
        self.assertNotIn(FAKE_KEY, text)
        self.assertNotIn("integrate optical payloads", text)
        profile_text = json.dumps(self.profile)
        for value in ("Synthetic State University", "calibration pipeline"):
            self.assertIn(value, profile_text)
            self.assertNotIn(value, text)

    def test_cli_has_no_application_or_contact_options(self):
        options = {o for action in be.build_parser()._actions for o in action.option_strings}
        for word in ("apply", "submit", "email", "contact", "send", "force"):
            self.assertFalse(any(word in o for o in options), options)

    def test_evaluations_are_validated_and_location_policy_untouched(self):
        # Stored rows are the validated raw evaluation (policy is applied at display time, as before).
        self.run_cli("--call-api")
        for row in self.stored():
            self.assertEqual(ce.validate_evaluation(json.loads(row["evaluation_json"])), [])
            self.assertEqual(row["schema_version"], ce.EVALUATION_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
