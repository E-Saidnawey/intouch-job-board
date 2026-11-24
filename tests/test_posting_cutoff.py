"""Offline tests for publication dates, the 30-day report cutoff, the seen/unseen report
controls, new-company endpoints and CLI compatibility.

No network (urlopen is mocked) and no Claude API calls: the evaluator client is never built.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from src import ashby, batch_evaluator, collect_jobs, database, greenhouse, posting_dates
from src import generate_report as gr
from src.company_config import load_config, parse_config
from tests.test_claude_evaluator import SYNTHETIC_PROFILE
from tests.test_multi_company import FakeResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
URLOPEN = "src.ats_common.urllib.request.urlopen"  # shared by the Greenhouse and Ashby clients


def iso(dt: datetime) -> str:
    return dt.isoformat()


def job(ats_job_id: str, title: str = "Satellite Operations Engineer", *, location: str = "New York, NY",
        published_at: str | None = None, updated_at: str | None = None, company: str = "Alpha Orbital") -> dict:
    return {"company": company, "ats_provider": "greenhouse", "ats_job_id": ats_job_id, "title": title,
            "location": location, "description": f"Description {ats_job_id}.",
            "application_url": f"https://example.invalid/jobs/{ats_job_id}",
            "updated_at": updated_at, "published_at": published_at}


class TempDbCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.db_path = self.tmp / "jobs.db"
        self.inputs = gr.CurrentInputs(model="claude-opus-5-5", schema_version=2, profile_sha256=None)

    def seed(self, jobs: list[dict], company: str = "Alpha Orbital", seen_at: str | None = None) -> None:
        conn = database.initialize_database(self.db_path)
        try:
            cid = database.upsert_company(conn, company, "greenhouse", company.lower().replace(" ", ""))
            database.upsert_jobs(conn, cid, jobs, seen_at=seen_at)
        finally:
            conn.close()

    def report(self, **kwargs) -> gr.ReportData:
        conn = gr.open_readonly(self.db_path)
        try:
            return gr.build_report_data(conn, self.inputs, now=kwargs.pop("now", NOW), **kwargs)
        finally:
            conn.close()

    def ids(self, data: gr.ReportData) -> set[str]:
        return {e.job["ats_job_id"] for e in data.entries}


# --------------------------------------------------------------------------- #
# Date parsing and classification
# --------------------------------------------------------------------------- #

class ParsePublishedAtTests(unittest.TestCase):
    def test_offsets_converted_to_utc(self):
        self.assertEqual(posting_dates.parse_published_at("2026-11-17T17:01:20-05:00"),
                         datetime(2026, 11, 17, 22, 1, 20, tzinfo=timezone.utc))
        self.assertEqual(posting_dates.parse_published_at("2026-09-15T13:57:17.235+00:00").tzinfo, timezone.utc)
        self.assertEqual(posting_dates.parse_published_at("2026-11-01T00:00:00Z"),
                         datetime(2026, 11, 1, tzinfo=timezone.utc))

    def test_unverified_values_are_none(self):
        for bad in (None, "", "   ", "not a date", "2026-11-01T00:00:00", "2026-11-01", 1730000000, {"x": 1}):
            with self.subTest(value=bad):
                self.assertIsNone(posting_dates.parse_published_at(bad))

    def test_classify_boundaries(self):
        cutoff = NOW - timedelta(days=30)
        self.assertEqual(posting_dates.classify_age(cutoff, NOW), posting_dates.RECENT)          # exactly on boundary
        self.assertEqual(posting_dates.classify_age(cutoff - timedelta(seconds=1), NOW), posting_dates.TOO_OLD)
        self.assertEqual(posting_dates.classify_age(NOW, NOW), posting_dates.RECENT)
        self.assertEqual(posting_dates.classify_age(None, NOW), posting_dates.UNKNOWN)

    def test_naive_now_rejected(self):
        with self.assertRaises(ValueError):
            posting_dates.cutoff_instant(datetime(2026, 11, 23))


# --------------------------------------------------------------------------- #
# Adapters capture the employer publication date (and only it)
# --------------------------------------------------------------------------- #

class AdapterPublicationDateTests(unittest.TestCase):
    def test_greenhouse_uses_first_published_not_updated_at(self):
        raw = {"id": 1, "title": "Engineer", "first_published": "2026-02-25T06:32:38-05:00",
               "updated_at": "2026-11-11T08:39:42-05:00"}
        normalized = greenhouse.normalize_job(raw, "LeoLabs")
        self.assertEqual(normalized["published_at"], "2026-02-25T06:32:38-05:00")
        self.assertEqual(normalized["updated_at"], "2026-11-11T08:39:42-05:00")

    def test_greenhouse_missing_first_published_is_none(self):
        normalized = greenhouse.normalize_job({"id": 1, "title": "Engineer", "updated_at": "2026-11-11T08:00:00Z"}, "X")
        self.assertIsNone(normalized["published_at"])  # updated_at is never substituted

    def test_ashby_uses_published_at(self):
        raw = {"id": "abc", "title": "Engineer", "publishedAt": "2026-09-15T13:57:17.235+00:00"}
        normalized = ashby.normalize_job(raw, "Slingshot Aerospace")
        self.assertEqual(normalized["published_at"], "2026-09-15T13:57:17.235+00:00")
        self.assertIsNone(normalized["updated_at"])


# --------------------------------------------------------------------------- #
# Database: additive column, history retained, duplicates prevented
# --------------------------------------------------------------------------- #

LEGACY_SCHEMA = database.SCHEMA  # the jobs table before the publication-date column


class DatabaseTests(TempDbCase):
    def test_legacy_database_migrated_additively(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(LEGACY_SCHEMA)
        conn.execute("INSERT INTO companies (name, ats_provider, board_token) VALUES ('A', 'greenhouse', 'a')")
        conn.execute("INSERT INTO jobs (company_id, ats_job_id, title, first_seen_at, last_seen_at, last_changed_at) "
                     "VALUES (1, '1', 'Old job', 'x', 'x', 'x')")
        conn.commit()
        conn.close()

        conn = database.initialize_database(self.db_path)
        try:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
            self.assertIn("source_published_at", columns)
            row = conn.execute("SELECT title, source_published_at, first_seen_at FROM jobs").fetchone()
            self.assertEqual((row["title"], row["source_published_at"], row["first_seen_at"]), ("Old job", None, "x"))
            self.assertFalse(database.ensure_published_column(conn))  # idempotent
        finally:
            conn.close()

    def test_report_reads_legacy_database_without_migrating(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(LEGACY_SCHEMA)
        conn.commit()
        conn.close()
        before = self.db_path.read_bytes()
        data = self.report()
        self.assertEqual(data.entries, [])
        self.assertEqual(self.db_path.read_bytes(), before)

    def test_duplicate_jobs_prevented_and_published_date_stored(self):
        published = iso(NOW - timedelta(days=3))
        self.seed([job("1", published_at=published), job("2", published_at=None)], seen_at="2026-11-20T00:00:00+00:00")
        self.seed([job("1", published_at=published), job("2", published_at=None)], seen_at="2026-11-21T00:00:00+00:00")
        conn = database.connect(self.db_path)
        try:
            rows = conn.execute("SELECT ats_job_id, source_published_at, first_seen_at, last_changed_at "
                                "FROM jobs ORDER BY ats_job_id").fetchall()
        finally:
            conn.close()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_published_at"], published)
        self.assertIsNone(rows[1]["source_published_at"])
        self.assertEqual(rows[0]["first_seen_at"], "2026-11-20T00:00:00+00:00")
        self.assertEqual(rows[0]["last_changed_at"], "2026-11-20T00:00:00+00:00")  # date is not a "change"

    def test_missing_date_never_erases_stored_date_and_backfill_is_not_a_change(self):
        published = iso(NOW - timedelta(days=3))
        self.seed([job("1", published_at=None)], seen_at="2026-11-20T00:00:00+00:00")
        self.seed([job("1", published_at=published)], seen_at="2026-11-21T00:00:00+00:00")
        self.seed([job("1", published_at=None)], seen_at="2026-11-22T00:00:00+00:00")
        conn = database.connect(self.db_path)
        try:
            row = conn.execute("SELECT source_published_at, last_changed_at, last_seen_at FROM jobs").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["source_published_at"], published)
        self.assertEqual(row["last_changed_at"], "2026-11-20T00:00:00+00:00")
        self.assertEqual(row["last_seen_at"], "2026-11-22T00:00:00+00:00")


# --------------------------------------------------------------------------- #
# 30-day cutoff applied before rendering
# --------------------------------------------------------------------------- #

class CutoffTests(TempDbCase):
    def setUp(self):
        super().setUp()
        self.seed([
            job("recent", "Satellite Operations Engineer", published_at=iso(NOW - timedelta(days=5))),
            job("boundary", "Payload Systems Engineer", published_at=iso(NOW - timedelta(days=30))),
            job("just-old", "Spacecraft Software Engineer", published_at=iso(NOW - timedelta(days=30, seconds=1))),
            job("old", "Satellite Systems Engineer", published_at=iso(NOW - timedelta(days=90))),
            # Missing publication date, although update and seen timestamps are recent:
            job("unknown", "Mission Operations Engineer", published_at=None, updated_at=iso(NOW)),
            job("naive", "GNC Engineer", published_at="2026-11-20T00:00:00"),
            job("garbage", "Flight Software Engineer", published_at="last Tuesday"),
        ], seen_at=iso(NOW))

    def test_old_and_unknown_jobs_excluded_boundary_included(self):
        data = self.report()
        self.assertEqual(self.ids(data), {"recent", "boundary"})
        m = data.metrics
        self.assertEqual((m["active_jobs"], m["recent_jobs"], m["excluded_too_old"], m["excluded_unknown_date"]),
                         (7, 2, 2, 3))
        self.assertEqual(data.cutoff, NOW - timedelta(days=30))

    def test_updated_and_seen_times_never_substituted(self):
        entries = {e.job["ats_job_id"] for e in self.report().entries}
        self.assertNotIn("unknown", entries)  # updated_at / first_seen_at / last_seen_at are all "now"

    def test_excluded_jobs_never_rendered(self):
        page = gr.render_html(self.report())
        for title in ("Spacecraft Software Engineer", "Satellite Systems Engineer", "Mission Operations Engineer",
                      "GNC Engineer", "Flight Software Engineer"):
            self.assertNotIn(title, page)
        self.assertIn("Satellite Operations Engineer", page)
        self.assertIn("Payload Systems Engineer", page)
        self.assertIn("2 published earlier, 3 with a missing or unverified publication date", page)

    def test_custom_window(self):
        self.assertEqual(self.ids(self.report(max_age_days=100)), {"recent", "boundary", "just-old", "old"})
        self.assertEqual(self.ids(self.report(max_age_days=7)), {"recent"})

    def test_history_retained(self):
        self.report()
        conn = database.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 7)
        finally:
            conn.close()

    def test_snapshot_date_used_and_preferred(self):
        snapshot_dates = {
            posting_dates.snapshot_key("greenhouse", "Alpha Orbital", "unknown"): iso(NOW - timedelta(days=1)),
            posting_dates.snapshot_key("greenhouse", "alpha orbital", "old"): iso(NOW - timedelta(days=2)),
        }
        self.assertEqual(self.ids(self.report(snapshot_dates=snapshot_dates)), {"recent", "boundary", "unknown", "old"})

    def test_invalid_snapshot_date_falls_back_to_database(self):
        snapshot_dates = {posting_dates.snapshot_key("greenhouse", "Alpha Orbital", "recent"): "garbage"}
        self.assertIn("recent", self.ids(self.report(snapshot_dates=snapshot_dates)))

    def test_main_logs_exclusions_and_reads_snapshots_next_to_db(self):
        (self.tmp / "alpha_orbital_jobs.json").write_text(json.dumps({
            "collected_at": iso(NOW), "source": {"ats_provider": "greenhouse", "board_token": "alphaorbital",
                                                 "company": "Alpha Orbital"}, "job_count": 1,
            "jobs": [dict(job("unknown"), published_at=iso(datetime.now(timezone.utc) - timedelta(days=1)))],
        }), encoding="utf-8")
        (self.tmp / "broken_jobs.json").write_text("{not json", encoding="utf-8")
        output = self.tmp / "out" / "report.html"
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = gr.main(["--db", str(self.db_path), "--output", str(output), "--max-age-days", "36500"])
        self.assertEqual(code, 0, err.getvalue())
        text = out.getvalue()
        self.assertIn("Posting-date cutoff:", text)
        self.assertIn("excluded 0 older and 2 with a missing or unverified publication date", text)
        self.assertIn("Note: skipped snapshot broken_jobs.json", text)
        self.assertIn("Mission Operations Engineer", output.read_text(encoding="utf-8"))  # date from snapshot

    def test_invalid_max_age_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            gr.build_parser().parse_args(["--max-age-days", "0"])


# --------------------------------------------------------------------------- #
# Sorting and seen/unseen controls in the rendered HTML
# --------------------------------------------------------------------------- #

class ReportControlsTests(TempDbCase):
    def setUp(self):
        super().setUp()
        self.seed([
            job("mid", "Payload Systems Engineer", published_at=iso(NOW - timedelta(days=10))),
            job("new", "Satellite Operations Engineer", published_at=iso(NOW - timedelta(days=1))),
            job("oldest", "Spacecraft Systems Engineer", published_at=iso(NOW - timedelta(days=20))),
        ])
        self.data = self.report(generated_at=iso(NOW))
        self.page = gr.render_html(self.data)

    def cards(self) -> list[str]:
        return re.findall(r'<article class="job".*?</article>', self.page, flags=re.S)

    def test_cards_rendered_newest_first(self):
        uids = [re.search(r'data-uid="([^"]+)"', c).group(1) for c in self.cards()]
        self.assertEqual(uids, ["greenhouse|Alpha Orbital|new", "greenhouse|Alpha Orbital|mid",
                                "greenhouse|Alpha Orbital|oldest"])
        published = [int(re.search(r'data-published="(\d+)"', c).group(1)) for c in self.cards()]
        self.assertEqual(published, sorted(published, reverse=True))

    def test_sort_newest_first_helper_keeps_rank_for_ties(self):
        entries = self.data.entries
        same = entries[0].published_at
        for e in entries:
            e.published_at = same
        self.assertEqual(gr.sort_newest_first(entries), entries)

    def test_posting_date_visible(self):
        card = self.cards()[0]
        self.assertIn('<time datetime="2026-11-22T12:00:00+00:00">2026-11-22</time>', card)
        self.assertIn("1 day ago", card)

    def test_default_view_unseen_and_sort_newest(self):
        self.assertRegex(self.page, r'data-view="unseen" aria-pressed="true"')
        self.assertRegex(self.page, r'data-view="all" aria-pressed="false"')
        self.assertRegex(self.page, r'data-view="seen" aria-pressed="false"')
        self.assertIn('<option value="newest" selected>Newest first</option>', self.page)
        self.assertIn('<option value="oldest">Oldest first</option>', self.page)
        self.assertIn('<option value="match">Highest match</option>', self.page)
        script = re.search(r"<script>(.*?)</script>", self.page, flags=re.S).group(1)
        self.assertIn("var view = 'unseen'", script)

    def test_seen_toggle_and_storage_wiring(self):
        for card in self.cards():
            self.assertIn('<button type="button" class="seen-toggle" aria-pressed="false">Mark as seen</button>', card)
        script = re.search(r"<script>(.*?)</script>", self.page, flags=re.S).group(1)
        for needle in ("localStorage.getItem(STORAGE_KEY)", "localStorage.setItem(STORAGE_KEY", "'Mark as unseen'",
                       "delete seen[card.dataset.uid]", "counts[s ? 'seen' : 'unseen']", "try {"):
            self.assertIn(needle, script)

    def test_self_contained(self):
        self.assertNotRegex(self.page, r'<(script|link|img|iframe)[^>]+(src|href)=')
        self.assertNotIn("fetch(", self.page)
        self.assertNotIn("XMLHttpRequest", self.page)
        self.assertEqual(self.page.count("<script>"), 1)


# --------------------------------------------------------------------------- #
# Company configuration and new-company endpoint handling
# --------------------------------------------------------------------------- #

class CompanyConfigCompatibilityTests(unittest.TestCase):
    def test_existing_config_unchanged_and_valid(self):
        path = PROJECT_ROOT / "config" / "companies.json"
        if not path.exists():
            self.skipTest("config/companies.json not present (personal, git-ignored)")
        companies = load_config(path)
        self.assertEqual(
            [(c.name, c.ats_provider, c.board_token, c.enabled, c.snapshot_filename) for c in companies[:5]],
            [("Muon Space", "greenhouse", "muonspace", True, "muon_space_jobs.json"),
             ("Planet Labs", "greenhouse", "planetlabs", True, "planet_labs_jobs.json"),
             ("AST SpaceMobile", "greenhouse", "astspacemobile", True, "ast_spacemobile_jobs.json"),
             ("LeoLabs", "greenhouse", "leolabsinc", True, "leolabs_jobs.json"),
             ("Slingshot Aerospace", "ashby", "slingshot-aerospace", True, "slingshot_aerospace_jobs.json")])

    def test_new_company_endpoints(self):
        companies = parse_config({"companies": [
            {"name": "Example Orbital", "ats_provider": "greenhouse", "board_token": "exampleorbital"},
            {"name": "Example Launch", "ats_provider": "Ashby", "board_token": "example-launch"},
        ]})
        self.assertEqual(greenhouse.build_url(companies[0].board_token),
                         "https://boards-api.greenhouse.io/v1/boards/exampleorbital/jobs?content=true")
        self.assertEqual(companies[1].ats_provider, "ashby")
        self.assertEqual(ashby.build_url(companies[1].board_token),
                         "https://api.ashbyhq.com/posting-api/job-board/example-launch")


class NewCompanyCollectionTests(unittest.TestCase):
    """End-to-end collection for newly configured Greenhouse and Ashby companies, all HTTP mocked."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.config, self.data_dir, self.db = tmp / "companies.json", tmp / "data", tmp / "data" / "jobs.db"
        self.config.write_text(json.dumps({"companies": [
            {"name": "Example Orbital", "ats_provider": "greenhouse", "board_token": "exampleorbital"},
            {"name": "Example Launch", "ats_provider": "ashby", "board_token": "example-launch"},
        ]}), encoding="utf-8")
        self.requested: list[str] = []

    def route(self, request, timeout=None):
        self.requested.append(request.full_url)
        if request.full_url.startswith("https://boards-api.greenhouse.io/"):
            return self.gh(request)
        if request.full_url.startswith("https://api.ashbyhq.com/"):
            return self.ab(request)
        raise AssertionError(f"unexpected URL {request.full_url}")

    def gh(self, request):
        self.assertIn("/boards/exampleorbital/jobs", request.full_url)
        jobs = [{"id": 7, "title": "Satellite Operations Engineer", "location": {"name": "New York, NY"},
                 "first_published": "2026-11-20T09:00:00-05:00", "updated_at": "2026-11-21T09:00:00-05:00"},
                {"id": 7, "title": "Satellite Operations Engineer"}]  # duplicate posting id
        return FakeResponse(json.dumps({"jobs": jobs}).encode())

    def ab(self, request):
        self.assertTrue(request.full_url.endswith("/job-board/example-launch"))
        jobs = [{"id": "u-1", "title": "Launch Engineer", "location": "Remote",
                 "publishedAt": "2026-11-19T00:00:00.000+00:00"}]
        return FakeResponse(json.dumps({"jobs": jobs}).encode())

    def run_collect(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch(URLOPEN, self.route), redirect_stdout(out), redirect_stderr(err):
            code = collect_jobs.main(["--all", "--config", str(self.config), "--data-dir", str(self.data_dir),
                                      "--db", str(self.db)])
        return code, out.getvalue(), err.getvalue()

    def test_collects_snapshots_and_db_with_publication_dates(self):
        code, out, err = self.run_collect()
        self.assertEqual(code, 0, err)
        self.assertIn("(1 duplicates removed)", out)
        self.assertEqual(self.requested, [
            "https://boards-api.greenhouse.io/v1/boards/exampleorbital/jobs?content=true",
            "https://api.ashbyhq.com/posting-api/job-board/example-launch"])
        gh_snap = json.loads((self.data_dir / "example_orbital_jobs.json").read_text(encoding="utf-8"))
        self.assertEqual(gh_snap["job_count"], 1)
        self.assertEqual(gh_snap["jobs"][0]["published_at"], "2026-11-20T09:00:00-05:00")
        dates = posting_dates.load_snapshot_published_dates(self.data_dir)
        self.assertEqual(dates[posting_dates.snapshot_key("ashby", "Example Launch", "u-1")],
                         "2026-11-19T00:00:00.000+00:00")

        code, _, err = self.run_collect()  # second run: no duplicate rows
        self.assertEqual(code, 0, err)
        conn = database.connect(self.db)
        try:
            rows = conn.execute("SELECT ats_job_id, source_published_at FROM jobs ORDER BY ats_job_id").fetchall()
        finally:
            conn.close()
        self.assertEqual([tuple(r) for r in rows],
                         [("7", "2026-11-20T09:00:00-05:00"), ("u-1", "2026-11-19T00:00:00.000+00:00")])


# --------------------------------------------------------------------------- #
# Existing CLI compatibility (InTouch commands)
# --------------------------------------------------------------------------- #

class CliCompatibilityTests(unittest.TestCase):
    def test_collect_jobs_arguments(self):
        args = collect_jobs.build_parser().parse_args(["--all"])
        self.assertTrue(args.all)
        self.assertIsNone(args.db)  # `--all` alone still writes snapshots only
        self.assertEqual(args.config, PROJECT_ROOT / "config" / "companies.json")
        self.assertEqual(args.data_dir, PROJECT_ROOT / "data")
        for argv in (["--company", "Muon Space", "--db", "data/jobs.db"], ["--all", "--allow-empty"],
                     ["--company", "LeoLabs", "--output", "x.json"]):
            collect_jobs.build_parser().parse_args(argv)

    def test_batch_evaluator_arguments(self):
        args = batch_evaluator.build_parser().parse_args(
            ["--profile", "config/claude_candidate_profile.json", "--call-api"])
        self.assertTrue(args.call_api)
        self.assertEqual(args.profile, Path("config/claude_candidate_profile.json"))
        self.assertEqual(Path(args.db).resolve(), (PROJECT_ROOT / "data" / "jobs.db").resolve())

    def test_generate_report_arguments_and_paths(self):
        args = gr.build_parser().parse_args(["--profile", "config/claude_candidate_profile.json"])
        self.assertEqual(args.db, PROJECT_ROOT / "data" / "jobs.db")
        self.assertEqual(args.output, PROJECT_ROOT / "reports" / "job_intelligence.html")
        self.assertEqual(args.max_age_days, 30)
        self.assertIsNone(args.data_dir)  # defaults to the database's directory (data/)
        args = gr.build_parser().parse_args(["--profile", "p.json", "--model", "m", "--db", "d.db", "--output", "o.html"])
        self.assertEqual((args.model, args.db, args.output), ("m", Path("d.db"), Path("o.html")))

    def test_report_main_never_calls_network_or_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "jobs.db"
            database.initialize_database(db).close()
            out = io.StringIO()
            with mock.patch("anthropic.Anthropic", side_effect=AssertionError("API client constructed")), \
                    mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used")), \
                    redirect_stdout(out), redirect_stderr(io.StringIO()):
                code = gr.main(["--db", str(db), "--output", str(Path(tmp) / "r.html"),
                                "--profile", str(SYNTHETIC_PROFILE)])
            self.assertEqual(code, 0)
            self.assertIn("Report written:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
