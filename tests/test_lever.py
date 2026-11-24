"""Tests for the Lever collector, provider dispatch and SQLite integration.

All HTTP is mocked; no network, no Claude API. Fixture postings are shaped like the
live public Postings API response (a top-level JSON list).
"""

from __future__ import annotations

import io
import json
import socket
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from src import collect_jobs, database, posting_dates, providers
from src.company_config import ConfigError, parse_config
from src.lever import LeverError, epoch_ms_to_iso, fetch_lever_jobs
from src.providers import CollectionError

URLOPEN = "src.ats_common.urllib.request.urlopen"
SITE = "acme-aero"  # fictional site identifier; never contacted
SITE_URL = f"https://api.lever.co/v0/postings/{SITE}?mode=json"
UUID = "ad208490-4052-4f91-a57e-433f2d1e484b"
CREATED_MS = 1_788_199_939_703  # 2026-08-31T18:12:19.703+00:00
CREATED_ISO = "2026-08-31T18:12:19.703+00:00"


class FakeResponse:
    def __init__(self, body: bytes | str):
        self._body = body.encode("utf-8") if isinstance(body, str) else body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def respond(payload) -> mock.Mock:
    body = payload if isinstance(payload, (str, bytes)) else json.dumps(payload)
    return mock.Mock(return_value=FakeResponse(body))


def http_error(code: int, reason: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(SITE_URL, code, reason, hdrs=None,
                                  fp=io.BytesIO(b'{"ok":false,"error":"Document not found"}'))


def lever_posting(job_id: str = UUID, **overrides) -> dict:
    posting = {
        "id": job_id,
        "text": "Guidance, Navigation & Control Engineer",
        "categories": {
            "commitment": "Full Time",
            "department": "Engineering",
            "location": "Long Beach, California",
            "team": "GNC",
            "allLocations": ["Long Beach, California"],
        },
        "country": "US",
        "workplaceType": "onsite",
        "createdAt": CREATED_MS,
        "description": "<div><b>About</b></div><div>Design R&amp;D flight&nbsp;software.</div>",
        "descriptionPlain": "About\nDesign R&D flight software.",
        "lists": [
            {"text": "Responsibilities", "content": "<li>Kalman filters</li><li>Monte Carlo</li>"},
            {"text": "Requirements", "content": "<li>ITAR eligibility</li>"},
        ],
        "additional": "<div>Equal opportunity employer.</div>",
        "additionalPlain": "Equal opportunity employer.",
        "hostedUrl": f"https://jobs.lever.co/{SITE}/{job_id}",
        "applyUrl": f"https://jobs.lever.co/{SITE}/{job_id}/apply",
        "salaryRange": {"min": 120000, "max": 180000, "currency": "USD", "interval": "per-year-salary"},
    }
    posting.update(overrides)
    return posting


EXPECTED_DESCRIPTION = (
    "About\n\nDesign R&D flight software.\n\n"
    "Responsibilities\n- Kalman filters\n- Monte Carlo\n\n"
    "Requirements\n- ITAR eligibility\n\n"
    "Equal opportunity employer."
)


class FetchLeverJobsTests(unittest.TestCase):
    def test_successful_normalization(self):
        with mock.patch(URLOPEN, respond([lever_posting()])) as m:
            jobs = fetch_lever_jobs(SITE, company="Acme Aero")

        request, = m.call_args.args
        self.assertEqual(request.full_url, SITE_URL)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(m.call_args.kwargs["timeout"], 30)
        self.assertEqual(jobs, [{
            "company": "Acme Aero",
            "ats_provider": "lever",
            "ats_job_id": UUID,
            "title": "Guidance, Navigation & Control Engineer",
            "location": "Long Beach, California",
            "description": EXPECTED_DESCRIPTION,
            "application_url": f"https://jobs.lever.co/{SITE}/{UUID}/apply",
            "updated_at": None,
            "published_at": CREATED_ISO,
        }])

    def test_same_schema_as_greenhouse_and_ashby(self):
        from src import ashby, greenhouse
        with mock.patch(URLOPEN, respond([lever_posting()])):
            lever_job, = fetch_lever_jobs(SITE, company="A")
        gh_job = greenhouse.normalize_job({"id": 1, "title": "T"}, "A")
        ab_job = ashby.normalize_job({"id": "x", "title": "T"}, "A")
        self.assertEqual(set(lever_job), set(gh_job))
        self.assertEqual(set(lever_job), set(ab_job))

    def test_published_at_is_aware_utc_and_parses_for_cutoff(self):
        with mock.patch(URLOPEN, respond([lever_posting()])):
            job, = fetch_lever_jobs(SITE)
        parsed = posting_dates.parse_published_at(job["published_at"])
        self.assertEqual(parsed, datetime(2026, 8, 31, 18, 12, 19, 703000, tzinfo=timezone.utc))

    def test_application_url_falls_back_to_hosted_url(self):
        with mock.patch(URLOPEN, respond([lever_posting(applyUrl=None)])):
            job, = fetch_lever_jobs(SITE)
        self.assertEqual(job["application_url"], f"https://jobs.lever.co/{SITE}/{UUID}")

    def test_missing_optional_fields(self):
        with mock.patch(URLOPEN, respond([{"id": UUID, "text": "Avionics Technician"}])):
            job, = fetch_lever_jobs(SITE, company="Acme Aero")
        self.assertEqual(job["title"], "Avionics Technician")
        for key in ("location", "application_url", "updated_at", "published_at"):
            self.assertIsNone(job[key], key)
        self.assertEqual(job["description"], "")

    def test_wrong_typed_optional_fields(self):
        raw = lever_posting(categories="Long Beach", lists="nope", description=123, descriptionPlain=None,
                            additional=None, additionalPlain=None, applyUrl=7, hostedUrl=["x"],
                            createdAt="2026-08-31")
        with mock.patch(URLOPEN, respond([raw])):
            job, = fetch_lever_jobs(SITE)
        self.assertIsNone(job["location"])
        self.assertIsNone(job["application_url"])
        self.assertIsNone(job["published_at"])  # not an epoch -> unknown, never guessed
        self.assertEqual(job["description"], "")

    def test_plain_text_fallbacks(self):
        raw = lever_posting(description="", lists=[], additional="")
        with mock.patch(URLOPEN, respond([raw])):
            job, = fetch_lever_jobs(SITE)
        self.assertEqual(job["description"], "About\nDesign R&D flight software.\n\nEqual opportunity employer.")

    def test_malformed_list_sections_skipped(self):
        raw = lever_posting(description="", descriptionPlain="", additional="", additionalPlain="",
                            lists=["x", {"text": "Perks"}, {"content": "<li>401k</li>"}, {}])
        with mock.patch(URLOPEN, respond([raw])):
            job, = fetch_lever_jobs(SITE)
        self.assertEqual(job["description"], "Perks\n\n- 401k")

    def test_genuine_updated_at_is_used_if_present(self):
        with mock.patch(URLOPEN, respond([lever_posting(updatedAt=CREATED_MS + 86_400_000)])):
            job, = fetch_lever_jobs(SITE)
        self.assertEqual(job["updated_at"], "2026-09-01T18:12:19.703+00:00")
        self.assertEqual(job["published_at"], CREATED_ISO)  # never replaced by updatedAt

    def test_company_defaults_to_site(self):
        with mock.patch(URLOPEN, respond([lever_posting()])):
            job, = fetch_lever_jobs(SITE)
        self.assertEqual(job["company"], SITE)

    def test_empty_site_is_valid_empty_list(self):
        with mock.patch(URLOPEN, respond([])):
            self.assertEqual(fetch_lever_jobs(SITE), [])

    def test_missing_required_fields_and_malformed_postings(self):
        cases = {
            "no id": [{"text": "T"}],
            "blank id": [lever_posting(id="  ")],
            "numeric id": [lever_posting(id=5)],
            "no title": [lever_posting(text=None)],
            "blank title": [lever_posting(text="")],
            "not an object": ["posting"],
            "one bad among good": [lever_posting(), {"id": "x"}],
        }
        for label, payload in cases.items():
            with self.subTest(label), mock.patch(URLOPEN, respond(payload)):
                with self.assertRaises(LeverError):
                    fetch_lever_jobs(SITE)

    def test_invalid_responses(self):
        for label, body in {"object not list": {"jobs": []}, "null": "null", "not json": "<html>",
                            "bad utf8": b"\xff\xfe"}.items():
            with self.subTest(label), mock.patch(URLOPEN, respond(body)):
                with self.assertRaises(LeverError):
                    fetch_lever_jobs(SITE)

    def test_timeout(self):
        with mock.patch(URLOPEN, side_effect=socket.timeout("timed out")):
            with self.assertRaisesRegex(LeverError, "Timed out after 30s"):
                fetch_lever_jobs(SITE)

    def test_custom_timeout_passed_through(self):
        with mock.patch(URLOPEN, respond([])) as m:
            fetch_lever_jobs(SITE, timeout=5)
        self.assertEqual(m.call_args.kwargs["timeout"], 5)

    def test_http_errors(self):
        for code, reason in ((404, "Not Found"), (429, "Too Many Requests"), (500, "Server Error")):
            with self.subTest(code), mock.patch(URLOPEN, side_effect=http_error(code, reason)):
                with self.assertRaisesRegex(LeverError, f"HTTP {code}"):
                    fetch_lever_jobs(SITE)

    def test_network_error(self):
        with mock.patch(URLOPEN, side_effect=urllib.error.URLError("dns failure")):
            with self.assertRaisesRegex(LeverError, "Network error"):
                fetch_lever_jobs(SITE)

    def test_empty_site_rejected_without_request(self):
        with mock.patch(URLOPEN) as m:
            for site in ("", "   ", None):
                with self.subTest(site=site), self.assertRaises(LeverError):
                    fetch_lever_jobs(site)
        m.assert_not_called()

    def test_site_is_url_quoted(self):
        with mock.patch(URLOPEN, respond([])) as m:
            fetch_lever_jobs("a/b?c")
        self.assertEqual(m.call_args.args[0].full_url, "https://api.lever.co/v0/postings/a%2Fb%3Fc?mode=json")


class EpochConversionTests(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(epoch_ms_to_iso(CREATED_MS), CREATED_ISO)
        self.assertEqual(epoch_ms_to_iso(1_700_000_000_000.0), "2023-11-14T22:13:20.000+00:00")

    def test_rejected_values(self):
        for value in (None, True, False, "1788199939703", 1_788_199_939, -1, 0, 10**16, [], {}):
            with self.subTest(value=value):
                self.assertIsNone(epoch_ms_to_iso(value))


class LeverDispatchTests(unittest.TestCase):
    def test_lever_is_supported_alongside_existing_providers(self):
        self.assertEqual(providers.SUPPORTED_PROVIDERS, {"greenhouse", "ashby", "lever"})

    def test_dispatches_to_fetch_lever_jobs_only(self):
        with mock.patch("src.lever.fetch_lever_jobs", return_value=[{"x": 1}]) as lv, \
                mock.patch("src.greenhouse.fetch_greenhouse_jobs") as gh, \
                mock.patch("src.ashby.fetch_ashby_jobs") as ab:
            result = providers.fetch_jobs("lever", SITE, company="Acme Aero")
        self.assertEqual(result, [{"x": 1}])
        lv.assert_called_once_with(SITE, company="Acme Aero")
        gh.assert_not_called()
        ab.assert_not_called()

    def test_greenhouse_and_ashby_dispatch_unchanged(self):
        with mock.patch("src.greenhouse.fetch_greenhouse_jobs", return_value=[]) as gh, \
                mock.patch("src.ashby.fetch_ashby_jobs", return_value=[]) as ab, \
                mock.patch("src.lever.fetch_lever_jobs") as lv:
            providers.fetch_jobs("greenhouse", "muonspace", company="Muon Space")
            providers.fetch_jobs("ashby", "slingshot-aerospace", company="Slingshot Aerospace")
        gh.assert_called_once_with("muonspace", company="Muon Space")
        ab.assert_called_once_with("slingshot-aerospace", company="Slingshot Aerospace")
        lv.assert_not_called()

    def test_lever_errors_wrapped_as_collection_error(self):
        with mock.patch(URLOPEN, side_effect=http_error(404, "Not Found")):
            with self.assertRaisesRegex(CollectionError, "HTTP 404") as cm:
                providers.fetch_jobs("lever", SITE, company="Acme Aero")
        self.assertIsInstance(cm.exception.__cause__, LeverError)

    def test_config_accepts_lever(self):
        company, = parse_config({"companies": [{"name": "Acme Aero", "ats_provider": "Lever",
                                                "board_token": SITE}]})
        self.assertEqual(company.ats_provider, "lever")
        self.assertEqual(company.snapshot_filename, "acme_aero_jobs.json")

    def test_config_still_rejects_unknown_providers(self):
        with self.assertRaisesRegex(ConfigError, "unsupported ats_provider 'workday'"):
            parse_config({"companies": [{"name": "X", "ats_provider": "workday", "board_token": "x"}]})

    def test_same_token_on_different_providers_is_not_a_duplicate(self):
        companies = parse_config({"companies": [
            {"name": "A", "ats_provider": "greenhouse", "board_token": "acme"},
            {"name": "B", "ats_provider": "lever", "board_token": "acme"},
        ]})
        self.assertEqual(len(companies), 2)
        with self.assertRaisesRegex(ConfigError, "duplicate source"):
            parse_config({"companies": [
                {"name": "A", "ats_provider": "lever", "board_token": "acme"},
                {"name": "B", "ats_provider": "lever", "board_token": "ACME"},
            ]})


class LeverSqliteIntegrationTests(unittest.TestCase):
    """End-to-end through collect_jobs.main with a mixed Greenhouse + Ashby + Lever config."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.config, self.data_dir, self.db = tmp / "companies.json", tmp / "data", tmp / "jobs.db"
        self.config.write_text(json.dumps({"companies": [
            {"name": "Muon Space", "ats_provider": "greenhouse", "board_token": "muonspace"},
            {"name": "Slingshot Aerospace", "ats_provider": "ashby", "board_token": "slingshot-aerospace"},
            {"name": "Acme Aero", "ats_provider": "lever", "board_token": SITE},
        ]}), encoding="utf-8")
        self.lever_jobs = [lever_posting(),
                           lever_posting("0b7e7d0a-0000-4000-8000-000000000002", text="Propulsion Engineer")]
        self.lever_fails = False

    def tearDown(self):
        self._tmp.cleanup()

    def fake_urlopen(self, request, timeout=None):
        url = request.full_url
        if url.startswith("https://api.lever.co/"):
            self.assertEqual(url, SITE_URL)
            if self.lever_fails:
                raise http_error(503, "Service Unavailable")
            return FakeResponse(json.dumps(self.lever_jobs))
        if url.startswith("https://api.ashbyhq.com/"):
            return FakeResponse(json.dumps({"jobs": [{"id": "ab-1", "title": "RF Engineer"}]}))
        if url.startswith("https://boards-api.greenhouse.io/"):
            return FakeResponse(json.dumps({"jobs": [{"id": 1, "title": "GNC Engineer"}]}))
        raise AssertionError(f"unexpected URL {url}")

    def run_main(self, *args, db=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["--config", str(self.config), "--data-dir", str(self.data_dir)]
        if db:
            argv += ["--db", str(self.db)]
        with mock.patch(URLOPEN, side_effect=self.fake_urlopen), redirect_stdout(stdout), redirect_stderr(stderr):
            code = collect_jobs.main([*argv, *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def rows(self, sql, params=()):
        conn = database.connect(self.db)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def test_snapshot_and_database_rows(self):
        code, out, err = self.run_main("--all")
        self.assertEqual(code, 0, err)
        self.assertIn("Collected 2 Acme Aero jobs", out)
        self.assertIn("Summary: 3/3 companies succeeded", out)

        snap = json.loads((self.data_dir / "acme_aero_jobs.json").read_text(encoding="utf-8"))
        self.assertEqual(snap["source"], {"ats_provider": "lever", "board_token": SITE, "company": "Acme Aero"})
        self.assertEqual(snap["job_count"], 2)
        self.assertEqual(snap["jobs"][0]["published_at"], CREATED_ISO)

        company, = self.rows("SELECT * FROM companies WHERE ats_provider = 'lever'")
        self.assertEqual((company["name"], company["board_token"]), ("Acme Aero", SITE))
        job, = self.rows("SELECT * FROM jobs WHERE ats_job_id = ?", (UUID,))
        self.assertEqual(job["company_id"], company["id"])
        self.assertEqual(job["title"], "Guidance, Navigation & Control Engineer")
        self.assertEqual(job["location"], "Long Beach, California")
        self.assertEqual(job["source_published_at"], CREATED_ISO)
        self.assertIsNone(job["source_updated_at"])
        self.assertEqual(job["application_url"], f"https://jobs.lever.co/{SITE}/{UUID}/apply")

    def test_rerun_deduplicates_and_detects_changes(self):
        self.run_main("--company", "Acme Aero")
        first_seen = self.rows("SELECT first_seen_at FROM jobs WHERE ats_job_id = ?", (UUID,))[0][0]

        code, out, _ = self.run_main("--company", "Acme Aero")
        self.assertEqual(code, 0)
        self.assertIn("0 inserted, 0 updated, 2 unchanged", out)
        self.assertEqual(len(self.rows("SELECT id FROM jobs")), 2)

        self.lever_jobs[0] = lever_posting(lists=[], additional="", additionalPlain="",
                                           description="<p>Updated description.</p>")
        code, out, _ = self.run_main("--company", SITE)  # select by site identifier
        self.assertEqual(code, 0)
        self.assertIn("0 inserted, 1 updated, 1 unchanged", out)
        job, = self.rows("SELECT * FROM jobs WHERE ats_job_id = ?", (UUID,))
        self.assertEqual(job["description"], "Updated description.")
        self.assertEqual(job["first_seen_at"], first_seen)

    def test_lever_failure_isolated_from_other_providers(self):
        self.run_main("--all")
        before = (self.data_dir / "acme_aero_jobs.json").read_text(encoding="utf-8")
        self.lever_fails = True

        code, out, err = self.run_main("--all")
        self.assertEqual(code, 1)
        self.assertIn("ERROR: Acme Aero: HTTP 503", err)
        self.assertIn("Summary: 2/3 companies succeeded; failed: Acme Aero", out)
        self.assertIn("Collected 1 Muon Space jobs", out)
        self.assertIn("Collected 1 Slingshot Aerospace jobs", out)
        self.assertEqual((self.data_dir / "acme_aero_jobs.json").read_text(encoding="utf-8"), before)
        self.assertEqual(len(self.rows("SELECT id FROM jobs WHERE ats_job_id = ?", (UUID,))), 1)

    def test_snapshot_only_run_without_db(self):
        code, out, err = self.run_main("--company", "Acme Aero", db=False)
        self.assertEqual(code, 0, err)
        self.assertTrue((self.data_dir / "acme_aero_jobs.json").exists())
        self.assertFalse(self.db.exists())
        dates = posting_dates.load_snapshot_published_dates(self.data_dir)
        self.assertEqual(dates[posting_dates.snapshot_key("lever", "Acme Aero", UUID)], CREATED_ISO)


class LeverReportCutoffTests(unittest.TestCase):
    """Lever createdAt feeds the existing 30-day cutoff like any other publication date."""

    def test_classification(self):
        now = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
        recent_ms = int((now - timedelta(days=3)).timestamp() * 1000)
        boundary_ms = int((now - timedelta(days=30)).timestamp() * 1000)
        old_ms = int((now - timedelta(days=30, seconds=1)).timestamp() * 1000)
        cases = {recent_ms: posting_dates.RECENT, boundary_ms: posting_dates.RECENT,
                 old_ms: posting_dates.TOO_OLD, None: posting_dates.UNKNOWN}
        for created, expected in cases.items():
            with self.subTest(created=created):
                parsed = posting_dates.parse_published_at(epoch_ms_to_iso(created))
                self.assertEqual(posting_dates.classify_age(parsed, now), expected)


if __name__ == "__main__":
    unittest.main()
