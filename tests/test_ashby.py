"""Tests for the Ashby collector, provider dispatch and SQLite integration. All HTTP is mocked."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src import ashby, collect_jobs, database, providers
from src.ashby import AshbyError, fetch_ashby_jobs
from src.providers import CollectionError

URLOPEN = "src.ats_common.urllib.request.urlopen"
SLINGSHOT_URL = "https://api.ashbyhq.com/posting-api/job-board/slingshot-aerospace"
UUID = "18b03b0e-f24b-44a9-a446-34cea181c22f"


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
    return urllib.error.HTTPError(SLINGSHOT_URL, code, reason, hdrs=None, fp=io.BytesIO(b""))


def ashby_posting(job_id: str = UUID, **overrides) -> dict:
    """Shaped like a live Slingshot Aerospace posting."""
    posting = {
        "id": job_id,
        "title": "Manager, Capture & Proposal",
        "location": "United States",
        "department": "Business Development",
        "team": "Capture",
        "isListed": True,
        "isRemote": True,
        "employmentType": "FullTime",
        "jobUrl": f"https://jobs.ashbyhq.com/slingshot-aerospace/{job_id}",
        "applyUrl": f"https://jobs.ashbyhq.com/slingshot-aerospace/{job_id}/application",
        "publishedAt": "2026-09-15T13:57:17.235+00:00",
        "descriptionHtml": "<h2>About</h2><p>Lead R&amp;D capture&nbsp;efforts.</p>"
                           "<ul><li>Proposals</li><li>Pricing</li></ul>",
        "descriptionPlain": "About\nLead R&D capture efforts.",
    }
    posting.update(overrides)
    return posting


class FetchAshbyJobsTests(unittest.TestCase):
    def test_successful_normalization(self):
        with mock.patch(URLOPEN, respond({"apiVersion": "1", "jobs": [ashby_posting()]})) as m:
            jobs = fetch_ashby_jobs("slingshot-aerospace", company="Slingshot Aerospace")

        request, = m.call_args.args
        self.assertEqual(request.full_url, SLINGSHOT_URL)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(m.call_args.kwargs["timeout"], 30)
        self.assertEqual(jobs, [{
            "company": "Slingshot Aerospace",
            "ats_provider": "ashby",
            "ats_job_id": UUID,
            "title": "Manager, Capture & Proposal",
            "location": "United States",
            "description": "About\n\nLead R&D capture efforts.\n\n- Proposals\n- Pricing",
            "application_url": f"https://jobs.ashbyhq.com/slingshot-aerospace/{UUID}/application",
            "updated_at": None,  # publishedAt is NOT an update timestamp
        }])

    def test_same_schema_as_greenhouse(self):
        from src.greenhouse import normalize_job as gh_normalize
        gh = gh_normalize({"id": 1, "title": "t"}, "X")
        with mock.patch(URLOPEN, respond({"jobs": [ashby_posting()]})):
            ab, = fetch_ashby_jobs("slingshot-aerospace", company="X")
        self.assertEqual(list(ab), list(gh))

    def test_genuine_updated_at_is_used(self):
        posting = ashby_posting(updatedAt="2026-10-01T08:00:00.000+00:00")
        with mock.patch(URLOPEN, respond({"jobs": [posting]})):
            job, = fetch_ashby_jobs("slingshot-aerospace")
        self.assertEqual(job["updated_at"], "2026-10-01T08:00:00.000+00:00")

    def test_application_url_falls_back_to_job_url(self):
        for apply_url in (None, "", "   "):
            with self.subTest(applyUrl=apply_url):
                with mock.patch(URLOPEN, respond({"jobs": [ashby_posting(applyUrl=apply_url)]})):
                    job, = fetch_ashby_jobs("slingshot-aerospace")
                self.assertEqual(job["application_url"], f"https://jobs.ashbyhq.com/slingshot-aerospace/{UUID}")

    def test_missing_optional_fields(self):
        with mock.patch(URLOPEN, respond({"jobs": [{"id": UUID, "title": "Technician"}]})):
            job, = fetch_ashby_jobs("slingshot-aerospace", company="Slingshot Aerospace")
        self.assertEqual(job["ats_job_id"], UUID)
        self.assertIsNone(job["location"])
        self.assertIsNone(job["application_url"])
        self.assertIsNone(job["updated_at"])
        self.assertEqual(job["description"], "")

    def test_wrong_typed_optional_fields(self):
        posting = {"id": UUID, "title": "Tech", "location": {"city": "Denver"}, "applyUrl": 5,
                   "jobUrl": None, "descriptionHtml": None, "updatedAt": 123}
        with mock.patch(URLOPEN, respond({"jobs": [posting]})):
            job, = fetch_ashby_jobs("slingshot-aerospace")
        self.assertIsNone(job["location"])
        self.assertIsNone(job["application_url"])
        self.assertIsNone(job["updated_at"])
        self.assertEqual(job["description"], "")

    def test_description_plain_fallback(self):
        posting = ashby_posting(descriptionHtml="", descriptionPlain="  Plain text only. ")
        with mock.patch(URLOPEN, respond({"jobs": [posting]})):
            job, = fetch_ashby_jobs("slingshot-aerospace")
        self.assertEqual(job["description"], "Plain text only.")

    def test_raw_html_is_not_double_unescaped(self):
        posting = ashby_posting(descriptionHtml="<p>Use &lt;script&gt; tags &amp;amp; entities</p>")
        with mock.patch(URLOPEN, respond({"jobs": [posting]})):
            job, = fetch_ashby_jobs("slingshot-aerospace")
        self.assertEqual(job["description"], "Use <script> tags &amp; entities")

    def test_company_defaults_to_board_token(self):
        with mock.patch(URLOPEN, respond({"jobs": [ashby_posting()]})):
            job, = fetch_ashby_jobs("slingshot-aerospace")
        self.assertEqual(job["company"], "slingshot-aerospace")

    def test_missing_required_fields_and_malformed_postings(self):
        cases = {
            "missing id": {k: v for k, v in ashby_posting().items() if k != "id"},
            "empty id": ashby_posting(job_id="  "),
            "numeric id": ashby_posting(job_id=12345),
            "missing title": {k: v for k, v in ashby_posting().items() if k != "title"},
            "blank title": ashby_posting(title=" "),
            "posting not object": "not-a-posting",
            "null posting": None,
        }
        for label, posting in cases.items():
            payload = {"jobs": [ashby_posting("ok-1"), posting]}
            with self.subTest(label), mock.patch(URLOPEN, respond(payload)):
                with self.assertRaises(AshbyError):
                    fetch_ashby_jobs("slingshot-aerospace")

    def test_invalid_responses(self):
        for payload in ("<html>Service Unavailable</html>", b"\xff\xfe\x00"):
            with self.subTest(payload=payload), mock.patch(URLOPEN, respond(payload)):
                with self.assertRaisesRegex(AshbyError, "Invalid JSON"):
                    fetch_ashby_jobs("slingshot-aerospace")
        for payload in ([], {}, {"jobs": None}, {"jobs": "x"}, {"results": []}):
            with self.subTest(payload=payload), mock.patch(URLOPEN, respond(payload)):
                with self.assertRaisesRegex(AshbyError, "Malformed response"):
                    fetch_ashby_jobs("slingshot-aerospace")

    def test_timeout(self):
        with mock.patch(URLOPEN, side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(AshbyError, "Timed out after 30s"):
                fetch_ashby_jobs("slingshot-aerospace")

    def test_custom_timeout_passed_through(self):
        with mock.patch(URLOPEN, respond({"jobs": []})) as m:
            fetch_ashby_jobs("slingshot-aerospace", timeout=5)
        self.assertEqual(m.call_args.kwargs["timeout"], 5)

    def test_http_errors(self):
        for code, reason in ((404, "Not Found"), (429, "Too Many Requests"), (500, "Server Error")):
            with self.subTest(code=code), mock.patch(URLOPEN, side_effect=http_error(code, reason)):
                with self.assertRaisesRegex(AshbyError, f"HTTP {code}"):
                    fetch_ashby_jobs("slingshot-aerospace")

    def test_network_error(self):
        with mock.patch(URLOPEN, side_effect=urllib.error.URLError("getaddrinfo failed")):
            with self.assertRaisesRegex(AshbyError, "Network error"):
                fetch_ashby_jobs("slingshot-aerospace")

    def test_empty_board_token_rejected(self):
        with self.assertRaises(AshbyError):
            fetch_ashby_jobs("")

    def test_board_token_is_url_quoted(self):
        self.assertEqual(ashby.build_url("a b/c"), "https://api.ashbyhq.com/posting-api/job-board/a%20b%2Fc")


class AshbyDispatchTests(unittest.TestCase):
    def test_ashby_is_supported(self):
        self.assertEqual(providers.SUPPORTED_PROVIDERS, {"greenhouse", "ashby"})

    def test_dispatches_to_fetch_ashby_jobs(self):
        with mock.patch("src.ashby.fetch_ashby_jobs", return_value=[{"x": 1}]) as ab, \
                mock.patch("src.greenhouse.fetch_greenhouse_jobs") as gh:
            result = providers.fetch_jobs("ashby", "slingshot-aerospace", company="Slingshot Aerospace")
        self.assertEqual(result, [{"x": 1}])
        ab.assert_called_once_with("slingshot-aerospace", company="Slingshot Aerospace")
        gh.assert_not_called()

    def test_greenhouse_dispatch_unchanged(self):
        with mock.patch("src.greenhouse.fetch_greenhouse_jobs", return_value=[]) as gh, \
                mock.patch("src.ashby.fetch_ashby_jobs") as ab:
            providers.fetch_jobs("greenhouse", "muonspace", company="Muon Space")
        gh.assert_called_once_with("muonspace", company="Muon Space")
        ab.assert_not_called()

    def test_ashby_errors_wrapped_as_collection_error(self):
        with mock.patch(URLOPEN, side_effect=http_error(503, "Unavailable")):
            with self.assertRaisesRegex(CollectionError, "HTTP 503") as cm:
                providers.fetch_jobs("ashby", "slingshot-aerospace", company="Slingshot Aerospace")
        self.assertIsInstance(cm.exception.__cause__, AshbyError)

    def test_config_accepts_ashby(self):
        from src.company_config import parse_config
        company, = parse_config({"companies": [{"name": "Slingshot Aerospace", "ats_provider": "Ashby",
                                                "board_token": "slingshot-aerospace"}]})
        self.assertEqual(company.ats_provider, "ashby")
        self.assertEqual(company.snapshot_filename, "slingshot_aerospace_jobs.json")


class AshbySqliteIntegrationTests(unittest.TestCase):
    """End-to-end through collect_jobs.main with a mixed Greenhouse + Ashby config."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.config, self.data_dir, self.db = tmp / "companies.json", tmp / "data", tmp / "jobs.db"
        self.config.write_text(json.dumps({"companies": [
            {"name": "Muon Space", "ats_provider": "greenhouse", "board_token": "muonspace"},
            {"name": "Slingshot Aerospace", "ats_provider": "ashby", "board_token": "slingshot-aerospace"},
        ]}), encoding="utf-8")
        self.ashby_jobs = [ashby_posting(), ashby_posting("2c7e7d0a-0000-4000-8000-000000000002", title="RF Engineer")]
        self.ashby_fails = False

    def tearDown(self):
        self._tmp.cleanup()

    def fake_urlopen(self, request, timeout=None):
        url = request.full_url
        if url.startswith("https://api.ashbyhq.com/"):
            if self.ashby_fails:
                raise http_error(500, "Server Error")
            return FakeResponse(json.dumps({"jobs": self.ashby_jobs}))
        if url.startswith("https://boards-api.greenhouse.io/"):
            return FakeResponse(json.dumps({"jobs": [{"id": 1, "title": "GNC Engineer"}]}))
        raise AssertionError(f"unexpected URL {url}")

    def run_main(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch(URLOPEN, side_effect=self.fake_urlopen), redirect_stdout(stdout), redirect_stderr(stderr):
            code = collect_jobs.main(["--config", str(self.config), "--data-dir", str(self.data_dir),
                                      "--db", str(self.db), *args])
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
        self.assertIn("Collected 2 Slingshot Aerospace jobs", out)
        self.assertIn("Summary: 2/2 companies succeeded", out)

        snap = json.loads((self.data_dir / "slingshot_aerospace_jobs.json").read_text(encoding="utf-8"))
        self.assertEqual(snap["source"], {"ats_provider": "ashby", "board_token": "slingshot-aerospace",
                                          "company": "Slingshot Aerospace"})
        self.assertEqual(snap["job_count"], 2)
        self.assertIsNone(snap["jobs"][0]["updated_at"])

        company, = self.rows("SELECT * FROM companies WHERE ats_provider = 'ashby'")
        self.assertEqual((company["name"], company["board_token"]), ("Slingshot Aerospace", "slingshot-aerospace"))
        job, = self.rows("SELECT * FROM jobs WHERE ats_job_id = ?", (UUID,))
        self.assertEqual(job["company_id"], company["id"])
        self.assertEqual(job["title"], "Manager, Capture & Proposal")
        self.assertIsNone(job["source_updated_at"])
        self.assertEqual(job["application_url"], f"https://jobs.ashbyhq.com/slingshot-aerospace/{UUID}/application")
        self.assertEqual(job["first_seen_at"], snap["collected_at"])

    def test_rerun_detects_unchanged_and_changed_postings(self):
        self.run_main("--company", "Slingshot Aerospace")
        first_seen = self.rows("SELECT first_seen_at FROM jobs WHERE ats_job_id = ?", (UUID,))[0][0]

        code, out, _ = self.run_main("--company", "Slingshot Aerospace")
        self.assertEqual(code, 0)
        self.assertIn("0 inserted, 0 updated, 2 unchanged", out)

        self.ashby_jobs[0] = ashby_posting(descriptionHtml="<p>Updated description.</p>")
        code, out, _ = self.run_main("--company", "slingshot-aerospace")
        self.assertEqual(code, 0)
        self.assertIn("0 inserted, 1 updated, 1 unchanged", out)
        job, = self.rows("SELECT * FROM jobs WHERE ats_job_id = ?", (UUID,))
        self.assertEqual(job["description"], "Updated description.")
        self.assertEqual(job["first_seen_at"], first_seen)

    def test_ashby_failure_isolated_from_greenhouse(self):
        self.run_main("--all")
        before = (self.data_dir / "slingshot_aerospace_jobs.json").read_text(encoding="utf-8")
        self.ashby_fails = True

        code, out, err = self.run_main("--all")
        self.assertEqual(code, 1)
        self.assertIn("ERROR: Slingshot Aerospace: HTTP 500", err)
        self.assertIn("Summary: 1/2 companies succeeded; failed: Slingshot Aerospace", out)
        self.assertIn("Collected 1 Muon Space jobs", out)
        self.assertEqual((self.data_dir / "slingshot_aerospace_jobs.json").read_text(encoding="utf-8"), before)
        self.assertEqual(len(self.rows("SELECT id FROM jobs WHERE ats_job_id = ?", (UUID,))), 1)


if __name__ == "__main__":
    unittest.main()
