"""Unit tests for the Greenhouse collector. No network access required."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src import collect_jobs, greenhouse
from src.greenhouse import GreenhouseError, fetch_greenhouse_jobs, html_to_text

URLOPEN = "src.greenhouse.urllib.request.urlopen"


class FakeResponse:
    """Minimal stand-in for the object returned by urlopen()."""

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


FULL_JOB = {
    "id": 4012345,
    "title": "  Spacecraft GNC Engineer ",
    "location": {"name": "Mountain View, CA"},
    "absolute_url": "https://job-boards.greenhouse.io/muonspace/jobs/4012345",
    "updated_at": "2026-11-01T10:15:00-04:00",
    "first_published": "2026-10-20T09:00:00-04:00",
    "company_name": "Muon Space",
    # Greenhouse returns content entity-escaped HTML.
    "content": "&lt;h2&gt;About&lt;/h2&gt;&lt;p&gt;Design &amp;amp; test&amp;nbsp;GNC.&lt;/p&gt;"
               "&lt;ul&gt;&lt;li&gt;MATLAB&lt;/li&gt;&lt;li&gt;C++&lt;/li&gt;&lt;/ul&gt;",
}


class FetchGreenhouseJobsTests(unittest.TestCase):
    def test_successful_collection(self):
        with mock.patch(URLOPEN, respond({"jobs": [FULL_JOB], "meta": {"total": 1}})) as m:
            jobs = fetch_greenhouse_jobs("muonspace", company="Muon Space")

        request, = m.call_args.args
        self.assertEqual(
            request.full_url, "https://boards-api.greenhouse.io/v1/boards/muonspace/jobs?content=true"
        )
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(m.call_args.kwargs["timeout"], 30)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            jobs[0],
            {
                "company": "Muon Space",
                "ats_provider": "greenhouse",
                "ats_job_id": "4012345",
                "title": "Spacecraft GNC Engineer",
                "location": "Mountain View, CA",
                "description": "About\n\nDesign & test GNC.\n\n- MATLAB\n- C++",
                "application_url": "https://job-boards.greenhouse.io/muonspace/jobs/4012345",
                "updated_at": "2026-11-01T10:15:00-04:00",
                "published_at": "2026-10-20T09:00:00-04:00",  # first_published, not updated_at
            },
        )

    def test_company_defaults_to_api_company_name(self):
        with mock.patch(URLOPEN, respond({"jobs": [FULL_JOB]})):
            jobs = fetch_greenhouse_jobs("muonspace")
        self.assertEqual(jobs[0]["company"], "Muon Space")

    def test_missing_optional_fields(self):
        with mock.patch(URLOPEN, respond({"jobs": [{"id": 7, "title": "Technician"}]})):
            job, = fetch_greenhouse_jobs("muonspace", company="Muon Space")
        self.assertEqual(job["ats_job_id"], "7")
        self.assertIsNone(job["location"])
        self.assertIsNone(job["application_url"])
        self.assertIsNone(job["updated_at"])  # never invented
        self.assertIsNone(job["published_at"])  # never invented
        self.assertEqual(job["description"], "")

    def test_null_and_wrong_typed_optional_fields(self):
        raw = {"id": 8, "title": "Tech", "location": None, "content": None,
               "updated_at": None, "absolute_url": 123}
        with mock.patch(URLOPEN, respond({"jobs": [raw]})):
            job, = fetch_greenhouse_jobs("muonspace", company="Muon Space")
        self.assertIsNone(job["location"])
        self.assertIsNone(job["updated_at"])
        self.assertIsNone(job["application_url"])
        self.assertEqual(job["description"], "")

    def test_invalid_json(self):
        with mock.patch(URLOPEN, respond("<html>Service Unavailable</html>")):
            with self.assertRaisesRegex(GreenhouseError, "Invalid JSON"):
                fetch_greenhouse_jobs("muonspace")

    def test_http_failure(self):
        err = urllib.error.HTTPError(
            "https://boards-api.greenhouse.io/x", 404, "Not Found", hdrs=None, fp=io.BytesIO(b"")
        )
        with mock.patch(URLOPEN, side_effect=err):
            with self.assertRaisesRegex(GreenhouseError, "HTTP 404"):
                fetch_greenhouse_jobs("doesnotexist")

    def test_network_failure(self):
        with mock.patch(URLOPEN, side_effect=urllib.error.URLError("Name or service not known")):
            with self.assertRaisesRegex(GreenhouseError, "Network error"):
                fetch_greenhouse_jobs("muonspace")

    def test_timeout(self):
        with mock.patch(URLOPEN, side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(GreenhouseError, "Timed out"):
                fetch_greenhouse_jobs("muonspace")

    def test_malformed_response_shapes(self):
        for payload in ([], {"meta": {}}, {"jobs": "nope"}, {"jobs": [None]},
                        {"jobs": [{"title": "No id"}]}, {"jobs": [{"id": 1}]}):
            with self.subTest(payload=payload), mock.patch(URLOPEN, respond(payload)):
                with self.assertRaises(GreenhouseError):
                    fetch_greenhouse_jobs("muonspace", company="Muon Space")

    def test_empty_board_token_rejected(self):
        with self.assertRaises(GreenhouseError):
            fetch_greenhouse_jobs("  ")


class HtmlToTextTests(unittest.TestCase):
    def test_strips_scripts_and_collapses_whitespace(self):
        raw = "<div>Hello <b>world</b></div><script>evil()</script><p>\n\n  Line   two </p><br/>"
        self.assertEqual(html_to_text(raw), "Hello world\n\nLine two")

    def test_empty(self):
        self.assertEqual(html_to_text(None), "")
        self.assertEqual(html_to_text(""), "")


def write_temp_config(directory: Path) -> Path:
    """A one-company config so these tests never depend on the personal config/companies.json."""
    path = directory / "companies.json"
    path.write_text(json.dumps({"companies": [
        {"name": "Muon Space", "ats_provider": "greenhouse", "board_token": "muonspace"}]}), encoding="utf-8")
    return path


class CollectJobsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "data" / "muon_space_jobs.json"
        self.config = write_temp_config(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = collect_jobs.main(["--config", str(self.config), "--output", str(self.out), *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_writes_snapshot_with_collected_at(self):
        with mock.patch(URLOPEN, respond({"jobs": [FULL_JOB]})):
            code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("Collected 1 Muon Space jobs", out)
        snap = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertIn("collected_at", snap)
        self.assertEqual(snap["job_count"], 1)
        self.assertEqual(snap["jobs"][0]["ats_job_id"], "4012345")
        # No temp files left behind.
        self.assertEqual([p.name for p in self.out.parent.iterdir()], [self.out.name])

    def test_duplicate_ids_are_removed(self):
        dup = dict(FULL_JOB, title="Same id, later copy")
        other = dict(FULL_JOB, id=999)
        with mock.patch(URLOPEN, respond({"jobs": [FULL_JOB, dup, other]})):
            code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("1 duplicates removed", out)
        snap = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual([j["ats_job_id"] for j in snap["jobs"]], ["4012345", "999"])
        self.assertEqual(snap["jobs"][0]["title"], "Spacecraft GNC Engineer")

    def test_dedupe_key_includes_company_and_provider(self):
        base = {"ats_provider": "greenhouse", "company": "A", "ats_job_id": "1"}
        jobs = [base, dict(base), dict(base, company="B"), dict(base, ats_provider="lever")]
        self.assertEqual(len(collect_jobs.dedupe_jobs(jobs)), 3)

    def _seed_existing_snapshot(self) -> str:
        self.out.parent.mkdir(parents=True)
        original = json.dumps({"collected_at": "2026-11-01T00:00:00+00:00", "jobs": [{"id": "old"}]})
        self.out.write_text(original, encoding="utf-8")
        return original

    def test_failures_do_not_overwrite_existing_snapshot(self):
        http_err = urllib.error.HTTPError("u", 503, "Service Unavailable", hdrs=None, fp=io.BytesIO(b""))
        cases = {
            "invalid_json": respond("not json"),
            "http_error": mock.Mock(side_effect=http_err),
            "malformed": respond({"jobs": [{"title": "missing id"}]}),
            "empty": respond({"jobs": []}),
        }
        original = self._seed_existing_snapshot()
        for name, fake in cases.items():
            with self.subTest(name), mock.patch(URLOPEN, fake):
                code, out, err = self.run_main()
                self.assertNotEqual(code, 0)
                self.assertIn("ERROR", err)
                self.assertEqual(out, "")
                self.assertEqual(self.out.read_text(encoding="utf-8"), original)

    def test_allow_empty_writes_zero_job_snapshot(self):
        with mock.patch(URLOPEN, respond({"jobs": []})):
            code, _, _ = self.run_main("--allow-empty")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.out.read_text(encoding="utf-8"))["job_count"], 0)

    def test_write_failure_leaves_existing_snapshot(self):
        original = self._seed_existing_snapshot()
        with mock.patch(URLOPEN, respond({"jobs": [FULL_JOB]})), \
                mock.patch.object(collect_jobs.os, "replace", side_effect=OSError("disk full")):
            code, _, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("disk full", err)
        self.assertEqual(self.out.read_text(encoding="utf-8"), original)
        self.assertEqual([p.name for p in self.out.parent.iterdir()], [self.out.name])


if __name__ == "__main__":
    unittest.main()
