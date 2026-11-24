"""Tests for config-driven multi-company collection. All HTTP is mocked."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src import collect_jobs, database, providers
from src.company_config import CompanyConfig, ConfigError, load_config, parse_config, select_companies
from src.providers import CollectionError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
URLOPEN = "src.greenhouse.urllib.request.urlopen"

COMPANIES = [
    {"name": "Muon Space", "ats_provider": "greenhouse", "board_token": "muonspace", "enabled": True},
    {"name": "Planet Labs", "ats_provider": "greenhouse", "board_token": "planetlabs", "enabled": True},
    {"name": "AST SpaceMobile", "ats_provider": "greenhouse", "board_token": "astspacemobile", "enabled": True},
    {"name": "LeoLabs", "ats_provider": "greenhouse", "board_token": "leolabsinc", "enabled": True},
]


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def gh_job(job_id: int, title: str = "Engineer", **extra) -> dict:
    job = {"id": job_id, "title": title, "location": {"name": "Remote"},
           "absolute_url": f"https://example.test/jobs/{job_id}",
           "updated_at": "2026-11-01T00:00:00Z", "content": "&lt;p&gt;Do things.&lt;/p&gt;"}
    job.update(extra)
    return job


class FakeGreenhouse:
    """Routes mocked urlopen calls by board token; tokens in `fail` return HTTP 500."""

    def __init__(self, boards: dict[str, list[dict]], fail: set[str] = frozenset()):
        self.boards, self.fail, self.requested = boards, set(fail), []

    def __call__(self, request, timeout=None):
        token = request.full_url.split("/boards/")[1].split("/")[0]
        self.requested.append(token)
        if token in self.fail:
            raise urllib.error.HTTPError(request.full_url, 500, "Server Error", hdrs=None, fp=io.BytesIO(b""))
        return FakeResponse(json.dumps({"jobs": self.boards.get(token, [])}).encode())


DEFAULT_BOARDS = {
    "muonspace": [gh_job(1, "GNC Engineer"), gh_job(2)],
    "planetlabs": [gh_job(10), gh_job(11), gh_job(1, "Same id as Muon job, different company")],
    "astspacemobile": [gh_job(20)],
    "leolabsinc": [gh_job(30), gh_job(30)],  # duplicate within a board
}


class ConfigValidationTests(unittest.TestCase):
    def test_example_config_template_is_valid(self):
        # The public template ships in git; the personal config/companies.json is git-ignored.
        companies = load_config(PROJECT_ROOT / "config" / "companies.example.json")
        self.assertEqual({c.ats_provider for c in companies}, set(providers.SUPPORTED_PROVIDERS))
        self.assertFalse(any(c.enabled for c in companies))      # placeholders are never collected
        with self.assertRaisesRegex(ConfigError, "No enabled companies"):
            select_companies(companies, all_enabled=True)

    def test_local_config_is_valid_if_present(self):
        path = PROJECT_ROOT / "config" / "companies.json"
        if not path.exists():
            self.skipTest("config/companies.json not present (personal, git-ignored)")
        self.assertTrue(load_config(path))   # validates providers, duplicates, filename collisions

    def test_personal_config_is_git_ignored(self):
        text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("config/companies.json", text)
        self.assertNotIn("config/companies.example.json", text)

    def test_missing_config_points_to_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = collect_jobs.main(["--config", str(Path(tmp) / "companies.json"), "--all"])
        self.assertEqual(code, 2)
        self.assertIn("Cannot read config", err.getvalue())
        self.assertIn("Hint: create it from the template: copy", err.getvalue())
        self.assertIn("companies.example.json", err.getvalue())

    def test_invalid_configs_rejected(self):
        ok = dict(COMPANIES[0])
        cases = {
            "not an object": [],
            "missing companies": {},
            "companies not a list": {"companies": {}},
            "empty list": {"companies": []},
            "entry not object": {"companies": ["Muon"]},
            "missing name": {"companies": [{**ok, "name": ""}]},
            "missing token": {"companies": [{k: v for k, v in ok.items() if k != "board_token"}]},
            "missing provider": {"companies": [{**ok, "ats_provider": None}]},
            "enabled not bool": {"companies": [{**ok, "enabled": "yes"}]},
            "unsupported provider": {"companies": [{**ok, "ats_provider": "workday"}]},
            "duplicate source": {"companies": [ok, {**ok, "name": "Other"}]},
            "duplicate name": {"companies": [ok, {**ok, "board_token": "other"}]},
            "filename collision": {"companies": [ok, {**ok, "name": "Muon-Space", "board_token": "x"}]},
        }
        for label, data in cases.items():
            with self.subTest(label), self.assertRaises(ConfigError):
                parse_config(data)

    def test_disabled_unsupported_provider_allowed(self):
        companies = parse_config({"companies": [
            COMPANIES[0], {"name": "Future Co", "ats_provider": "workday", "board_token": "f", "enabled": False}]})
        self.assertEqual(len(companies), 2)
        self.assertEqual([c.name for c in select_companies(companies, all_enabled=True)], ["Muon Space"])

    def test_enabled_defaults_to_true(self):
        entry = {k: v for k, v in COMPANIES[0].items() if k != "enabled"}
        self.assertTrue(parse_config({"companies": [entry]})[0].enabled)

    def test_unreadable_and_invalid_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ConfigError, "Cannot read"):
                load_config(Path(tmp) / "missing.json")
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "Invalid JSON"):
                load_config(bad)

    def test_select_by_name_or_token(self):
        companies = parse_config({"companies": COMPANIES})
        self.assertEqual([c.name for c in select_companies(companies, ["muon space", "LEOLABSINC"])],
                         ["Muon Space", "LeoLabs"])
        with self.assertRaisesRegex(ConfigError, "Unknown company"):
            select_companies(companies, ["SpaceX"])
        disabled = parse_config({"companies": [{**COMPANIES[0], "enabled": False}]})
        with self.assertRaisesRegex(ConfigError, "disabled"):
            select_companies(disabled, ["Muon Space"])
        with self.assertRaisesRegex(ConfigError, "No enabled"):
            select_companies(disabled, all_enabled=True)


class DispatcherTests(unittest.TestCase):
    def test_unsupported_provider(self):
        with self.assertRaisesRegex(CollectionError, "Unsupported ATS provider 'workday'"):
            providers.fetch_jobs("workday", "token", company="X")

    def test_greenhouse_delegates_to_fetch_greenhouse_jobs(self):
        with mock.patch("src.greenhouse.fetch_greenhouse_jobs", return_value=[{"x": 1}]) as m:
            self.assertEqual(providers.fetch_jobs("greenhouse", "muonspace", company="Muon Space"), [{"x": 1}])
        m.assert_called_once_with("muonspace", company="Muon Space")

    def test_provider_errors_wrapped(self):
        with mock.patch(URLOPEN, FakeGreenhouse({}, fail={"muonspace"})):
            with self.assertRaisesRegex(CollectionError, "HTTP 500"):
                providers.fetch_jobs("greenhouse", "muonspace", company="Muon Space")


class MultiCompanyCollectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.config, self.data_dir, self.db = tmp / "companies.json", tmp / "data", tmp / "data" / "jobs.db"
        self.write_config(COMPANIES)

    def tearDown(self):
        self._tmp.cleanup()

    def write_config(self, companies):
        self.config.write_text(json.dumps({"companies": companies}), encoding="utf-8")

    def run_main(self, *args, fake=None):
        fake = fake or FakeGreenhouse(DEFAULT_BOARDS)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch(URLOPEN, fake), redirect_stdout(stdout), redirect_stderr(stderr):
            code = collect_jobs.main(["--config", str(self.config), "--data-dir", str(self.data_dir), *args])
        return code, stdout.getvalue(), stderr.getvalue(), fake

    def snapshot(self, filename):
        return json.loads((self.data_dir / filename).read_text(encoding="utf-8"))

    def db_rows(self, sql, params=()):
        conn = database.connect(self.db)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def test_all_collects_every_enabled_company(self):
        code, out, err, fake = self.run_main("--all", "--db", str(self.db))
        self.assertEqual(code, 0, err)
        self.assertEqual(fake.requested, ["muonspace", "planetlabs", "astspacemobile", "leolabsinc"])
        self.assertIn("Summary: 4/4 companies succeeded", out)
        self.assertIn("Collected 1 LeoLabs jobs (1 duplicates removed)", out)

        muon = self.snapshot("muon_space_jobs.json")
        self.assertEqual(muon["source"], {"ats_provider": "greenhouse", "board_token": "muonspace",
                                          "company": "Muon Space"})
        self.assertEqual(muon["job_count"], 2)
        self.assertEqual(self.snapshot("planet_labs_jobs.json")["jobs"][0]["company"], "Planet Labs")

        counts = dict(self.db_rows("SELECT c.board_token, COUNT(j.id) FROM companies c "
                                   "JOIN jobs j ON j.company_id = c.id GROUP BY c.id"))
        self.assertEqual(counts, {"muonspace": 2, "planetlabs": 3, "astspacemobile": 1, "leolabsinc": 1})
        # Same ATS job id under two companies is two distinct rows.
        self.assertEqual(len(self.db_rows("SELECT id FROM jobs WHERE ats_job_id = '1'")), 2)

    def test_second_run_reports_unchanged_and_preserves_first_seen(self):
        self.run_main("--all", "--db", str(self.db))
        before = {r["ats_job_id"]: r["first_seen_at"] for r in self.db_rows("SELECT * FROM jobs")}
        boards = dict(DEFAULT_BOARDS, muonspace=[gh_job(1, "GNC Engineer", content="&lt;p&gt;New&lt;/p&gt;"),
                                                  gh_job(2)])
        code, out, _, _ = self.run_main("--all", "--db", str(self.db), fake=FakeGreenhouse(boards))
        self.assertEqual(code, 0)
        self.assertIn("0 inserted, 1 updated, 1 unchanged", out)  # Muon Space
        after = {r["ats_job_id"]: r["first_seen_at"] for r in self.db_rows("SELECT * FROM jobs")}
        self.assertEqual(before, after)

    def test_one_company_failure_is_isolated(self):
        self.data_dir.mkdir(parents=True)
        old = '{"collected_at": "2026-11-01T00:00:00+00:00", "jobs": []}'
        (self.data_dir / "planet_labs_jobs.json").write_text(old, encoding="utf-8")

        fake = FakeGreenhouse(DEFAULT_BOARDS, fail={"planetlabs"})
        code, out, err, _ = self.run_main("--all", "--db", str(self.db), fake=fake)

        self.assertEqual(code, 1)
        self.assertEqual(len(fake.requested), 4)  # later companies still attempted
        self.assertIn("ERROR: Planet Labs: HTTP 500", err)
        self.assertIn("Summary: 3/4 companies succeeded; failed: Planet Labs", out)
        self.assertEqual((self.data_dir / "planet_labs_jobs.json").read_text(encoding="utf-8"), old)
        for name in ("muon_space_jobs.json", "ast_spacemobile_jobs.json", "leolabs_jobs.json"):
            self.assertTrue((self.data_dir / name).exists(), name)
        tokens = {r[0] for r in self.db_rows(
            "SELECT DISTINCT c.board_token FROM companies c JOIN jobs j ON j.company_id = c.id")}
        self.assertEqual(tokens, {"muonspace", "astspacemobile", "leolabsinc"})

    def test_invalid_json_and_empty_board_count_as_failures(self):
        class Mixed(FakeGreenhouse):
            def __call__(self, request, timeout=None):
                if "/astspacemobile/" in request.full_url:
                    self.requested.append("astspacemobile")
                    return FakeResponse(b"<html>oops</html>")
                return super().__call__(request, timeout)

        boards = dict(DEFAULT_BOARDS, leolabsinc=[])
        code, out, err, _ = self.run_main("--all", fake=Mixed(boards))
        self.assertEqual(code, 1)
        self.assertIn("AST SpaceMobile: Invalid JSON", err)
        self.assertIn("LeoLabs: API returned zero jobs", err)
        self.assertIn("failed: AST SpaceMobile, LeoLabs", out)

    def test_single_company_selection(self):
        code, out, _, fake = self.run_main("--company", "Planet Labs")
        self.assertEqual(code, 0)
        self.assertEqual(fake.requested, ["planetlabs"])
        self.assertNotIn("Summary", out)
        self.assertEqual(sorted(p.name for p in self.data_dir.iterdir()), ["planet_labs_jobs.json"])

    def test_default_without_flags_is_muon_space(self):
        code, _, _, fake = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(fake.requested, ["muonspace"])
        self.assertTrue((self.data_dir / "muon_space_jobs.json").exists())

    def test_disabled_companies_skipped_by_all(self):
        self.write_config([COMPANIES[0], {**COMPANIES[1], "enabled": False}])
        code, _, _, fake = self.run_main("--all")
        self.assertEqual(code, 0)
        self.assertEqual(fake.requested, ["muonspace"])

    def test_existing_database_records_preserved(self):
        conn = database.initialize_database(self.db)
        try:
            muon_id = database.upsert_company(conn, "Muon Space", "greenhouse", "muonspace")
            normalized = [{"ats_job_id": "1", "title": "GNC Engineer", "location": "Remote",
                           "description": "Do things.", "application_url": "https://example.test/jobs/1",
                           "updated_at": "2026-11-01T00:00:00Z"}]
            database.upsert_jobs(conn, muon_id, normalized, seen_at="2026-01-01T00:00:00+00:00")
            job_row_id = conn.execute("SELECT id FROM jobs").fetchone()[0]
        finally:
            conn.close()

        code, out, _, _ = self.run_main("--all", "--db", str(self.db))
        self.assertEqual(code, 0)
        self.assertIn("1 inserted, 0 updated, 1 unchanged", out)
        row = self.db_rows("SELECT * FROM jobs WHERE id = ?", (job_row_id,))[0]
        self.assertEqual(row["company_id"], muon_id)
        self.assertEqual(row["first_seen_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(row["last_changed_at"], "2026-01-01T00:00:00+00:00")
        self.assertNotEqual(row["last_seen_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(self.db_rows("SELECT id FROM companies WHERE board_token = 'muonspace'")[0][0], muon_id)

    def test_config_errors_exit_2_without_network(self):
        fake = FakeGreenhouse(DEFAULT_BOARDS)
        self.config.write_text("{bad", encoding="utf-8")
        code, _, err, _ = self.run_main("--all", fake=fake)
        self.assertEqual(code, 2)
        self.assertIn("ERROR: configuration", err)
        self.write_config(COMPANIES)
        code, _, err, _ = self.run_main("--company", "Nope", fake=fake)
        self.assertEqual(code, 2)
        self.assertEqual(fake.requested, [])

    def test_argument_conflicts(self):
        for args in (["--all", "--company", "Muon Space"], ["--all", "--output", "x.json"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                self.run_main(*args)
            self.assertEqual(cm.exception.code, 2)

    def test_snapshot_filename_slug(self):
        self.assertEqual(CompanyConfig("AST SpaceMobile", "greenhouse", "x").snapshot_filename,
                         "ast_spacemobile_jobs.json")


if __name__ == "__main__":
    unittest.main()
