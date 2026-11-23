"""Offline tests for the deterministic location classifier and its read-only CLI."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src import database, location_filter
from src.location_filter import (
    BOSTON_HYBRID, CLASSIFICATIONS, COMPATIBLE, INCOMPATIBLE, REVIEW, classify_location,
)


class ClassifierTestCase(unittest.TestCase):
    def assertClassified(self, cases: dict):
        for location, expected in cases.items():
            with self.subTest(location=location):
                result = classify_location(location)
                self.assertEqual(result["classification"], expected, result["reason"])
                self.assertIsInstance(result["reason"], str)
                self.assertTrue(result["reason"])


class ObservedExamplesTests(ClassifierTestCase):
    """The real postings' location strings from the requirements."""

    def test_observed_examples(self):
        self.assertClassified({
            "US - Remote": COMPATIBLE,
            "Remote (U.S.)": COMPATIBLE,
            "United States, Remote": COMPATIBLE,
            "US, Remote; Canada, Remote": COMPATIBLE,
            "Remote": REVIEW,
            "Other - Remote": REVIEW,
            "United States": REVIEW,
            "US": REVIEW,
            "Remote (Europe / Middle East / Africa)": INCOMPATIBLE,
            "EMEA - Remote": INCOMPATIBLE,
            "Germany, Remote": INCOMPATIBLE,
            "San Jose, CA": INCOMPATIBLE,
            "Midland, Texas, United States": INCOMPATIBLE,
            "New York, NY": COMPATIBLE,
            "Brooklyn, NY": COMPATIBLE,
            "Boston, MA/Hybrid": BOSTON_HYBRID,
            "Boston, MA": INCOMPATIBLE,
            "Menlo Park, California/Hybrid": INCOMPATIBLE,
        })

    def test_result_shape(self):
        result = classify_location("New York, NY")
        self.assertEqual(set(result), {"classification", "reason"})
        self.assertIn(result["classification"], CLASSIFICATIONS)


class NewYorkCityTests(ClassifierTestCase):
    def test_boroughs_and_aliases(self):
        self.assertClassified({
            "Manhattan, NY": COMPATIBLE,
            "Brooklyn": COMPATIBLE,
            "Queens, New York": COMPATIBLE,
            "The Bronx, NY": COMPATIBLE,
            "Bronx": COMPATIBLE,
            "Staten Island, NY": COMPATIBLE,
            "Long Island City, NY": COMPATIBLE,
            "NYC": COMPATIBLE,
            "New York City": COMPATIBLE,
            "New York": COMPATIBLE,
            "New York, New York, United States": COMPATIBLE,
            "New York NY": COMPATIBLE,
            "Greater New York City Area": COMPATIBLE,
            "United States - New York": COMPATIBLE,
        })

    def test_nyc_with_work_mode(self):
        self.assertClassified({
            "New York, NY (Hybrid)": COMPATIBLE,
            "New York, NY - On-site": COMPATIBLE,
            "Remote - New York City": COMPATIBLE,
            "Brooklyn, NY / In-Office": COMPATIBLE,
        })

    def test_no_substring_false_positives(self):
        self.assertClassified({
            "Leeds, Yorkshire": INCOMPATIBLE,
            "York, PA": INCOMPATIBLE,
            "York, United Kingdom": INCOMPATIBLE,
            "New York Mills, MN": INCOMPATIBLE,
            "Newark, NJ": INCOMPATIBLE,
            "Queensland, Australia": INCOMPATIBLE,
            "Brooklyn Park, MN": INCOMPATIBLE,
            "Bostonia, CA": INCOMPATIBLE,
            "Remoteville, TX": INCOMPATIBLE,  # 'remote' must be a whole word
        })

    def test_new_york_state_outside_nyc(self):
        self.assertClassified({
            "Albany, New York": INCOMPATIBLE,
            "Rochester, NY": INCOMPATIBLE,
            "Buffalo, New York, United States": INCOMPATIBLE,
            "NY": REVIEW,
            "New York State": REVIEW,
        })


class RemoteTests(ClassifierTestCase):
    def test_explicit_us_remote(self):
        self.assertClassified({
            "Remote - USA": COMPATIBLE,
            "Remote, United States of America": COMPATIBLE,
            "U.S. Remote": COMPATIBLE,
            "Remote US": COMPATIBLE,
            "USA (Remote)": COMPATIBLE,
            "Remote - US or Canada": COMPATIBLE,
            "Remote - USA (East Coast)": COMPATIBLE,
            "Fully Remote - United States": COMPATIBLE,
            "Work from Home, US": COMPATIBLE,
        })

    def test_unspecified_remote(self):
        self.assertClassified({
            "Remote - Anywhere": REVIEW,
            "Remote (Global)": REVIEW,
            "Worldwide - Remote": REVIEW,
            "Remote - North America": REVIEW,
            "Virtual": REVIEW,
        })

    def test_remote_tied_to_non_nyc_us_location(self):
        self.assertClassified({
            "San Francisco, CA (Remote)": REVIEW,
            "Remote - California": REVIEW,
            "Remote - NY": REVIEW,
            "Boston, MA - Remote": REVIEW,
        })

    def test_foreign_remote(self):
        self.assertClassified({
            "Remote - Canada": INCOMPATIBLE,
            "UK - Remote": INCOMPATIBLE,
            "Remote (APAC)": INCOMPATIBLE,
            "Remote, India": INCOMPATIBLE,
        })


class BostonTests(ClassifierTestCase):
    def test_boston_area_hybrid(self):
        self.assertClassified({
            "Boston, MA (Hybrid)": BOSTON_HYBRID,
            "Hybrid - Boston": BOSTON_HYBRID,
            "Boston, Massachusetts, United States / Hybrid": BOSTON_HYBRID,
            "Cambridge, MA/Hybrid": BOSTON_HYBRID,
            "Waltham, Massachusetts - Hybrid": BOSTON_HYBRID,
            "Greater Boston Area - Hybrid": BOSTON_HYBRID,
        })

    def test_boston_area_not_hybrid(self):
        self.assertClassified({
            "Boston": INCOMPATIBLE,
            "Boston, MA - On-site": INCOMPATIBLE,
            "Cambridge, MA": INCOMPATIBLE,
        })

    def test_suburb_names_require_massachusetts(self):
        self.assertClassified({
            "Cambridge, UK / Hybrid": INCOMPATIBLE,
            "Newton, Iowa - Hybrid": INCOMPATIBLE,
            "Burlington, VT (Hybrid)": INCOMPATIBLE,
        })


class ElsewhereTests(ClassifierTestCase):
    def test_other_us_and_abroad(self):
        self.assertClassified({
            "Washington, DC": INCOMPATIBLE,
            "On-site - Huntsville, Alabama": INCOMPATIBLE,
            "Denver, CO/Hybrid": INCOMPATIBLE,
            "California": INCOMPATIBLE,
            "TX": INCOMPATIBLE,
            "Toronto, Ontario, Canada": INCOMPATIBLE,
            "Munich, Germany": INCOMPATIBLE,
            "Luxembourg": INCOMPATIBLE,
        })


class AmbiguousTests(ClassifierTestCase):
    def test_missing_and_ambiguous(self):
        self.assertClassified({
            None: REVIEW,
            "": REVIEW,
            "   ": REVIEW,
            ";": REVIEW,
            "--": REVIEW,
            "Hybrid": REVIEW,
            "On-site": REVIEW,
            "Multiple Locations": REVIEW,
            "TBD": REVIEW,
            "HQ": REVIEW,
        })
        self.assertEqual(classify_location(None)["reason"], "location missing")

    def test_non_string_input(self):
        self.assertEqual(classify_location(123)["classification"], REVIEW)  # type: ignore[arg-type]


class NormalizationTests(ClassifierTestCase):
    def test_case_punctuation_and_whitespace(self):
        self.assertClassified({
            "  new york ,  ny  ": COMPATIBLE,
            "NEW YORK, NY": COMPATIBLE,
            "brooklyn,ny": COMPATIBLE,
            "us-remote": COMPATIBLE,  # unspaced hyphen -> "us remote"
            "U.S.A. — Remote": COMPATIBLE,
            "Remote – U.S.": COMPATIBLE,
            "BOSTON, MA / HYBRID": BOSTON_HYBRID,
            "Remote\t(U.S.)": COMPATIBLE,
            "remote (u.s.)": COMPATIBLE,
        })

    def test_multiple_locations_any_compatible_wins(self):
        self.assertClassified({
            "San Jose, CA; New York, NY": COMPATIBLE,
            "Germany, Remote; US - Remote": COMPATIBLE,
            "Boston, MA/Hybrid; Denver, CO": BOSTON_HYBRID,
            "Boston, MA/Hybrid; Brooklyn, NY": COMPATIBLE,
            "Remote; San Jose, CA": REVIEW,
            "Denver, CO; Munich, Germany": INCOMPATIBLE,
            "San Jose, CA;;  ": INCOMPATIBLE,
            "Austin, TX\nNew York, NY": COMPATIBLE,
        })

    def test_multi_location_reason_names_deciding_segment(self):
        result = classify_location("San Jose, CA; New York, NY")
        self.assertIn("New York, NY", result["reason"])

    def test_deterministic(self):
        self.assertEqual({classify_location("Remote")["reason"] for _ in range(5)},
                         {"remote with unspecified geography"})


class ReadOnlyCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "jobs.db"
        conn = database.initialize_database(self.db)
        try:
            muon = database.upsert_company(conn, "Muon Space", "greenhouse", "muonspace")
            sling = database.upsert_company(conn, "Slingshot Aerospace", "ashby", "slingshot-aerospace")
            database.upsert_jobs(conn, muon, [
                {"ats_job_id": "1", "title": "GNC Engineer", "location": "New York, NY"},
                {"ats_job_id": "2", "title": "Test Tech", "location": "San Jose, CA"},
                {"ats_job_id": "3", "title": "Mystery Role", "location": None},
            ], seen_at="2026-11-01T00:00:00+00:00")
            database.upsert_jobs(conn, sling, [
                {"ats_job_id": "a", "title": "Capture Manager", "location": "US - Remote"},
                {"ats_job_id": "b", "title": "Analyst", "location": "Remote"},
                {"ats_job_id": "c", "title": "Ops Lead", "location": "Boston, MA/Hybrid"},
                {"ats_job_id": "d", "title": "Closed Role", "location": "New York, NY"},
            ], seen_at="2026-11-01T00:00:00+00:00")
            conn.execute("UPDATE jobs SET is_active = 0 WHERE ats_job_id = 'd'")
        finally:
            conn.close()

    def tearDown(self):
        self._tmp.cleanup()

    def digest(self) -> str:
        return hashlib.sha256(self.db.read_bytes()).hexdigest()

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = location_filter.main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_report_counts_and_listings(self):
        before = self.digest()
        code, out, err = self.run_cli("--db", str(self.db))
        self.assertEqual(code, 0, err)
        self.assertIn("Location classification for 6 active jobs", out)
        self.assertRegex(out, r"compatible\s+2\b")
        self.assertRegex(out, r"boston_hybrid\s+1\b")
        self.assertRegex(out, r"review\s+2\b")
        self.assertRegex(out, r"incompatible\s+1\b")

        self.assertIn("== compatible (2) ==", out)
        self.assertIn("Muon Space | GNC Engineer | New York, NY", out)
        self.assertIn("Slingshot Aerospace | Capture Manager | US - Remote", out)
        self.assertIn("== review (2) ==", out)
        self.assertIn("Muon Space | Mystery Role | (no location)  [location missing]", out)
        self.assertIn("Slingshot Aerospace | Analyst | Remote  [remote with unspecified geography]", out)
        self.assertNotIn("Test Tech", out)       # incompatible not listed by default
        self.assertNotIn("Ops Lead", out)        # boston_hybrid not listed by default
        self.assertNotIn("Closed Role", out)     # inactive jobs excluded
        self.assertEqual(self.digest(), before)  # database untouched

    def test_show_option(self):
        code, out, _ = self.run_cli("--db", str(self.db), "--show", "boston_hybrid,incompatible")
        self.assertEqual(code, 0)
        self.assertIn("Ops Lead", out)
        self.assertIn("Test Tech", out)
        self.assertNotIn("== compatible", out)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli("--db", str(self.db), "--show", "nope")

    def test_connection_is_read_only(self):
        import sqlite3
        conn = location_filter.open_readonly(self.db)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM jobs")
        finally:
            conn.close()

    def test_missing_database_is_not_created(self):
        missing = Path(self._tmp.name) / "nope.db"
        code, _, err = self.run_cli("--db", str(missing))
        self.assertEqual(code, 1)
        self.assertIn("database not found", err)
        self.assertFalse(missing.exists())

    def test_non_database_file(self):
        bogus = Path(self._tmp.name) / "bogus.db"
        bogus.write_text("not sqlite", encoding="utf-8")
        code, _, err = self.run_cli("--db", str(bogus))
        self.assertEqual(code, 1)
        self.assertIn("cannot read", err)


if __name__ == "__main__":
    unittest.main()
