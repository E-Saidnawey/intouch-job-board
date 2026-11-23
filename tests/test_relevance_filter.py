"""Offline tests for the weighted title-relevance classifier and its read-only CLI."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src import database, relevance_filter
from src.relevance_filter import HIGH, LOW, MEDIUM, classify_relevance


class RelevanceTestCase(unittest.TestCase):
    def assertClassified(self, expected: str, titles):
        for title in titles:
            with self.subTest(title=title):
                result = classify_relevance(title)
                self.assertEqual(result["classification"], expected, f"{result['score']}: {result['reason']}")

    def reason(self, title: str) -> str:
        return classify_relevance(title)["reason"]


class SpecExamplesTests(RelevanceTestCase):
    def test_expected_high(self):
        self.assertClassified(HIGH, [
            "Mission Operations Engineer",
            "Operations and Automation Engineer",
            "Staff Radar Systems Engineer",
            "Senior FPGA RF Signal Processing Engineer",
            "Senior Instrument Scientist, Calibration Design",
            "Payload Performance Analyst",
            "Spacecraft Operations Specialist",
            "Sensor Characterization Scientist",
            "Mission Systems Analyst",
            "Integration & Verification Lead",
        ])

    def test_expected_medium(self):
        self.assertClassified(MEDIUM, [
            "Full Stack Software Engineer",
            "Senior Site Reliability Engineer",
            "Software Product Manager",
            "Partner Technical Lead - Japan",
        ])

    def test_expected_low(self):
        self.assertClassified(LOW, [
            "Manager, Capture & Proposal",
            "Director of Global Compliance",
            "Senior Commercial Attorney",
            "Events and Social Media Marketing Manager",
        ])

    def test_result_shape(self):
        result = classify_relevance("Mission Operations Engineer")
        self.assertEqual(set(result), {"classification", "score", "reason"})
        self.assertIsInstance(result["score"], float)
        self.assertEqual(result["score"], 7.0)
        self.assertIn("phrase 'mission operations' +5", result["reason"])
        self.assertIn("role 'engineer' +2", result["reason"])


class MatchingMethodTests(RelevanceTestCase):
    def test_phrase_words_not_double_counted(self):
        reason = self.reason("Mission Operations Engineer")
        self.assertNotIn("domain 'mission'", reason)
        self.assertNotIn("generic 'operations'", reason)
        self.assertEqual(classify_relevance("Signal Processing Engineer")["score"], 7.0)

    def test_repeated_words_count_once(self):
        self.assertEqual(classify_relevance("Satellite Satellite Engineer")["score"],
                         classify_relevance("Satellite Engineer")["score"])
        self.assertEqual(classify_relevance("Mission Operations Engineer, Mission Operations")["score"], 7.0)

    def test_only_strongest_role_counts(self):
        both = classify_relevance("Radar Engineer / Analyst")
        self.assertEqual(both["score"], classify_relevance("Radar Engineer")["score"])
        self.assertNotIn("role 'analyst'", both["reason"])
        self.assertIn("role 'technical lead' +2", self.reason("Payload Technical Lead"))
        self.assertIn("role 'engineer'", self.reason("Lead Payload Engineer"))

    def test_individual_domain_words_with_role(self):
        self.assertClassified(HIGH, [
            "Satellite Engineer", "Spacecraft Engineer", "Payload Engineer", "Telemetry Engineer",
            "Optical Engineer", "Infrared Sensor Scientist", "Calibration Scientist", "Radar Analyst",
            "Avionics Engineer", "Propulsion Engineer", "Flight Test Engineer", "Anomaly Resolution Engineer",
            "Test Automation Engineer", "Verification Engineer, Spacecraft",
        ])

    def test_scores_are_additive_and_monotonic(self):
        base = classify_relevance("Radar Engineer")["score"]
        self.assertGreater(classify_relevance("Radar Signal Processing Engineer")["score"], base)
        self.assertGreater(base, classify_relevance("Engineer")["score"])


class SynonymAndStructureTests(RelevanceTestCase):
    def test_synonyms_and_abbreviations(self):
        self.assertClassified(HIGH, [
            "Satellite Ops Engineer",
            "Mission Ops Specialist",
            "Sr. GNC Eng.",
            "I&T Technician",
            "V&V Engineer, Flight Software",
            "Guidance, Navigation and Control Engineer",
            "Guidance, Navigation & Control Engineer",
            "Electro-Optical Systems Engineer",
            "Hardware-in-the-Loop Test Engineer",
        ])
        self.assertIn("phrase 'satellite operations'", self.reason("Satellite Ops Engineer"))

    def test_plurals(self):
        self.assertClassified(HIGH, ["Sensors Engineer", "Payloads Analyst", "Satellites Operations Lead"])
        self.assertIn("phrase 'radar systems'", self.reason("Radar System Engineer"))

    def test_varied_title_structures(self):
        self.assertClassified(HIGH, [
            "Engineer, Mission Operations",
            "Engineer II - Spacecraft Operations",
            "Principal Engineer (Radar Systems)",
            "Scientist | Sensor Characterization",
            "Analyst: Payload Performance",
            "MISSION OPERATIONS ENGINEER",
            "  mission   operations   engineer  ",
            "Software Engineer, Flight Software",
            "Remote Sensing Scientist - Hyperspectral",
        ])

    def test_seniority_not_penalized(self):
        base = classify_relevance("Radar Systems Engineer")["score"]
        for prefix in ("Senior", "Sr.", "Staff", "Lead", "Principal", "Junior", "Chief"):
            with self.subTest(prefix=prefix):
                result = classify_relevance(f"{prefix} Radar Systems Engineer")
                self.assertEqual(result["classification"], HIGH)
                self.assertEqual(result["score"], base)
        self.assertEqual(classify_relevance("Radar Systems Engineer III")["score"], base)


class GenericTermTests(RelevanceTestCase):
    def test_generic_terms_alone_never_high(self):
        self.assertClassified(MEDIUM, [
            "Engineer",
            "Operations Engineer",
            "Systems Engineer",
            "Senior Systems Operations Engineer",
            "Software Systems Operations Engineer",
            "Senior Electrical Hardware Design Engineer",
            "Analyst",
            "Operations Manager",
            "Engineering Manager",
            "Data Analyst",
            "Technical Lead",
        ])

    def test_capped_reason_explains(self):
        result = classify_relevance("Senior Electrical Hardware Design Engineer")
        self.assertGreaterEqual(result["score"], relevance_filter.HIGH_THRESHOLD)
        self.assertIn("no aerospace domain term", result["reason"])


class NegativeSignalTests(RelevanceTestCase):
    def test_unrelated_functions_low(self):
        self.assertClassified(LOW, [
            "Marketing Manager",
            "Account Executive, Federal Sales",
            "Senior Financial Analyst",
            "Finance Manager",
            "Legal Counsel",
            "Paralegal",
            "Technical Recruiter",
            "Talent Acquisition Partner",
            "Procurement Specialist",
            "Human Resources Business Partner",
            "HR Generalist",
            "People Operations Coordinator",
            "Business Development Director",
            "Contracts Manager",
            "Payroll Specialist",
            "Executive Assistant to the CEO",
            "Marketing Automation Specialist",
            "Legal Counsel, Space Systems",  # domain phrase does not rescue a legal role
        ])

    def test_people_operations_does_not_score_operations(self):
        self.assertNotIn("generic 'operations'", self.reason("People Operations Coordinator"))

    def test_sales_in_technical_engineering_title_not_low(self):
        self.assertClassified(MEDIUM, ["Sales Engineer", "Radar Sales Engineer", "Procurement Engineer"])
        self.assertIn("damped: technical role", self.reason("Sales Engineer"))
        self.assertClassified(HIGH, ["Satellite Payload Sales Engineer"])

    def test_negative_damping_is_configurable(self):
        with mock.patch.object(relevance_filter, "NEGATIVE_DAMPING", 1.0):
            self.assertEqual(classify_relevance("Sales Engineer")["classification"], LOW)

    def test_quality_assurance_context(self):
        self.assertClassified(HIGH, ["Quality Assurance Automation Engineer", "QA Automation Engineer"])
        self.assertNotIn("negative", self.reason("Quality Assurance Automation Engineer"))
        self.assertClassified(LOW, ["Quality Assurance Specialist", "QA Clerk"])
        self.assertClassified(MEDIUM, ["Quality Assurance Engineer"])
        self.assertClassified(HIGH, ["Mission Assurance Engineer"])


class FalsePositiveTests(RelevanceTestCase):
    def test_whole_words_only(self):
        cases = {
            "Salesforce Developer": "negative 'sales'",
            "Human Factors Engineer": "negative 'human resources'",
            "Marketplace Engineer": "negative 'marketing'",
            "Legacy Systems Engineer": "negative 'legal'",
            "Spaceport Engineer": "domain 'space'",
            "Radarsat Data Analyst": "domain 'radar'",
            "Taxonomy Specialist": "negative 'tax'",
            "Testimonial Writer": "domain 'test'",
            "Chrome Developer": "negative 'hr'",
            "Missionary Coordinator": "domain 'mission'",
        }
        for title, absent in cases.items():
            with self.subTest(title=title):
                self.assertNotIn(absent, self.reason(title))
        self.assertClassified(MEDIUM, ["Salesforce Developer", "Human Factors Engineer"])

    def test_aerospace_words_in_unrelated_roles_not_high(self):
        self.assertClassified(MEDIUM, ["Mission Coordinator", "Space Planner"])
        self.assertClassified(LOW, ["Satellite Sales Manager", "Spacecraft Program Finance Lead"])


class AmbiguousTitleTests(RelevanceTestCase):
    def test_missing_titles_medium(self):
        for title in (None, "", "   ", 42):
            with self.subTest(title=title):
                result = classify_relevance(title)  # type: ignore[arg-type]
                self.assertEqual(result, {"classification": MEDIUM, "score": 0.0,
                                          "reason": "title missing; manual review"})

    def test_unrecognized_titles_medium(self):
        self.assertClassified(MEDIUM, ["TBD", "Chief of Staff", "Senior", "Associate", "Job Posting", "---", "Ninja"])
        self.assertIn("no recognized terms", self.reason("Chief of Staff"))

    def test_deterministic(self):
        results = {tuple(classify_relevance("Payload Performance Analyst").items()) for _ in range(5)}
        self.assertEqual(len(results), 1)


class EoIrRegressionTests(RelevanceTestCase):
    def test_principal_mission_architect_eo_ir(self):
        result = classify_relevance("Principal Mission Architect - EO/IR")
        self.assertEqual(result["classification"], HIGH, result["reason"])
        self.assertIn("phrase 'eo ir' +5", result["reason"])
        self.assertEqual(classify_relevance("Principal Mission Architect")["classification"], MEDIUM)

    def test_eo_ir_variants(self):
        self.assertClassified(HIGH, [
            "EO/IR Systems Engineer",
            "EO-IR Systems Engineer",
            "EO / IR Payload Analyst",
            "EOIR Payload Analyst",
            "Electro-Optical Engineer",
            "Electro Optical Systems Engineer",
            "Electro-Optic Engineer",
            "Electro-Optics Scientist",
            "Electrooptical Engineer",
            "Electro-Optical/Infrared Sensor Engineer",
            "Infrared Systems Engineer",
            "Multispectral Imaging Scientist",
            "Multi-Spectral Imaging Scientist",
            "SWIR Camera Engineer",
            "Senior Staff EO/IR Engineer",
        ])

    def test_eo_ir_phrase_not_double_counted(self):
        self.assertNotIn("domain 'electro", self.reason("Electro-Optic Engineer"))
        self.assertEqual(classify_relevance("EO/IR Engineer")["score"], 7.0)

    def test_eo_ir_no_false_positives(self):
        self.assertNotIn("eo ir", self.reason("CEO Irvine Office"))
        self.assertNotIn("eo ir", self.reason("Video Editor"))
        self.assertClassified(MEDIUM, ["IR Specialist"])  # bare "IR" is ambiguous


class RegulatoryAffairsRegressionTests(RelevanceTestCase):
    def test_regulatory_affairs_launch_services_low(self):
        for title in ("Director, Regulatory Affairs (Launch Services)",
                      "Manager, Regulatory Affairs (Launch Services)"):
            with self.subTest(title=title):
                result = classify_relevance(title)
                self.assertEqual(result["classification"], LOW, result["reason"])
                self.assertIn("negative 'regulatory affairs' -6", result["reason"])
                self.assertIn("domain 'launch' +2", result["reason"])  # still explained, just outweighed

    def test_regulatory_variants(self):
        self.assertClassified(LOW, [
            "Regulatory Affairs Specialist",
            "Senior Manager, Regulatory Affairs",
            "Regulatory Affairs Lead - Spectrum & Launch Licensing",
            "Regulatory Counsel",
            "Regulatory Analyst",
        ])

    def test_launch_still_positive_without_regulatory(self):
        self.assertClassified(HIGH, ["Launch Operations Engineer"])
        self.assertClassified(MEDIUM, ["Regulatory Engineer"])  # technical role damps the negative


class SubcontractsRegressionTests(RelevanceTestCase):
    def test_senior_subcontracts_specialist_low(self):
        result = classify_relevance("Senior Subcontracts Specialist")
        self.assertEqual(result["classification"], LOW, result["reason"])
        self.assertIn("negative 'subcontracts' -6", result["reason"])

    def test_subcontract_variants(self):
        self.assertClassified(LOW, [
            "Subcontract Administrator",
            "Subcontracts Administrator",
            "Subcontracts Manager",
            "Manager, Subcontract Administration",
            "Subcontracts Administration Lead",
            "Subcontracting Specialist",
            "Senior Subcontracts Administrator - Spacecraft Programs",
        ])
        self.assertIn("negative 'subcontract administration' -6",
                      self.reason("Manager, Subcontract Administration"))

    def test_subcontract_in_engineering_title_not_low(self):
        self.assertClassified(MEDIUM, ["Subcontract Management Engineer"])
        self.assertIn("damped: technical role", self.reason("Subcontract Management Engineer"))


class ThresholdConfigurationTests(RelevanceTestCase):
    def test_thresholds_are_adjustable(self):
        with mock.patch.object(relevance_filter, "HIGH_THRESHOLD", 100.0):
            self.assertEqual(classify_relevance("Mission Operations Engineer")["classification"], MEDIUM)
        with mock.patch.object(relevance_filter, "LOW_THRESHOLD", 5.0):
            self.assertEqual(classify_relevance("Full Stack Software Engineer")["classification"], LOW)
        with mock.patch.object(relevance_filter, "MIN_DOMAIN_FOR_HIGH", 0.0):
            self.assertEqual(classify_relevance("Senior Systems Operations Engineer")["classification"], MEDIUM)
            self.assertEqual(classify_relevance("Senior Electrical Hardware Design Engineer")["classification"],
                             HIGH)


class ReadOnlyCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "jobs.db"
        conn = database.initialize_database(self.db)
        try:
            muon = database.upsert_company(conn, "Muon Space", "greenhouse", "muonspace")
            sling = database.upsert_company(conn, "Slingshot Aerospace", "ashby", "slingshot-aerospace")
            seen = "2026-11-01T00:00:00+00:00"
            database.upsert_jobs(conn, muon, [
                {"ats_job_id": "101", "title": "Mission Operations Engineer", "location": "New York, NY"},
                {"ats_job_id": "102", "title": "Mission Operations Engineer", "location": "US - Remote"},
                {"ats_job_id": "103", "title": "Payload Performance Analyst", "location": "Boston, MA/Hybrid"},
                {"ats_job_id": "104", "title": "Radar Systems Engineer", "location": "San Jose, CA"},
                {"ats_job_id": "105", "title": "Full Stack Software Engineer", "location": "Remote"},
                {"ats_job_id": "106", "title": "Senior Commercial Attorney", "location": "New York, NY"},
                {"ats_job_id": "107", "title": "Closed Radar Role Engineer", "location": "New York, NY"},
            ], seen_at=seen)
            database.upsert_jobs(conn, sling, [
                {"ats_job_id": "18b03b0e-f24b-44a9-a446-34cea181c22f", "title": "Manager, Capture & Proposal",
                 "location": "United States"},
                {"ats_job_id": "uuid-2", "title": "Spacecraft Operations Specialist", "location": None},
            ], seen_at=seen)
            conn.execute("UPDATE jobs SET is_active = 0 WHERE ats_job_id = '107'")
        finally:
            conn.close()

    def tearDown(self):
        self._tmp.cleanup()

    def digest(self) -> str:
        return hashlib.sha256(self.db.read_bytes()).hexdigest()

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = relevance_filter.main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_report(self):
        before = self.digest()
        code, out, err = self.run_cli("--db", str(self.db))
        self.assertEqual(code, 0, err)
        self.assertIn("8 active jobs; 7 with eligible locations", out)
        self.assertIn("1 excluded as incompatible", out)
        self.assertRegex(out, r"high\s+4\b")
        self.assertRegex(out, r"medium\s+1\b")
        self.assertRegex(out, r"low\s+2\b")

        # Distinct ATS postings with the same title are listed separately.
        self.assertIn("greenhouse:101", out)
        self.assertIn("greenhouse:102", out)
        self.assertEqual(out.count("Mission Operations Engineer"), 2)
        # Eligible locations include boston_hybrid and review; explanation is printed.
        self.assertIn("Payload Performance Analyst | Boston, MA/Hybrid (boston_hybrid)", out)
        self.assertIn("Spacecraft Operations Specialist | (no location) (review)", out)
        self.assertIn("phrase 'payload performance' +5", out)
        self.assertIn("Full Stack Software Engineer | Remote (review)", out)
        # Incompatible location, low relevance and inactive jobs are not listed.
        self.assertNotIn("Radar Systems Engineer", out)
        self.assertNotIn("Attorney", out)
        self.assertNotIn("Capture", out)
        self.assertNotIn("Closed Radar Role", out)
        self.assertEqual(self.digest(), before)

    def test_high_sorted_by_score(self):
        _, out, _ = self.run_cli("--db", str(self.db), "--show", "high")
        high_section = out.split("== high")[1]
        scores = [float(line.split("]")[0].strip(" [")) for line in high_section.splitlines()
                  if line.strip().startswith("[")]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertNotIn("== medium", out)

    def test_show_low(self):
        _, out, _ = self.run_cli("--db", str(self.db), "--show", "low")
        self.assertIn("ashby:18b03b0e-f24b-44a9-a446-34cea181c22f", out)
        self.assertIn("Senior Commercial Attorney", out)

    def test_invalid_show(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli("--db", str(self.db), "--show", "urgent")

    def test_missing_and_invalid_database(self):
        missing = Path(self._tmp.name) / "missing.db"
        code, _, err = self.run_cli("--db", str(missing))
        self.assertEqual(code, 1)
        self.assertIn("database not found", err)
        self.assertFalse(missing.exists())

        bogus = Path(self._tmp.name) / "bogus.db"
        bogus.write_text("not sqlite", encoding="utf-8")
        code, _, err = self.run_cli("--db", str(bogus))
        self.assertEqual(code, 1)
        self.assertIn("cannot read", err)

    def test_prioritize_keeps_posting_ids(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            rows, location_counts = relevance_filter.prioritize(relevance_filter.load_jobs(conn))
        finally:
            conn.close()
        ids = [(r["ats_provider"], r["ats_job_id"]) for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(location_counts["incompatible"], 1)
        self.assertEqual(rows[0]["classification"], HIGH)


if __name__ == "__main__":
    unittest.main()
