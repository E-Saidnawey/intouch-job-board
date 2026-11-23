"""Offline tests for candidate profile loading/validation, using temporary profiles."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src import candidate_profile
from src.candidate_profile import ProfileError, ProfileFileError, load_profile, validate_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID = {
    "schema_version": 1,
    "target_roles": ["Role area A", "Role area B"],
    "preferred_locations": {"primary": ["City A, ST", "Remote (United States)"], "secondary": ["City B (hybrid)"]},
    "years_experience": {"discipline_a": 4},
    "education": [{"degree": "M.S.", "field": "Field A", "institution": "University A", "status": "in_progress"}],
    "aerospace_experience": ["Experience A", "Experience B"],
    "technical_skills": ["Tool A", "Tool B"],
    "transferable_skills": ["Skill A"],
}


def profile(**changes) -> dict:
    data = copy.deepcopy(VALID)
    for key, value in changes.items():
        if value is ...:
            data.pop(key, None)
        else:
            data[key] = value
    return data


class TempProfileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, data, name="candidate_profile.json") -> Path:
        path = self.dir / name
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path

    def assertProblem(self, data, *fragments):
        errors = validate_profile(data)
        joined = "\n".join(errors)
        for fragment in fragments:
            self.assertIn(fragment, joined)
        return errors


class LoadProfileTests(TempProfileTestCase):
    def test_valid_profile_loads(self):
        self.assertEqual(load_profile(self.write(VALID)), VALID)
        self.assertEqual(validate_profile(VALID), [])

    def test_optional_secondary_and_schema_version(self):
        data = profile(schema_version=...)
        del data["preferred_locations"]["secondary"]
        self.assertEqual(validate_profile(data), [])
        data["preferred_locations"]["secondary"] = []
        self.assertEqual(validate_profile(data), [])

    def test_missing_file(self):
        with self.assertRaisesRegex(ProfileFileError, "Profile not found.*candidate_profile.example.json"):
            load_profile(self.dir / "nope.json")

    def test_invalid_json(self):
        with self.assertRaisesRegex(ProfileFileError, r"Invalid JSON .*line 1, column"):
            load_profile(self.write("{not json"))

    def test_invalid_profile_raises_with_all_errors(self):
        path = self.write(profile(target_roles=..., technical_skills="Python"))
        with self.assertRaises(ProfileError) as cm:
            load_profile(path)
        self.assertNotIsInstance(cm.exception, ProfileFileError)
        self.assertEqual(len(cm.exception.errors), 2)
        message = str(cm.exception)
        self.assertIn("2 problem(s)", message)
        self.assertIn("target_roles: required section is missing", message)
        self.assertIn("technical_skills: expected a list of strings, got str", message)


class SectionValidationTests(TempProfileTestCase):
    def test_top_level_must_be_object(self):
        for data in ([], "profile", 3, None):
            with self.subTest(data=data):
                self.assertProblem(data, "profile: expected a JSON object")

    def test_every_required_section(self):
        for section in candidate_profile.REQUIRED_SECTIONS:
            with self.subTest(section=section):
                errors = self.assertProblem(profile(**{section: ...}), f"{section}: required section is missing")
                self.assertEqual(len(errors), 1)

    def test_all_missing_sections_reported_together(self):
        self.assertEqual(len(validate_profile({})), len(candidate_profile.REQUIRED_SECTIONS))

    def test_string_list_sections(self):
        for section in candidate_profile.STRING_LIST_SECTIONS:
            with self.subTest(section=section):
                self.assertProblem(profile(**{section: "x"}), f"{section}: expected a list of strings, got str")
                self.assertProblem(profile(**{section: []}), f"{section}: must contain at least one entry")
                self.assertProblem(profile(**{section: ["ok", 5]}), f"{section}[1]: expected a non-empty string, got int")
                self.assertProblem(profile(**{section: ["ok", "  "]}), f"{section}[1]: expected a non-empty string")
                self.assertProblem(profile(**{section: ["Python", "python "]}), f"{section}[1]: duplicate entry")

    def test_preferred_locations(self):
        self.assertProblem(profile(preferred_locations=["NYC"]), "preferred_locations: expected an object")
        self.assertProblem(profile(preferred_locations={"secondary": ["X"]}),
                           "preferred_locations.primary: required list is missing")
        self.assertProblem(profile(preferred_locations={"primary": []}),
                           "preferred_locations.primary: must contain at least one entry")
        self.assertProblem(profile(preferred_locations={"primary": ["A"], "secondary": "B"}),
                           "preferred_locations.secondary: expected a list of strings")
        self.assertProblem(profile(preferred_locations={"primary": ["A"], "tertiary": ["B"]}),
                           "preferred_locations.tertiary: unknown key")

    def test_years_experience(self):
        self.assertProblem(profile(years_experience=4), "years_experience: expected a non-empty object")
        self.assertProblem(profile(years_experience={}), "years_experience: expected a non-empty object")
        for bad in (-1, "4", True, None, 61, [4]):
            with self.subTest(value=bad):
                self.assertProblem(profile(years_experience={"a": bad}),
                                   "years_experience.a: expected a number between 0 and 60")
        self.assertProblem(profile(years_experience={" ": 2}), "discipline names must be non-empty")
        self.assertEqual(validate_profile(profile(years_experience={"a": 0, "b": 2.5})), [])

    def test_education(self):
        self.assertProblem(profile(education={}), "education: expected a non-empty list of objects")
        self.assertProblem(profile(education=[]), "education: expected a non-empty list of objects")
        self.assertProblem(profile(education=["M.S."]), "education[0]: expected an object, got str")
        entry = dict(VALID["education"][0])
        for field in candidate_profile.EDUCATION_FIELDS:
            with self.subTest(field=field):
                partial = {k: v for k, v in entry.items() if k != field}
                self.assertProblem(profile(education=[partial]), f"education[0].{field}: required field is missing")
                self.assertProblem(profile(education=[{**entry, field: ""}]), f"education[0].{field}: expected a non-empty string")
        self.assertProblem(profile(education=[{**entry, "status": "graduated"}]),
                           "education[0].status: must be one of completed, in_progress")
        self.assertEqual(validate_profile(profile(education=[{**entry, "status": "completed"}])), [])

    def test_schema_version(self):
        self.assertProblem(profile(schema_version=2), "schema_version: unsupported version 2")

    def test_contact_information_rejected(self):
        self.assertProblem(profile(technical_skills=["Python", "someone@example.com"]),
                           "technical_skills[1]: looks like contact information")
        self.assertProblem(profile(transferable_skills=["Call (555) 123-4567"]),
                           "transferable_skills[0]: looks like contact information")
        self.assertEqual(validate_profile(profile(technical_skills=["C++", "Python 3.12", "ISO 9001"])), [])


EVIDENCE = [
    {"skill": "Capability A", "evidence": "Used capability A to do something measurable."},
    {"skill": "Capability B", "evidence": "Applied capability B in a generalized way."},
]


class ExperienceEvidenceTests(TempProfileTestCase):
    def test_backward_compatible_without_section(self):
        self.assertNotIn("experience_evidence", VALID)
        self.assertEqual(validate_profile(VALID), [])
        self.assertEqual(load_profile(self.write(VALID)), VALID)

    def test_valid_evidence_loads(self):
        data = profile(experience_evidence=EVIDENCE)
        self.assertEqual(validate_profile(data), [])
        self.assertEqual(load_profile(self.write(data))["experience_evidence"], EVIDENCE)

    def test_empty_list_allowed(self):
        self.assertEqual(validate_profile(profile(experience_evidence=[])), [])

    def test_must_be_list(self):
        for bad in ({"skill": "A", "evidence": "B"}, "A: B", 3, None):
            with self.subTest(value=bad):
                self.assertProblem(profile(experience_evidence=bad), "experience_evidence: expected a list of objects")

    def test_entries_must_be_objects(self):
        for bad in ("Python automation", ["skill", "evidence"], 7, None):
            with self.subTest(entry=bad):
                self.assertProblem(profile(experience_evidence=[EVIDENCE[0], bad]),
                                   "experience_evidence[1]: expected an object with 'skill' and 'evidence'")

    def test_required_fields(self):
        for field in ("skill", "evidence"):
            with self.subTest(field=field):
                entry = {k: v for k, v in EVIDENCE[0].items() if k != field}
                self.assertProblem(profile(experience_evidence=[entry]),
                                   f"experience_evidence[0].{field}: required field is missing")
                for blank in ("", "   ", None, 5, ["text"]):
                    self.assertProblem(profile(experience_evidence=[{**EVIDENCE[0], field: blank}]),
                                       f"experience_evidence[0].{field}: expected a non-empty string")

    def test_unknown_fields_rejected(self):
        self.assertProblem(profile(experience_evidence=[{"skill": "A", "evidance": "typo"}]),
                           "experience_evidence[0].evidence: required field is missing",
                           "experience_evidence[0].evidance: unknown field")

    def test_contact_information_rejected_in_evidence(self):
        self.assertProblem(profile(experience_evidence=[{"skill": "A", "evidence": "Email me at a@b.com"}]),
                           "experience_evidence[0].evidence: looks like contact information")

    def test_every_malformed_entry_reported(self):
        errors = validate_profile(profile(experience_evidence=["x", {"skill": ""}, EVIDENCE[1]]))
        self.assertEqual(len(errors), 3, errors)

    def test_cli_reports_evidence_count(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = candidate_profile.main(["--profile", str(self.write(profile(experience_evidence=EVIDENCE)))])
        self.assertEqual(code, 0)
        self.assertIn("experience_evidence:  2 (optional)", stdout.getvalue())
        self.assertNotIn("Capability A", stdout.getvalue())


class EducationYearTests(TempProfileTestCase):
    ENTRY = {"degree": "B.S.", "field": "Field B", "institution": "University B", "status": "completed"}

    def test_year_optional(self):
        self.assertEqual(validate_profile(profile(education=[self.ENTRY])), [])
        self.assertEqual(validate_profile(profile(education=[{**self.ENTRY, "year": 2022}])), [])

    def test_multiple_degrees(self):
        data = profile(education=[VALID["education"][0], {**self.ENTRY, "year": 2022}])
        self.assertEqual(validate_profile(data), [])

    def test_invalid_year(self):
        for bad in ("2022", 2022.0, True, 1800, 3000, None):
            with self.subTest(year=bad):
                self.assertProblem(profile(education=[{**self.ENTRY, "year": bad}]),
                                   "education[0].year: expected an integer year between 1950 and 2100")

    def test_unknown_education_field_rejected(self):
        self.assertProblem(profile(education=[{**self.ENTRY, "gpa": "4.0"}]), "education[0].gpa: unknown field")


class ShippedProfileFilesTests(unittest.TestCase):
    def test_example_profile_is_valid(self):
        data = load_profile(candidate_profile.EXAMPLE_PROFILE)
        self.assertEqual(set(candidate_profile.REQUIRED_SECTIONS) - set(data), set())

    def test_example_profile_demonstrates_optional_fields(self):
        data = load_profile(candidate_profile.EXAMPLE_PROFILE)
        self.assertGreaterEqual(len(data["experience_evidence"]), 1)
        self.assertTrue(all(set(e) == {"skill", "evidence"} for e in data["experience_evidence"]))
        self.assertTrue(any("year" in e for e in data["education"]))
        self.assertTrue(all(e["skill"].lower().startswith(("example", "another example"))
                            for e in data["experience_evidence"]))

    @unittest.skipUnless(candidate_profile.DEFAULT_PROFILE.exists(), "local profile not present")
    def test_local_profile_is_valid_and_same_schema(self):
        local = load_profile(candidate_profile.DEFAULT_PROFILE)
        example = load_profile(candidate_profile.EXAMPLE_PROFILE)
        self.assertEqual(set(local), set(example))

    # Structural checks only: this test file is public, so it never asserts the
    # local profile's actual (personal) content.
    @unittest.skipUnless(candidate_profile.DEFAULT_PROFILE.exists(), "local profile not present")
    def test_local_profile_evidence_and_education_structure(self):
        local = load_profile(candidate_profile.DEFAULT_PROFILE)
        self.assertGreaterEqual(len(local["experience_evidence"]), 1)
        for entry in local["experience_evidence"]:
            self.assertEqual(set(entry), {"skill", "evidence"})
        self.assertLessEqual({e["status"] for e in local["education"]}, set(candidate_profile.EDUCATION_STATUSES))
        self.assertTrue(all("year" in e for e in local["education"] if e["status"] == "completed"))

    @unittest.skipUnless(candidate_profile.DEFAULT_PROFILE.exists(), "local profile not present")
    def test_example_does_not_copy_local_content(self):
        local = load_profile(candidate_profile.DEFAULT_PROFILE)
        example = load_profile(candidate_profile.EXAMPLE_PROFILE)

        def texts(p):
            values = set(p["technical_skills"]) | set(p["aerospace_experience"]) | set(p["target_roles"])
            values |= {e[k] for e in p.get("experience_evidence", []) for k in ("skill", "evidence")}
            values |= {e[k] for e in p["education"] for k in ("field", "institution")}
            return {v.strip().lower() for v in values}

        self.assertEqual(texts(local) & texts(example), set())

    def test_local_profile_is_gitignored(self):
        lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("config/candidate_profile.json", [line.strip() for line in lines])
        self.assertNotIn("config/candidate_profile.example.json", [line.strip() for line in lines])

    def test_module_has_no_network_dependencies(self):
        source = Path(candidate_profile.__file__).read_text(encoding="utf-8")
        for module in ("urllib", "http", "socket", "requests", "ssl"):
            self.assertNotRegex(source, rf"^\s*(?:import|from)\s+{module}\b")


class CliTests(TempProfileTestCase):
    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = candidate_profile.main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_valid_prints_counts_not_contents(self):
        code, out, _ = self.run_cli("--profile", str(self.write(VALID)))
        self.assertEqual(code, 0)
        self.assertIn("Candidate profile OK", out)
        self.assertIn("2 primary, 1 secondary", out)
        self.assertNotIn("University A", out)
        self.assertNotIn("Tool A", out)

    def test_invalid_exit_1(self):
        code, _, err = self.run_cli("--profile", str(self.write(profile(education=...))))
        self.assertEqual(code, 1)
        self.assertIn("education: required section is missing", err)

    def test_missing_or_unparseable_exit_2(self):
        code, _, err = self.run_cli("--profile", str(self.dir / "missing.json"))
        self.assertEqual(code, 2)
        self.assertIn("Profile not found", err)
        code, _, err = self.run_cli("--profile", str(self.write("[1,")))
        self.assertEqual(code, 2)
        self.assertIn("Invalid JSON", err)


if __name__ == "__main__":
    unittest.main()
