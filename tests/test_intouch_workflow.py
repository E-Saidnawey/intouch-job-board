"""Tests for the InTouch workflow template and its checker (src.intouch_workflow).

Offline: nothing in the workflow is run against the real project. The runtime tests
execute the template's own task scripts in a temporary directory, with the pipeline
command replaced by a harmless ``print``; no jobs are collected, no API is called and
no email is sent.
"""

from __future__ import annotations

import copy
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src import intouch_workflow as iw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT_ROOT / "intouch" / "aerospace_job_intelligence.example.iml"
REAL_ROOT = r"D:\jobs\intouch-job-board"      # stands in for a personal checkout (not a real path)
REAL_EMAIL = "someone@mail.test"


def template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def task(data: dict, name: str) -> dict:
    return next(t for t in data["tasks"] if t["name"] == name)


def personalized(data: dict) -> dict:
    """The template as a user would fill it in: real root and recipient."""
    text = json.dumps(data, ensure_ascii=False)
    text = text.replace(json.dumps(iw.PLACEHOLDER_ROOT)[1:-1], json.dumps(REAL_ROOT)[1:-1])
    text = text.replace(json.dumps(iw.PLACEHOLDER_ROOT.replace("\\", "\\\\"))[1:-1],
                        json.dumps(REAL_ROOT.replace("\\", "\\\\"))[1:-1])
    data = json.loads(text)
    task(data, "send-report-email")["properties"]["to"] = REAL_EMAIL
    return data


def set_command(data: dict, name: str, transform) -> dict:
    props = task(data, name)["properties"]
    props["command"] = transform(props["command"])
    return data


def set_args(data: dict, name: str, args: list[str]) -> dict:
    return set_command(data, name, lambda c: re.sub(r"(?m)^(\s*)args=\[.*\]$",
                                                    lambda m: m.group(1) + "args=" + repr(args), c))


def args_of(data: dict, name: str) -> list[str]:
    return iw._parse_args_line(task(data, name)["properties"]["command"])


class TemplateTests(unittest.TestCase):
    def test_template_exists_and_is_valid_json(self):
        self.assertTrue(TEMPLATE.is_file())
        self.assertIsInstance(template(), dict)

    def test_template_has_everything(self):
        self.assertEqual(iw.check_workflow(template(), template=True), [])

    def test_task_order(self):
        self.assertEqual([t["name"] for t in template()["tasks"]], list(iw.TASK_ORDER))

    def test_template_contains_no_personal_data(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        for needle in ("ToolRepository", "gmail", "C:\\\\Users", "/Users/"):
            self.assertNotIn(needle, text)
        self.assertEqual(task(template(), "send-report-email")["properties"]["to"], "you@example.com")

    def test_template_commands_match_the_pipeline(self):
        data = template()
        self.assertEqual(args_of(data, "collect-jobs")[2:],
                         ["src.collect_jobs", "--all", "--db", iw.PLACEHOLDER_ROOT + r"\data\jobs.db"])
        self.assertEqual(args_of(data, "evaluate-jobs")[2:],
                         ["src.batch_evaluator", "--profile", "config/claude_candidate_profile.json", "--call-api"])
        self.assertEqual(args_of(data, "generate-report")[2:],
                         ["src.generate_report", "--profile", "config/claude_candidate_profile.json"])

    def test_report_path_matches_generate_report_default(self):
        from src import generate_report
        self.assertEqual(generate_report.DEFAULT_OUTPUT.relative_to(PROJECT_ROOT).parts, iw.REPORT_RELATIVE)
        from src import claude_evaluator
        self.assertEqual(claude_evaluator.DEFAULT_DB.relative_to(PROJECT_ROOT).parts, iw.DB_RELATIVE)

    def test_template_is_tracked_and_personal_iml_ignored(self):
        lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("*.iml", lines)
        self.assertIn("!*.example.iml", lines)
        self.assertGreater(lines.index("!*.example.iml"), lines.index("*.iml"))   # exception must come after


class PersonalWorkflowTests(unittest.TestCase):
    def test_filled_in_workflow_passes_normal_check(self):
        self.assertEqual(iw.check_workflow(personalized(template())), [])

    def test_filled_in_workflow_fails_template_check(self):
        problems = iw.check_workflow(personalized(template()), template=True)
        self.assertTrue(any("placeholder" in p for p in problems), problems)
        self.assertTrue(any(REAL_EMAIL in p for p in problems), problems)

    def test_relative_db_path_is_accepted(self):
        data = personalized(template())
        args = args_of(data, "collect-jobs")
        args[-1] = "data/jobs.db"
        self.assertEqual(iw.check_workflow(set_args(data, "collect-jobs", args)), [])


class MissingPieceTests(unittest.TestCase):
    """Each broken workflow must be reported with a specific, readable problem."""

    def assertProblem(self, data, fragment, *, template_mode=False):
        problems = iw.check_workflow(data, template=template_mode)
        self.assertTrue(any(fragment in p for p in problems), f"{fragment!r} not in {problems}")

    def test_each_missing_task_is_reported(self):
        for name in iw.TASK_ORDER:
            with self.subTest(name):
                data = template()
                data["tasks"] = [t for t in data["tasks"] if t["name"] != name]
                self.assertProblem(data, f"missing task {name!r}")

    def test_duplicate_task(self):
        data = template()
        data["tasks"].append(copy.deepcopy(task(data, "verify-report")))
        self.assertProblem(data, "task 'verify-report' appears 2 times")

    def test_out_of_order_and_wrong_first_task(self):
        data = template()
        data["tasks"][0], data["tasks"][1] = data["tasks"][1], data["tasks"][0]
        self.assertProblem(data, "tasks are out of order")
        self.assertProblem(data, "first task must be 'collect-jobs'")

    def test_unexpected_task(self):
        data = template()
        data["tasks"].append({"name": "apply-to-jobs", "tool": "run_command", "properties": {}})
        self.assertProblem(data, "unexpected task(s): ['apply-to-jobs']")

    def test_missing_db_flag(self):
        data = template()
        self.assertProblem(set_args(data, "collect-jobs", args_of(data, "collect-jobs")[:4]),
                           "missing required flag --db")

    def test_wrong_db_path(self):
        data = template()
        args = args_of(data, "collect-jobs")
        args[-1] = iw.PLACEHOLDER_ROOT + r"\data\other.db"
        self.assertProblem(set_args(data, "collect-jobs", args), "--db must be")

    def test_missing_call_api(self):
        data = template()
        self.assertProblem(set_args(data, "evaluate-jobs", args_of(data, "evaluate-jobs")[:-1]),
                           "missing required flag --call-api")

    def test_real_profile_instead_of_sanitized(self):
        data = template()
        args = args_of(data, "generate-report")
        args[-1] = "config/candidate_profile.json"
        self.assertProblem(set_args(data, "generate-report", args), "--profile must be config/claude_candidate_profile.json")

    def test_unknown_flag_rejected_by_real_parser(self):
        data = template()
        self.assertProblem(set_args(data, "collect-jobs", args_of(data, "collect-jobs") + ["--everything"]),
                           "src.collect_jobs rejects its arguments: ")

    def test_wrong_module(self):
        data = template()
        args = args_of(data, "generate-report")
        args[2] = "src.job_matcher"
        self.assertProblem(set_args(data, "generate-report", args), "must run '<python> -m src.generate_report'")

    def test_python_outside_project_venv(self):
        data = template()
        args = args_of(data, "evaluate-jobs")
        args[0] = r"C:\Python314\python.exe"
        self.assertProblem(set_args(data, "evaluate-jobs", args), "Python must be the project's virtualenv")

    def test_inline_python_that_does_not_compile(self):
        data = set_command(template(), "generate-report", lambda c: c.replace("try:\n", "try\n", 1))
        self.assertProblem(data, "inline Python does not compile")

    def test_step_name_mismatch(self):
        data = set_command(template(), "generate-report",
                           lambda c: c.replace("step='generate-report'", "step='verify-report'"))
        self.assertProblem(data, "script says step='verify-report'")

    def test_different_roots(self):
        data = set_command(template(), "verify-report",
                           lambda c: c.replace(iw.PLACEHOLDER_ROOT, r"C:\elsewhere\intouch-job-board", 1))
        self.assertProblem(data, "tasks use different project roots")

    def test_missing_run_marker_logic(self):
        data = set_command(template(), "evaluate-jobs",
                           lambda c: c.replace("marker=logs/'intouch_current_run.json'\n", ""))
        self.assertProblem(data, "use the shared run marker")

    def test_timeout_not_above_inner_timeout(self):
        data = template()
        task(data, "evaluate-jobs")["properties"]["timeoutSeconds"] = 3600
        self.assertProblem(data, "timeoutSeconds (3600) must exceed the inner subprocess timeout (3600)")

    def test_success_codes_must_be_zero(self):
        data = template()
        task(data, "collect-jobs")["properties"]["successCodes"] = "0,1"
        self.assertProblem(data, "successCodes must be '0'")

    def test_verify_report_checks_wrong_file(self):
        data = set_command(template(), "verify-report", lambda c: c.replace("'job_intelligence.html'", "'other.html'"))
        self.assertProblem(data, "must check <root>/reports/job_intelligence.html")

    def test_email_task_problems(self):
        data = template()
        props = task(data, "send-report-email")["properties"]
        props["to"], props["subject"] = "", ""
        props["attachments"] = iw.PLACEHOLDER_ROOT + r"\reports\old.html"
        self.assertProblem(data, "'to' must contain an email address")
        self.assertProblem(data, "'subject' is missing")
        self.assertProblem(data, "'attachments' must be")

    def test_wrong_tool(self):
        data = template()
        task(data, "send-report-email")["tool"] = "run_command"
        task(data, "collect-jobs")["tool"] = "message"
        self.assertProblem(data, "task 'send-report-email': tool must be 'message'")
        self.assertProblem(data, "task 'collect-jobs': tool must be 'run_command'")

    def test_credentials_and_secrets_rejected(self):
        data = template()
        task(data, "collect-jobs")["properties"]["envVars"] = {"ANTHROPIC_API_KEY": "sk-ant-abcdefghijkl"}
        self.assertProblem(data, "properties.envVars is set")
        self.assertProblem(data, "appears to contain a secret")

    def test_schedule_rejected(self):
        data = template()
        data["schedule"] = {"cron": "0 7 * * *"}
        self.assertProblem(data, "workflow has a 'schedule'")

    def test_template_mode_catches_personal_data(self):
        data = template()
        task(data, "send-report-email")["properties"]["to"] = "real.person@gmail.com"
        task(data, "send-report-email")["properties"]["body"] = r"see C:\Users\me\report"
        self.assertProblem(data, "real-looking email address 'real.person@gmail.com'", template_mode=True)
        self.assertProblem(data, "personal user directory path", template_mode=True)

    def test_not_an_object_and_empty_tasks(self):
        self.assertEqual(iw.check_workflow([]), ["workflow must be a JSON object"])
        self.assertIn("workflow 'tasks' list is missing or empty", iw.check_workflow({"name": "x", "description": "y"}))


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = iw.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_default_checks_template(self):
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("OK (template; 6 tasks", out)

    def test_problems_exit_1_and_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = template()
            data["tasks"] = data["tasks"][:3]
            path = Path(tmp) / "mine.iml"
            path.write_text(json.dumps(data), encoding="utf-8")
            code, out, _ = self.run_cli(str(path))
        self.assertEqual(code, 1)
        self.assertIn("missing task 'verify-report'", out)
        self.assertIn("missing task 'send-report-email'", out)

    def test_personal_file_is_not_held_to_template_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mine.iml"
            path.write_text(json.dumps(personalized(template())), encoding="utf-8")
            code, out, _ = self.run_cli(str(path))
            self.assertEqual(code, 0, out)
            code, out, _ = self.run_cli(str(path), "--template")
            self.assertEqual(code, 1)

    def test_unreadable_or_invalid_file_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.iml"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(self.run_cli(str(bad))[0], 2)
            self.assertEqual(self.run_cli(str(Path(tmp) / "missing.iml"))[0], 2)


class TemplateScriptRuntimeTests(unittest.TestCase):
    """Execute the template's task scripts in a temporary root (pipeline command replaced)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "project"
        self.root.mkdir()
        self.data = template()

    def run_step(self, name: str, fake_args: list[str] | None = None) -> subprocess.CompletedProcess:
        command = task(self.data, name)["properties"]["command"].replace(iw.PLACEHOLDER_ROOT, str(self.root))
        if fake_args is not None:
            command = re.sub(r"(?m)^(\s*)args=\[.*\]$", lambda m: m.group(1) + "args=" + repr(fake_args), command)
        return subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=60)

    def log_text(self) -> str:
        marker = json.loads((self.root / "logs" / "intouch_current_run.json").read_text(encoding="utf-8"))
        return (self.root / "logs" / f"intouch_{marker['run_id']}.log").read_text(encoding="utf-8")

    def test_steps_refuse_to_run_without_marker(self):
        result = self.run_step("generate-report", [sys.executable, "-c", "print('should not run')"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("no run marker", result.stderr)

    def test_full_sequence_shares_one_log(self):
        ok = [sys.executable, "-c", "print('pipeline step ok')"]
        self.assertEqual(self.run_step("collect-jobs", ok).returncode, 0)
        self.assertEqual(self.run_step("evaluate-jobs", ok).returncode, 0)
        (self.root / "reports").mkdir()
        (self.root / "reports" / "job_intelligence.html").write_text("<html></html>", encoding="utf-8")
        self.assertEqual(self.run_step("generate-report", ok).returncode, 0)
        self.assertEqual(self.run_step("verify-report").returncode, 0)
        self.assertEqual(self.run_step("email-confirmed").returncode, 0)
        log = self.log_text()
        for name in ("collect-jobs", "evaluate-jobs", "generate-report", "verify-report", "email-confirmed"):
            self.assertIn(f"START {name}", log)
            self.assertIn(f"END {name} status=SUCCESS exit_code=0", log)
        self.assertIn("OUTPUT pipeline step ok", log)
        self.assertIn("REPORT_OK path=", log)
        self.assertIn("EMAIL_TASK_RETURNED_SUCCESS", log)

    def test_failing_command_fails_the_step(self):
        self.run_step("collect-jobs", [sys.executable, "-c", "print('ok')"])
        result = self.run_step("evaluate-jobs", [sys.executable, "-c", "import sys; sys.exit(3)"])
        self.assertEqual(result.returncode, 3)
        self.assertIn("END evaluate-jobs status=FAILED exit_code=3", self.log_text())

    def test_verify_report_fails_on_missing_or_empty_report(self):
        self.run_step("collect-jobs", [sys.executable, "-c", "print('ok')"])
        self.assertEqual(self.run_step("verify-report").returncode, 1)
        (self.root / "reports").mkdir()
        (self.root / "reports" / "job_intelligence.html").write_text("", encoding="utf-8")
        self.assertEqual(self.run_step("verify-report").returncode, 1)
        self.assertIn("Report is empty", self.log_text())


if __name__ == "__main__":
    unittest.main()
