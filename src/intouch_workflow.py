"""Check an InTouch AI workflow (.iml) for everything this pipeline needs.

Offline and read-only: the file is parsed, never run. Each task's inline Python is
compiled (not executed), and every pipeline command line is parsed with the real
``build_parser()`` of the module it calls, so a renamed or removed flag is caught.

    python -m src.intouch_workflow                    # check the public template
    python -m src.intouch_workflow path\\to\\your.iml   # check your own workflow

Exit codes: 0 everything present, 1 problems found (each one is listed), 2 the file
cannot be read or is not valid JSON.

Checks:
* the six tasks exist, once each, in pipeline order:
  collect-jobs, evaluate-jobs, generate-report, verify-report, send-report-email, email-confirmed;
* every run_command task runs inline Python that compiles, names its own step,
  uses the same project root, writes to logs/ and shares the run marker; only the
  first task creates the marker;
* collect-jobs runs ``src.collect_jobs --all --db <root>\\data\\jobs.db`` (the database
  the evaluator and the report read); evaluate-jobs runs ``src.batch_evaluator`` with
  the sanitized ``config/claude_candidate_profile.json`` and ``--call-api``;
  generate-report runs ``src.generate_report`` with the same profile;
* every pipeline command uses the project's ``.venv`` Python and is accepted by that
  module's own argument parser, and each task's InTouch timeout exceeds the inner
  subprocess timeout;
* verify-report checks ``reports/job_intelligence.html`` (the report's output path) and
  send-report-email attaches that same file and has a recipient and a subject;
* no credentials, API keys or schedule are stored in the file.

With ``template=True`` (automatic for ``*.example.iml``) it also checks that the file
holds placeholders only: the root ``C:\\path\\to\\intouch-job-board`` and an
``@example.com`` recipient, with no real paths or addresses.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import io
import json
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "intouch" / "aerospace_job_intelligence.example.iml"
PLACEHOLDER_ROOT = r"C:\path\to\intouch-job-board"
PLACEHOLDER_EMAIL_DOMAIN = "@example.com"

TASK_ORDER = ("collect-jobs", "evaluate-jobs", "generate-report", "verify-report",
              "send-report-email", "email-confirmed")
RUN_TASKS = ("collect-jobs", "evaluate-jobs", "generate-report", "verify-report", "email-confirmed")
SANITIZED_PROFILE = "config/claude_candidate_profile.json"
REPORT_RELATIVE = ("reports", "job_intelligence.html")
DB_RELATIVE = ("data", "jobs.db")
VENV_PYTHON_RELATIVE = (".venv", "Scripts", "python.exe")

# step -> (module, flags that must be present)
PIPELINE_STEPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "collect-jobs": ("src.collect_jobs", ("--all", "--db")),
    "evaluate-jobs": ("src.batch_evaluator", ("--profile", "--call-api")),
    "generate-report": ("src.generate_report", ("--profile",)),
}
CREDENTIAL_KEYS = ("credentialNames", "credentialEnvVarOverrides", "envVars", "environmentVariables")
SECRET_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}|ANTHROPIC_API_KEY\s*=\s*\S+|password\s*[=:]", re.I)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_ROOT_RE = re.compile(r'^root=pathlib\.Path\(r"(?P<root>[^"]*)"\)', re.M)
_STEP_RE = re.compile(r"^step='(?P<step>[^']*)'", re.M)
_ARGS_RE = re.compile(r"^\s*args=(?P<args>\[.*\])\s*$", re.M)
_WAIT_RE = re.compile(r"p\.wait\(timeout=(?P<wait>\d+)\)")


class WorkflowFileError(Exception):
    """The workflow file cannot be read or is not valid JSON."""


def load_workflow(path: str | Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkflowFileError(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkflowFileError(f"{path} is not valid JSON: {exc}") from exc


def _win(path: str) -> PureWindowsPath:
    return PureWindowsPath(path)


def _same_path(a: str | PureWindowsPath, b: str | PureWindowsPath) -> bool:
    return str(_win(str(a))).rstrip("\\").casefold() == str(_win(str(b))).rstrip("\\").casefold()


def _parse_args_line(command: str) -> list[str] | None:
    match = _ARGS_RE.search(command)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group("args"))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return value


def _cli_problem(module_name: str, argv: list[str]) -> str | None:
    """Parse ``argv`` with the module's own parser; return the parser's complaint, if any."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return f"cannot import {module_name}: {exc}"
    build = getattr(module, "build_parser", None)
    if build is None:
        return f"{module_name} has no build_parser()"
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            build().parse_args(argv)
    except SystemExit:
        lines = [line for line in stderr.getvalue().splitlines() if line.strip()]
        return lines[-1] if lines else "rejected by the argument parser"
    return None


def _check_run_task(task: dict[str, Any], name: str, roots: dict[str, str], problems: list[str]) -> None:
    where = f"task {name!r}"
    props = task.get("properties")
    if not isinstance(props, dict):
        problems.append(f"{where}: missing 'properties'")
        return
    for key, expected in (("environment", "python"), ("scriptSource", "inline")):
        if props.get(key) != expected:
            problems.append(f"{where}: properties.{key} must be {expected!r} (found {props.get(key)!r})")
    if str(props.get("successCodes", "")).strip() != "0":
        problems.append(f"{where}: properties.successCodes must be '0' so any failure stops the workflow")
    command = props.get("command")
    if not isinstance(command, str) or not command.strip():
        problems.append(f"{where}: properties.command (inline Python) is missing")
        return
    try:
        compile(command, f"<{name}>", "exec")
    except SyntaxError as exc:
        problems.append(f"{where}: inline Python does not compile (line {exc.lineno}: {exc.msg})")
        return

    step = _STEP_RE.search(command)
    if not step:
        problems.append(f"{where}: script has no step='...' line")
    elif step.group("step") != name:
        problems.append(f"{where}: script says step={step.group('step')!r}; it must match the task name")

    root = _ROOT_RE.search(command)
    if not root:
        problems.append(f"{where}: script has no root=pathlib.Path(r\"...\") line")
    else:
        roots[name] = root.group("root")

    for needle, what in (("logs=root/'logs'", "write its log under <root>/logs"),
                         ("marker=logs/'intouch_current_run.json'", "use the shared run marker"),
                         ("log=logs/('intouch_'+run_id+'.log')", "log to logs/intouch_<run-id>.log"),
                         ("if step=='collect-jobs':", "create the run marker only in collect-jobs"),
                         ("sys.exit(rc)", "exit with the step's return code")):
        if needle not in command:
            problems.append(f"{where}: script must {what} (missing {needle!r})")

    timeout = props.get("timeoutSeconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        problems.append(f"{where}: properties.timeoutSeconds must be a positive integer")
        timeout = None

    if name in PIPELINE_STEPS:
        _check_pipeline_command(name, command, timeout, roots.get(name), problems)
    elif name == "verify-report":
        if "report=root/'reports'/'job_intelligence.html'" not in command:
            problems.append(f"{where}: must check <root>/reports/job_intelligence.html (the report's output path)")
        if "if stat.st_size<=0:raise" not in command:
            problems.append(f"{where}: must fail when the report is empty")
    elif name == "email-confirmed":
        if "EMAIL_TASK_RETURNED_SUCCESS" not in command:
            problems.append(f"{where}: must log EMAIL_TASK_RETURNED_SUCCESS")

    for key in CREDENTIAL_KEYS:
        if props.get(key):
            problems.append(f"{where}: properties.{key} is set; keep credentials in InTouch or .env, not in the file")


def _check_pipeline_command(name: str, command: str, timeout: int | None, root: str | None,
                            problems: list[str]) -> None:
    where = f"task {name!r}"
    module, required = PIPELINE_STEPS[name]
    args = _parse_args_line(command)
    if args is None:
        problems.append(f"{where}: no readable args=[...] command list")
        return
    if len(args) < 3 or args[1:3] != ["-m", module]:
        problems.append(f"{where}: must run '<python> -m {module}' (found {args[1:3]})")
        return
    if root is not None and not _same_path(args[0], _win(root).joinpath(*VENV_PYTHON_RELATIVE)):
        problems.append(f"{where}: Python must be the project's virtualenv "
                        f"{_win(root).joinpath(*VENV_PYTHON_RELATIVE)} (found {args[0]})")
    flags = args[3:]
    for flag in required:
        if flag not in flags:
            problems.append(f"{where}: missing required flag {flag}")
    complaint = _cli_problem(module, flags)
    if complaint:
        problems.append(f"{where}: {module} rejects its arguments: {complaint}")

    if "--profile" in flags:
        index = flags.index("--profile")
        value = flags[index + 1] if index + 1 < len(flags) else None
        if value is None or value.replace("\\", "/") != SANITIZED_PROFILE:
            problems.append(f"{where}: --profile must be {SANITIZED_PROFILE} (the sanitized profile; found {value!r})")
    if name == "collect-jobs" and "--db" in flags:
        index = flags.index("--db")
        value = flags[index + 1] if index + 1 < len(flags) else None
        expected = _win(root).joinpath(*DB_RELATIVE) if root else None
        ok = value is not None and (
            (expected is not None and _same_path(value, expected))
            or value.replace("\\", "/") == "/".join(DB_RELATIVE))
        if not ok:
            problems.append(f"{where}: --db must be {expected or 'data/jobs.db'}, the database the evaluator and "
                            f"report read (found {value!r})")

    wait = _WAIT_RE.search(command)
    if not wait:
        problems.append(f"{where}: subprocess has no p.wait(timeout=...) limit")
    elif timeout is not None and int(wait.group("wait")) >= timeout:
        problems.append(f"{where}: timeoutSeconds ({timeout}) must exceed the inner subprocess timeout "
                        f"({wait.group('wait')}) so the script can log the timeout")


def _check_message_task(task: dict[str, Any], roots: dict[str, str], problems: list[str]) -> None:
    where = "task 'send-report-email'"
    if task.get("tool") != "message":
        problems.append(f"{where}: tool must be 'message' (found {task.get('tool')!r})")
    props = task.get("properties")
    if not isinstance(props, dict):
        problems.append(f"{where}: missing 'properties'")
        return
    recipient = props.get("to")
    if not isinstance(recipient, str) or not EMAIL_PATTERN.search(recipient):
        problems.append(f"{where}: 'to' must contain an email address")
    if not isinstance(props.get("subject"), str) or not props["subject"].strip():
        problems.append(f"{where}: 'subject' is missing")
    attachment = props.get("attachments")
    root = next(iter(roots.values()), None)
    if not isinstance(attachment, str) or not attachment.strip():
        problems.append(f"{where}: 'attachments' must be the report path")
    elif root is not None and not _same_path(attachment, _win(root).joinpath(*REPORT_RELATIVE)):
        problems.append(f"{where}: 'attachments' must be {_win(root).joinpath(*REPORT_RELATIVE)} "
                        f"(found {attachment})")


def _check_template(data: dict[str, Any], roots: dict[str, str], problems: list[str]) -> None:
    for name, root in roots.items():
        if not _same_path(root, PLACEHOLDER_ROOT):
            problems.append(f"template: task {name!r} root must be the placeholder {PLACEHOLDER_ROOT} (found {root})")
    text = json.dumps(data, ensure_ascii=False)
    for address in sorted(set(EMAIL_PATTERN.findall(text))):
        if not address.lower().endswith(PLACEHOLDER_EMAIL_DOMAIN):
            problems.append(f"template: real-looking email address {address!r}; use a {PLACEHOLDER_EMAIL_DOMAIN} placeholder")
    if re.search(r"[A-Za-z]:\\\\Users\\\\|/Users/|/home/", text):
        problems.append("template: contains a personal user directory path")
    decoded = "\n".join(t.get("properties", {}).get("command", "") for t in data.get("tasks", [])
                        if isinstance(t, dict) and isinstance(t.get("properties"), dict))
    for line in decoded.splitlines():
        if line.lstrip().startswith(("root=", "args=")) and PLACEHOLDER_ROOT.lower() not in line.lower().replace("\\\\", "\\"):
            problems.append(f"template: path not replaced by the placeholder: {line.strip()[:90]}")


def check_workflow(data: Any, *, template: bool = False) -> list[str]:
    """Return a list of human-readable problems (empty when the workflow has everything)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["workflow must be a JSON object"]
    for key in ("name", "description"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"workflow {key!r} is missing")
    if "schedule" in data:
        problems.append("workflow has a 'schedule'; configure it in InTouch after a validated test run")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return problems + ["workflow 'tasks' list is missing or empty"]

    names = [t.get("name") if isinstance(t, dict) else None for t in tasks]
    for name in TASK_ORDER:
        count = names.count(name)
        if count == 0:
            problems.append(f"missing task {name!r}")
        elif count > 1:
            problems.append(f"task {name!r} appears {count} times")
    present = [n for n in names if n in TASK_ORDER]
    if present != sorted(present, key=TASK_ORDER.index):
        problems.append(f"tasks are out of order: expected {', '.join(TASK_ORDER)}")
    if names and names[0] != "collect-jobs":
        problems.append("the first task must be 'collect-jobs' (it creates the run marker the others need)")
    extra = [n for n in names if n not in TASK_ORDER]
    if extra:
        problems.append(f"unexpected task(s): {extra}")

    roots: dict[str, str] = {}
    for task in tasks:
        if not isinstance(task, dict):
            problems.append("a task entry is not an object")
            continue
        name = task.get("name")
        if name in RUN_TASKS:
            if task.get("tool") != "run_command":
                problems.append(f"task {name!r}: tool must be 'run_command' (found {task.get('tool')!r})")
                continue
            _check_run_task(task, name, roots, problems)
        elif name == "send-report-email":
            _check_message_task(task, roots, problems)

    if len({str(_win(r)).casefold() for r in roots.values()}) > 1:
        problems.append(f"tasks use different project roots: {sorted(set(roots.values()))}")

    text = json.dumps(data, ensure_ascii=False)
    if SECRET_PATTERN.search(text):
        problems.append("workflow appears to contain a secret (API key or password); keep secrets in .env or InTouch")

    if template:
        _check_template(data, roots, problems)
    return problems


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.intouch_workflow",
                                description="Check an InTouch workflow (.iml) for everything the pipeline needs.")
    p.add_argument("path", nargs="?", type=Path, default=TEMPLATE_PATH,
                   help="workflow file (default: the public template)")
    p.add_argument("--template", action="store_true",
                   help="also require placeholders only (automatic for *.example.iml)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    template = args.template or args.path.name.endswith(".example.iml")
    try:
        data = load_workflow(args.path)
    except WorkflowFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    problems = check_workflow(data, template=template)
    if problems:
        print(f"{args.path}: {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    kind = "template" if template else "workflow"
    print(f"{args.path}: OK ({kind}; {len(data['tasks'])} tasks, all required steps present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
