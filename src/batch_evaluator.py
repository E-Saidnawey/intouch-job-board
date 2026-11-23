"""Batch Claude evaluation of every eligible job (geographically eligible + HIGH/MEDIUM title relevance).

Reuses the single-job evaluator (``src.claude_evaluator``): the same prompt, structured
output schema, validation, location policy and the same SQLite evaluation cache.

* Dry run by default: lists the jobs that would be evaluated and makes no request.
  The dry run opens the database read-only and never creates or migrates anything.
* ``--call-api`` evaluates each new or changed posting with one request per job, one
  job at a time. Jobs with a current cached evaluation are skipped (no request).
* Each successful, validated evaluation is appended to the ``evaluations`` table as it
  arrives; existing rows are never updated or deleted. Failures are never stored.
* A failed evaluation is logged and the batch continues with the next job. The batch
  stops early only when continuing cannot help: an authentication / permission /
  unknown-model error, or ``--max-consecutive-failures`` failures in a row.
* The profile path is explicit; ``config/candidate_profile.json`` (the real profile) is
  refused. Use the sanitized ``config/claude_candidate_profile.json``.
* Nothing is submitted, sent or contacted other than the Claude API itself.
  The API key, profile contents, prompts and job descriptions are never printed.

    python -m src.batch_evaluator --profile config/claude_candidate_profile.json             # dry run
    python -m src.batch_evaluator --profile config/claude_candidate_profile.json --call-api  # evaluate

Exit codes: 0 success (or dry run ready), 1 one or more evaluations failed / batch
stopped / dry run not ready, 2 invalid arguments, profile or database, 130 interrupted.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import database, location_filter, relevance_filter
from src.candidate_profile import ProfileError, load_profile
from src.claude_evaluator import (
    DEFAULT_DB, DEFAULT_EFFORT, DEFAULT_MAX_RETRIES, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS, EFFORT_LEVELS,
    EVALUATION_SCHEMA_VERSION, MAX_RETRIES_CAP, MODEL_ENV_VAR, PROJECT_ROOT, ApiAccessError, EvaluatorConfig,
    EvaluatorError, EvaluatorInputError, _is_real_profile, _usable, load_api_key, make_cache_key, make_client,
    request_evaluation, resolve_model, validate_job_record,
)
from src.evaluation_store import EvaluationStoreError, ensure_schema, find_cached, save_evaluation

ELIGIBLE_LOCATIONS = relevance_filter.ELIGIBLE_LOCATIONS          # compatible, review, boston_hybrid
ELIGIBLE_RELEVANCE = (relevance_filter.HIGH, relevance_filter.MEDIUM)
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

# Reasons a job is skipped without any API request.
SKIP_LOCATION = "location not eligible"
SKIP_RELEVANCE = "low title relevance"
SKIP_INVALID = "incomplete job record"


@dataclass
class Candidate:
    job: dict[str, Any]
    location_class: str
    relevance: str
    relevance_score: float

    @property
    def label(self) -> str:
        job = self.job
        return f"{job.get('company')} | {job.get('title')} | {job.get('location') or 'no location'}"


@dataclass
class BatchPlan:
    pending: list[Candidate] = field(default_factory=list)   # need an API request
    cached: list[Candidate] = field(default_factory=list)    # current cached evaluation exists
    skipped: Counter = field(default_factory=Counter)        # reason -> count
    active_jobs: int = 0

    @property
    def skipped_total(self) -> int:
        return sum(self.skipped.values())


@dataclass
class BatchResult:
    evaluated: int = 0
    cached: int = 0
    failed: int = 0
    skipped: int = 0
    not_attempted: int = 0
    stopped_reason: str | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)  # (job label, safe error message)


# --------------------------------------------------------------------------- #
# Selection (read-only)
# --------------------------------------------------------------------------- #

def load_active_jobs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT j.id AS job_id, c.name AS company, c.ats_provider, j.ats_job_id, j.title, j.location,
               j.description, j.application_url
        FROM jobs j JOIN companies c ON c.id = j.company_id
        WHERE j.is_active = 1
        ORDER BY c.name, j.title, j.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def select_eligible(jobs: list[dict[str, Any]]) -> tuple[list[Candidate], Counter]:
    """Geographically eligible jobs with HIGH or MEDIUM title relevance, best first."""
    skipped: Counter = Counter()
    selected: list[Candidate] = []
    for job in jobs:
        loc = location_filter.classify_location(job.get("location"))["classification"]
        if loc not in ELIGIBLE_LOCATIONS:
            skipped[SKIP_LOCATION] += 1
            continue
        rel = relevance_filter.classify_relevance(job.get("title"))
        if rel["classification"] not in ELIGIBLE_RELEVANCE:
            skipped[SKIP_RELEVANCE] += 1
            continue
        selected.append(Candidate(job, loc, rel["classification"], rel["score"]))
    selected.sort(key=lambda c: (ELIGIBLE_RELEVANCE.index(c.relevance), -c.relevance_score,
                                 str(c.job.get("company")), str(c.job.get("title")), c.job["job_id"]))
    return selected, skipped


def plan_batch(conn: sqlite3.Connection, profile: dict[str, Any], config: EvaluatorConfig,
               limit: int | None = None) -> BatchPlan:
    """Split eligible jobs into cached / pending / skipped. Read-only (works on a ``mode=ro`` connection)."""
    jobs = load_active_jobs(conn)
    selected, skipped = select_eligible(jobs)
    plan = BatchPlan(skipped=skipped, active_jobs=len(jobs))
    for cand in selected:
        try:
            validate_job_record(cand.job)
        except EvaluatorInputError:
            plan.skipped[SKIP_INVALID] += 1
            continue
        if _usable(find_cached(conn, make_cache_key(cand.job, profile, config))) is not None:
            plan.cached.append(cand)
        else:
            plan.pending.append(cand)
    if limit is not None:
        plan.pending = plan.pending[:limit]
    return plan


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def run_batch(conn: sqlite3.Connection, plan: BatchPlan, profile: dict[str, Any], config: EvaluatorConfig,
              *, client_factory, api_key: str, max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
              out=None) -> BatchResult:
    """Evaluate ``plan.pending`` one job at a time on a writable connection.

    Re-checks the cache right before each request, stores each validated evaluation
    immediately (append-only) and continues past individual failures.
    """
    out = out or sys.stdout
    ensure_schema(conn)
    result = BatchResult(cached=len(plan.cached), skipped=plan.skipped_total)
    client = None
    consecutive = 0
    total = len(plan.pending)
    for index, cand in enumerate(plan.pending, start=1):
        prefix = f"[{index}/{total}]"
        key = make_cache_key(cand.job, profile, config)
        if _usable(find_cached(conn, key)) is not None:   # stored meanwhile (e.g. by another run)
            result.cached += 1
            print(f"{prefix} cached     {cand.label}", file=out)
            continue
        try:
            if client is None:
                client = client_factory(api_key, config)
            evaluation, response_model = request_evaluation(
                cand.job, profile, client=client, config=config, allow_api_call=True)
        except ApiAccessError as exc:
            result.failed += 1
            result.failures.append((cand.label, str(exc)))
            result.not_attempted = total - index
            result.stopped_reason = f"stopped: {exc}"
            print(f"{prefix} FAILED     {cand.label}: {exc}", file=out)
            break
        except EvaluatorError as exc:
            result.failed += 1
            result.failures.append((cand.label, str(exc)))
            print(f"{prefix} FAILED     {cand.label}: {exc}", file=out)
            consecutive += 1
            if consecutive >= max_consecutive_failures:
                result.not_attempted = total - index
                result.stopped_reason = f"stopped after {consecutive} consecutive failures"
                break
            continue
        try:
            stored = save_evaluation(conn, key, evaluation, response_model)
        except (EvaluationStoreError, sqlite3.Error) as exc:
            result.failed += 1
            message = f"evaluation succeeded but could not be saved: {exc}"
            result.failures.append((cand.label, message))
            print(f"{prefix} FAILED     {cand.label}: {message}", file=out)
            consecutive += 1
            if consecutive >= max_consecutive_failures:
                result.not_attempted = total - index
                result.stopped_reason = f"stopped after {consecutive} consecutive failures"
                break
            continue
        consecutive = 0
        result.evaluated += 1
        print(f"{prefix} evaluated  {cand.label} -> fit {evaluation['overall_fit_score']}/100 "
              f"(evaluation #{stored.id})", file=out)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.batch_evaluator",
        description="Evaluate every geographically eligible HIGH/MEDIUM-relevance job with Claude. "
                    "Dry run unless --call-api is given; jobs with current cached evaluations are skipped.",
    )
    p.add_argument("--profile", required=True, type=Path,
                   help="explicit path to the sanitized profile (e.g. config/claude_candidate_profile.json)")
    p.add_argument("--db", type=Path, default=DEFAULT_DB,
                   help=f"SQLite database (default {DEFAULT_DB.relative_to(PROJECT_ROOT)})")
    p.add_argument("--call-api", action="store_true",
                   help="send one Claude API request per new or changed eligible job")
    p.add_argument("--limit", type=_positive_int, default=None,
                   help="optional cap on API requests this run (default: no limit)")
    p.add_argument("--max-consecutive-failures", type=_positive_int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
                   help=f"stop after this many failures in a row (default {DEFAULT_MAX_CONSECUTIVE_FAILURES})")
    p.add_argument("--model", help=f"Claude model (default: ${MODEL_ENV_VAR} or {DEFAULT_MODEL})")
    p.add_argument("--effort", choices=EFFORT_LEVELS, default=DEFAULT_EFFORT)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="request timeout in seconds")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"SDK retries (0-{MAX_RETRIES_CAP})")
    p.add_argument("--max-description-chars", type=int, default=None,
                   help="optional cap on description length (default: send the full description)")
    return p


def _print_plan(plan: BatchPlan, config: EvaluatorConfig, out) -> None:
    print(f"Model: {config.model} (effort {config.effort}, schema v{EVALUATION_SCHEMA_VERSION})", file=out)
    print(f"Active jobs: {plan.active_jobs}", file=out)
    print(f"Eligible (location + HIGH/MEDIUM relevance): {len(plan.pending) + len(plan.cached)}", file=out)
    print(f"  already evaluated (cached): {len(plan.cached)}", file=out)
    print(f"  need evaluation:            {len(plan.pending)}", file=out)
    print(f"Skipped: {plan.skipped_total}" + "".join(
        f"\n  {reason}: {count}" for reason, count in sorted(plan.skipped.items())), file=out)


def _print_summary(result: BatchResult, out) -> None:
    print(f"\nSummary: evaluated={result.evaluated} cached={result.cached} failed={result.failed} "
          f"skipped={result.skipped} not_attempted={result.not_attempted}", file=out)
    if result.stopped_reason:
        print(f"Batch {result.stopped_reason}", file=out)


def main(argv: list[str] | None = None, *, client_factory=make_client) -> int:
    args = build_parser().parse_args(argv)
    out = sys.stdout

    if _is_real_profile(args.profile):
        print("error: refusing to use config/candidate_profile.json; "
              "use the sanitized config/claude_candidate_profile.json", file=sys.stderr)
        return 2
    if not args.db.is_file():
        print(f"error: database not found: {args.db}", file=sys.stderr)
        return 2
    try:
        config = EvaluatorConfig(
            model=resolve_model(args.model),
            effort=args.effort, timeout=args.timeout, max_retries=args.max_retries,
            max_description_chars=args.max_description_chars,
        )
        profile = load_profile(args.profile)
    except (ProfileError, EvaluatorInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        ro = location_filter.open_readonly(args.db)
        try:
            plan = plan_batch(ro, profile, config, args.limit)
        finally:
            ro.close()
    except (EvaluationStoreError, sqlite3.Error) as exc:
        print(f"error: database: {exc}", file=sys.stderr)
        return 2

    api_key = load_api_key() if plan.pending else None
    _print_plan(plan, config, out)

    if not args.call_api:
        if plan.pending:
            print("\nWould evaluate (one API request each):", file=out)
            for index, cand in enumerate(plan.pending, start=1):
                print(f"  {index:>3}. [{cand.relevance}] {cand.label} "
                      f"({cand.job.get('ats_provider')}:{cand.job.get('ats_job_id')})", file=out)
        ready = not plan.pending or api_key is not None
        if not ready:
            print("\nproblem: ANTHROPIC_API_KEY is not set (.env or environment)", file=out)
        print(f"\nDry run: no request was sent. Re-run with --call-api to make {len(plan.pending)} "
              f"API request(s).", file=out)
        return 0 if ready else 1

    if not plan.pending:
        _print_summary(BatchResult(cached=len(plan.cached), skipped=plan.skipped_total), out)
        return 0
    if api_key is None:
        print("error: ANTHROPIC_API_KEY is not set (.env or environment)", file=sys.stderr)
        return 2

    print(f"\nEvaluating {len(plan.pending)} job(s), one API request each...", file=out)
    conn = database.connect(args.db)
    result: BatchResult | None = None
    try:
        result = run_batch(conn, plan, profile, config, client_factory=client_factory, api_key=api_key,
                           max_consecutive_failures=args.max_consecutive_failures, out=out)
    except KeyboardInterrupt:
        print("\ninterrupted: evaluations completed so far are saved", file=sys.stderr)
        return 130
    except (EvaluationStoreError, sqlite3.Error) as exc:
        print(f"error: evaluation database: {exc}", file=sys.stderr)
        return 2
    finally:
        conn.close()

    _print_summary(result, out)
    return 1 if result.failed or result.stopped_reason else 0


if __name__ == "__main__":
    sys.exit(main())
