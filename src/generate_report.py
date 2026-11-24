"""Generate a standalone HTML Job Intelligence Report from the local SQLite database.

    python -m src.generate_report --profile config/claude_candidate_profile.json
    python -m src.generate_report                     # no profile: AI evaluations shown as unverified

Offline and read-only: the database is opened with ``mode=ro``; no ATS or Claude
requests are made. The output is one self-contained HTML file (embedded CSS and a
small vanilla-JS filter script, locked down with a hash-based Content-Security-Policy).

Which saved evaluations are shown as *current*
---------------------------------------------
An evaluation is current only when every part of its cache key matches the current
inputs: the job's description hash, normalized location, the model, the candidate
profile hash and the evaluation schema version. The model is ``--model`` or the
evaluator's default (``$CLAUDE_EVALUATOR_MODEL`` / built-in default); the profile
must be given with ``--profile`` and is only hashed locally, never written to the
report. Without ``--profile``, saved evaluations are marked *unverified* and are
neither scored nor ranked. Stale evaluations are never displayed as current.

Ranking: jobs with a current evaluation first, by overall fit score (highest
first); then every other job by title relevance (high, medium, low). Ties break
on relevance score, company and title. Unevaluated jobs never show an AI score.

Posting-age cutoff
------------------
Only jobs whose employer publication date (Greenhouse ``first_published``, Ashby
``publishedAt``) is within ``--max-age-days`` (default 30, boundary inclusive, UTC) are
rendered. Dates come from the latest collector snapshots (``--data-dir``, default the
database's directory) and then the database's ``source_published_at`` column. Jobs
with no verified publication date are excluded; ``updated_at`` and first/last-seen
times are never substituted. Exclusion counts are printed; the database is unchanged.

Exit codes: 0 report written, 2 invalid input (missing database, invalid profile).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src import database, location_filter, posting_dates, relevance_filter
from src.candidate_profile import ProfileError, load_profile
from src.claude_evaluator import (
    EVALUATION_SCHEMA_VERSION, apply_location_policy, resolve_model, validate_evaluation,
)
from src.evaluation_store import EVALUATIONS_TABLE, normalize_location, profile_sha256, sha256_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "jobs.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "job_intelligence.html"
DEFAULT_MAX_AGE_DAYS = posting_dates.DEFAULT_MAX_AGE_DAYS

ELIGIBLE_LOCATIONS = relevance_filter.ELIGIBLE_LOCATIONS  # compatible, review, boston_hybrid
RELEVANCE_ORDER = {relevance_filter.HIGH: 0, relevance_filter.MEDIUM: 1, relevance_filter.LOW: 2}

# Evaluation statuses
CURRENT = "current"            # matches every current input; shown and ranked by score
STALE = "stale"                # saved, but some input changed; not shown as current
UNVERIFIED = "unverified"      # saved, but no profile given, so it cannot be verified
NOT_EVALUATED = "not_evaluated"


class ReportError(Exception):
    """Report inputs are unusable."""


# --------------------------------------------------------------------------- #
# Data loading (read-only)
# --------------------------------------------------------------------------- #

def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    if not db_path.is_file():
        raise ReportError(f"database not found: {db_path.name}")
    return location_filter.open_readonly(db_path)


def load_active_jobs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    # Databases created before the publication-date column exist read-only here: no migration.
    columns = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    published = (f"j.{database.PUBLISHED_COLUMN}" if database.PUBLISHED_COLUMN in columns else "NULL")
    rows = conn.execute(
        f"""
        SELECT j.id AS job_id, c.name AS company, c.ats_provider, j.ats_job_id, j.title, j.location,
               j.description, j.application_url, {published} AS source_published_at
        FROM jobs j JOIN companies c ON c.id = j.company_id
        WHERE j.is_active = 1
        ORDER BY c.name, j.title, j.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def load_evaluations(conn: sqlite3.Connection) -> dict[int, list[dict[str, Any]]]:
    """All saved evaluations grouped by job, newest first. Empty if the table is absent."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (EVALUATIONS_TABLE,)
    ).fetchone()
    if not exists:
        return {}
    columns = {r[1] for r in conn.execute(f"PRAGMA table_info({EVALUATIONS_TABLE})")}
    location_col = "location_key" if "location_key" in columns else "NULL AS location_key"
    rows = conn.execute(
        f"""
        SELECT id, job_id, description_sha256, profile_sha256, model, response_model, schema_version,
               evaluation_json, evaluated_at, {location_col}
        FROM {EVALUATIONS_TABLE}
        ORDER BY evaluated_at DESC, id DESC
        """
    ).fetchall()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["job_id"], []).append(dict(row))
    return grouped


# --------------------------------------------------------------------------- #
# Assessment
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CurrentInputs:
    model: str
    schema_version: int
    profile_sha256: str | None  # None: profile not provided -> evaluations unverified


@dataclass
class JobEntry:
    job: dict[str, Any]
    location: dict[str, str]
    relevance: dict[str, Any]
    status: str = NOT_EVALUATED
    evaluation: dict[str, Any] | None = None    # post-location-policy, only when CURRENT
    evaluated_at: str | None = None
    stale_reasons: list[str] = field(default_factory=list)
    published_at: datetime | None = None         # verified employer publication date, UTC

    @property
    def uid(self) -> str:
        """Stable id for browser-side seen/unseen state: ATS provider | employer | posting id."""
        return "|".join(str(self.job.get(k) or "") for k in ("ats_provider", "company", "ats_job_id"))

    @property
    def eligible(self) -> bool:
        return self.location["classification"] in ELIGIBLE_LOCATIONS

    @property
    def score(self) -> int | None:
        return self.evaluation["overall_fit_score"] if self.evaluation else None

    @property
    def recommendation(self) -> str | None:
        return self.evaluation["recommendation"] if self.evaluation else None


def _parse_valid(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        data = json.loads(row["evaluation_json"])
    except (json.JSONDecodeError, TypeError):
        return None
    return data if not validate_evaluation(data) else None


def _mismatches(row: dict[str, Any], job: dict[str, Any], inputs: CurrentInputs) -> list[str]:
    reasons = []
    if row["description_sha256"] != sha256_text(job.get("description") or ""):
        reasons.append("job description changed")
    if row["location_key"] is None:
        reasons.append("location not recorded (older evaluation)")
    elif row["location_key"] != normalize_location(job.get("location")):
        reasons.append("job location changed")
    if row["model"] != inputs.model:
        reasons.append("different model")
    if inputs.profile_sha256 is not None and row["profile_sha256"] != inputs.profile_sha256:
        reasons.append("different candidate profile")
    if row["schema_version"] != inputs.schema_version:
        reasons.append("older evaluation schema version")
    return reasons


def assess_job(job: dict[str, Any], rows: list[dict[str, Any]], inputs: CurrentInputs) -> JobEntry:
    entry = JobEntry(
        job=job,
        location=location_filter.classify_location(job.get("location")),
        relevance=relevance_filter.classify_relevance(job.get("title")),
    )
    if not rows:
        return entry
    if inputs.profile_sha256 is None:
        entry.status = UNVERIFIED
        return entry
    for row in rows:  # newest first
        if not _mismatches(row, job, inputs):
            evaluation = _parse_valid(row)
            if evaluation is not None:
                entry.status = CURRENT
                entry.evaluation = apply_location_policy(evaluation, job.get("location"))
                entry.evaluated_at = row["evaluated_at"]
                return entry
    entry.status = STALE
    newest = rows[0]
    entry.stale_reasons = _mismatches(newest, job, inputs) or ["saved evaluation failed validation"]
    return entry


def rank(entries: list[JobEntry]) -> list[JobEntry]:
    """Current evaluations by score first; then everything else by title relevance."""
    def key(e: JobEntry):
        relevance = (RELEVANCE_ORDER.get(e.relevance["classification"], 3), -float(e.relevance.get("score") or 0))
        tail = ((e.job.get("company") or "").casefold(), (e.job.get("title") or "").casefold(), e.job["job_id"])
        if e.status == CURRENT:
            return (0, -e.score, *relevance, *tail)
        return (1, 0, *relevance, *tail)
    return sorted(entries, key=key)


def sort_newest_first(entries: list[JobEntry]) -> list[JobEntry]:
    """Newest publication first; ties keep their existing (ranked) order. Undated entries last."""
    return sorted(entries, key=lambda e: -(e.published_at.timestamp() if e.published_at else float("-inf")))


def resolve_published_at(job: dict[str, Any],
                         snapshot_dates: dict[posting_dates.SnapshotKey, str] | None) -> datetime | None:
    """The job's verified employer publication date (latest snapshot first, then the database).

    Only ``published_at`` / ``source_published_at`` are consulted; update and
    first/last-seen timestamps are never used as a substitute.
    """
    return posting_dates.resolve_job_published_at(job, snapshot_dates)


@dataclass
class ReportData:
    entries: list[JobEntry]              # eligible jobs within the posting-age window, ranked
    metrics: dict[str, int]
    inputs: CurrentInputs
    model_source: str
    generated_at: str
    max_age_days: int = DEFAULT_MAX_AGE_DAYS
    cutoff: datetime | None = None       # earliest publication instant included (UTC)


def build_report_data(conn: sqlite3.Connection, inputs: CurrentInputs, model_source: str = "",
                      generated_at: str | None = None, *, max_age_days: int = DEFAULT_MAX_AGE_DAYS,
                      now: datetime | None = None,
                      snapshot_dates: dict[posting_dates.SnapshotKey, str] | None = None) -> ReportData:
    """Assess active jobs and keep only eligible ones published within ``max_age_days``.

    The posting-age cutoff is applied here, before ranking and rendering: jobs older
    than the window, and jobs without a verified employer publication date, never
    reach the HTML. The database is only read; historical rows are untouched.
    ``now`` defaults to ``generated_at`` (if given) or the current UTC time.
    """
    if now is None:
        now = (datetime.fromisoformat(generated_at) if generated_at else datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ReportError("report reference time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    generated_at = generated_at or now.isoformat(timespec="seconds")

    jobs = load_active_jobs(conn)
    evaluations = load_evaluations(conn)
    age_counts = {posting_dates.RECENT: 0, posting_dates.TOO_OLD: 0, posting_dates.UNKNOWN: 0}
    recent_jobs = []
    published_by_job: dict[int, datetime] = {}
    for job in jobs:
        published = resolve_published_at(job, snapshot_dates)
        age = posting_dates.classify_age(published, now, max_age_days)
        age_counts[age] += 1
        if age == posting_dates.RECENT:
            recent_jobs.append(job)
            published_by_job[job["job_id"]] = published

    all_entries = []
    for job in recent_jobs:
        entry = assess_job(job, evaluations.get(job["job_id"], []), inputs)
        entry.published_at = published_by_job[job["job_id"]]
        all_entries.append(entry)
    eligible = [e for e in all_entries if e.eligible]
    metrics = {
        "active_jobs": len(jobs),
        "recent_jobs": age_counts[posting_dates.RECENT],
        "excluded_too_old": age_counts[posting_dates.TOO_OLD],
        "excluded_unknown_date": age_counts[posting_dates.UNKNOWN],
        "eligible_jobs": len(eligible),
        "relevance_high": sum(e.relevance["classification"] == relevance_filter.HIGH for e in eligible),
        "relevance_medium": sum(e.relevance["classification"] == relevance_filter.MEDIUM for e in eligible),
        "relevance_low": sum(e.relevance["classification"] == relevance_filter.LOW for e in eligible),
        "current_evaluations": sum(e.status == CURRENT for e in all_entries),
        "stale_evaluations": sum(e.status == STALE for e in all_entries),
        "unverified_evaluations": sum(e.status == UNVERIFIED for e in all_entries),
    }
    return ReportData(
        entries=rank(eligible), metrics=metrics, inputs=inputs, model_source=model_source,
        generated_at=generated_at, max_age_days=max_age_days,
        cutoff=posting_dates.cutoff_instant(now, max_age_days),
    )


# --------------------------------------------------------------------------- #
# HTML rendering (everything dynamic goes through esc())
# --------------------------------------------------------------------------- #

def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def safe_url(url: Any) -> str | None:
    """Only http(s) links are rendered; anything else (javascript:, data:, ...) is dropped."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return url


LOCATION_LABELS = {
    location_filter.COMPATIBLE: "Compatible",
    location_filter.BOSTON_HYBRID: "Boston hybrid",
    location_filter.REVIEW: "Needs review",
}
# Hover text (title attribute) explaining why a job shows "Not evaluated".
STATUS_HINTS = {
    NOT_EVALUATED: "No saved AI evaluation for this job.",
    STALE: "A saved evaluation exists but is stale",
    UNVERIFIED: "A saved evaluation exists but cannot be verified: re-run with --profile.",
}
MISSING_STATUS_LABELS = {"confirmed_missing": "Confirmed missing", "not_established": "Not established"}


def _ul(items: list[str]) -> str:
    if not items:
        return '<p class="muted small">None listed.</p>'
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def _block(title: str, body: str) -> str:
    return f'<div class="block"><h4>{esc(title)}</h4>{body}</div>'


def _link_html(job: dict[str, Any]) -> str:
    url = safe_url(job.get("application_url"))
    if not url:
        return '<span class="muted">No valid application link</span>'
    return f'<a class="apply" href="{esc(url)}" target="_blank" rel="noopener noreferrer">Application page</a>'


def _location_html(e: JobEntry) -> str:
    loc_class = e.location["classification"]
    return (f'<span>{esc(e.job.get("location") or "Location not listed")}</span>'
            f'<span class="badge loc-{esc(loc_class)}" title="{esc(e.location["reason"])}">'
            f'{esc(LOCATION_LABELS.get(loc_class, loc_class))}</span>')


def _relevance_badge(e: JobEntry) -> str:
    rel = e.relevance["classification"]
    return (f'<span class="badge rel-{esc(rel)}" title="{esc(e.relevance.get("reason"))}">'
            f'Title relevance: {esc(rel)}</span>')


def _details_html(e: JobEntry) -> str:
    """Closed-by-default native <details>: only the four qualification categories."""
    ev = e.evaluation
    responsibilities = _ul([esc(r) for r in ev["key_responsibilities"]])
    matched = _ul([esc(m["qualification"]) for m in ev["matched_qualifications"]])
    missing = _ul([
        f'{esc(m["qualification"])} <span class="tag st-{esc(m["status"])}">'
        f'{esc(MISSING_STATUS_LABELS.get(m["status"], m["status"]))}</span>'
        for m in ev["missing_qualifications"]
    ])
    transferable = _ul([esc(t["skill"]) for t in ev["transferable_skills"]])
    return f"""
      <details>
        <summary>More details</summary>
        <div class="facts">
          {_block("Key responsibilities", responsibilities)}
          {_block("Matched skills", matched)}
          {_block("Missing qualifications", missing)}
          {_block("Transferable skills", transferable)}
        </div>
      </details>"""


def _days_ago(published: datetime, now: datetime) -> str:
    days = (now.date() - published.date()).days
    if days <= 0:
        return "today"
    return "1 day ago" if days == 1 else f"{days} days ago"


def _posted_html(e: JobEntry, now: datetime | None) -> str:
    if e.published_at is None:  # never rendered: the cutoff removes undated jobs first
        return '<span class="posted muted">Posting date unknown</span>'
    day = e.published_at.strftime("%Y-%m-%d")
    ago = f" · {_days_ago(e.published_at, now)}" if now else ""
    return (f'<span class="posted" title="Employer publication date (UTC)">Posted '
            f'<time datetime="{esc(e.published_at.isoformat(timespec="seconds"))}">{esc(day)}</time>{esc(ago)}</span>')


def _card(rank_no: int, e: JobEntry, match_rank: int | None = None, now: datetime | None = None) -> str:
    """One compact card layout for every job.

    Visible: position, company, title, AI fit score (or "Not evaluated"), posting date,
    location and compatibility, title relevance, application link and a Mark as seen
    toggle. Evaluated jobs add a closed "More details" section. The recommendation is
    used only as a filter attribute.
    """
    job = e.job
    loc_class = e.location["classification"]
    rel_class = e.relevance["classification"]
    recommendation = e.recommendation or "none"
    match_rank = rank_no if match_rank is None else match_rank
    published_ms = int(e.published_at.timestamp() * 1000) if e.published_at else 0
    if e.status == CURRENT:
        ai = f'<span class="ai score">{esc(e.score)}<small>/100 fit</small></span>'
        details = _details_html(e)
    else:
        hint = STATUS_HINTS[e.status]
        if e.stale_reasons:
            hint = f"{hint}: {'; '.join(e.stale_reasons)}."
        ai = f'<span class="ai none" title="{esc(hint)}">Not evaluated</span>'
        details = ""
    search = " ".join(str(job.get(k) or "") for k in ("company", "title")).casefold()
    return f"""
    <article class="job" data-uid="{esc(e.uid)}" data-company="{esc(job.get("company"))}"
             data-location="{esc(loc_class)}" data-location-text="{esc(job.get("location") or "")}"
             data-relevance="{esc(rel_class)}" data-recommendation="{esc(recommendation)}"
             data-score="{esc(e.score if e.score is not None else "")}" data-published="{published_ms}"
             data-match-rank="{match_rank}" data-search="{esc(search)}">
      <div class="head">
        <span class="rank">#{rank_no}</span>
        <div class="titles">
          <p class="company">{esc(job.get("company"))}</p>
          <h3>{esc(job.get("title"))}</h3>
        </div>
        <div class="ai-box">{ai}</div>
      </div>
      <div class="meta">{_posted_html(e, now)}{_location_html(e)}{_relevance_badge(e)}{_link_html(job)}<button type="button" class="seen-toggle" aria-pressed="false">Mark as seen</button></div>{details}
    </article>"""


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2330;--muted:#667085;--border:#d9dee7;--accent:#1f5fbf;
--good:#16794c;--warn:#a15c00;--bad:#b42318;--chip:#eef2f7}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--text:#e6e9ef;--muted:#9aa4b2;
--border:#2c3441;--accent:#7fb0ff;--good:#5fd39a;--warn:#f0b35a;--bad:#ff8a80;--chip:#252c38}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--text)}main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.15rem;margin:1.5em 0 .5em}h3{font-size:1.05rem;margin:0}
h4{font-size:.9rem;margin:1em 0 .2em}.muted{color:var(--muted)}.small{font-size:.85rem}
.inputs{font-size:.85rem;color:var(--muted)}.warn{color:var(--warn)}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0}
.metric{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.metric b{display:block;font-size:1.4rem}.metric span{font-size:.8rem;color:var(--muted)}
.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;background:var(--card);
border:1px solid var(--border);border-radius:10px;padding:10px}.filters .search{grid-column:1/-1}
.filters label{display:flex;flex-direction:column;gap:2px;font-size:.75rem;color:var(--muted)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;margin:10px 0;
position:sticky;top:0;z-index:1;background:var(--bg);padding:6px 0}
.views{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.views button{border:0;border-right:1px solid var(--border);background:var(--card);color:var(--text);
padding:6px 12px;font:inherit;cursor:pointer}.views button:last-child{border-right:0}
.views button[aria-pressed="true"]{background:var(--accent);color:var(--card)}
.views .count{font-size:.8rem;opacity:.85;margin-left:4px}
.toolbar .sort{display:flex;gap:6px;align-items:center;font-size:.85rem;color:var(--muted)}
.toolbar select,.reset{padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--card);
color:var(--text);font:inherit;cursor:pointer}
.posted{font-weight:600}
.seen-toggle{margin-left:auto;border:1px solid var(--border);border-radius:6px;background:var(--card);
color:var(--accent);padding:2px 10px;font:inherit;font-size:.85rem;cursor:pointer}
.job.is-seen{opacity:.6}.job.is-seen .seen-toggle{color:var(--muted)}
.filters input,.filters select{width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;
background:var(--bg);color:var(--text);font:inherit}
.job{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 14px;margin:8px 0}
.head{display:flex;gap:12px;align-items:flex-start}.rank{color:var(--muted);font-weight:600;min-width:2.5em}
.titles{flex:1;min-width:0}.company{margin:0;color:var(--muted)}.ai-box{text-align:right;display:flex;
flex-direction:column;gap:4px;align-items:flex-end}.ai.score{font-weight:700;font-size:1.05rem}
.ai.none{color:var(--muted);font-size:.85rem;cursor:help}
.meta{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin-top:6px;font-size:.9rem}
.badge,.tag{background:var(--chip);border-radius:999px;padding:1px 9px;font-size:.8rem;white-space:nowrap}
.loc-compatible,.rel-high{color:var(--good)}.loc-review,.rel-medium{color:var(--warn)}
.rel-low{color:var(--bad)}.apply{color:var(--accent)}
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:.9rem}
ul{padding-left:1.2em;margin:.3em 0}li{margin:.1em 0}
.empty{color:var(--muted);font-style:italic}
.company{font-size:.85rem}.ai.score{font-size:1.4rem;line-height:1}
.ai.score small{font-size:.75rem;font-weight:400;color:var(--muted);margin-left:2px}
.block h4{margin:.6em 0 .15em;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.block ul{margin:0}
.facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 20px}
.st-confirmed_missing{color:var(--bad)}.st-not_established{color:var(--muted)}
@media (max-width:700px){.facts{grid-template-columns:1fr}.filters{grid-template-columns:1fr 1fr}.toolbar{position:static}
.head{flex-wrap:wrap}.ai-box{width:100%;align-items:flex-start;text-align:left;flex-direction:row;flex-wrap:wrap}}
"""

JS = """
(function () {
  var STORAGE_KEY = 'jobIntelligence.seenJobs.v1';
  var seen = {};
  try {
    var parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}');
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) { seen = parsed; }
  } catch (e) { seen = {}; }
  function persist() {
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(seen)); } catch (e) { /* storage unavailable */ }
  }
  function byId(id) { return document.getElementById(id); }
  var list = byId('job-list');
  var cards = Array.prototype.slice.call(list.querySelectorAll('article.job'));
  var q = byId('q');
  var selects = { company: byId('f-company'), locationText: byId('f-location-text'), location: byId('f-location'),
                  relevance: byId('f-relevance'), recommendation: byId('f-recommendation') };
  var minScore = byId('f-min-score');
  var sortSel = byId('f-sort');
  var viewButtons = Array.prototype.slice.call(document.querySelectorAll('button[data-view]'));
  var view = 'unseen';

  function isSeen(card) { return Object.prototype.hasOwnProperty.call(seen, card.dataset.uid); }
  function paintSeen(card) {
    var s = isSeen(card);
    card.classList.toggle('is-seen', s);
    var b = card.querySelector('.seen-toggle');
    b.textContent = s ? 'Mark as unseen' : 'Mark as seen';
    b.setAttribute('aria-pressed', s ? 'true' : 'false');
  }
  function matchesFilters(card) {
    var text = q.value.trim().toLowerCase();
    if (text && card.dataset.search.indexOf(text) === -1) { return false; }
    for (var key in selects) {
      var v = selects[key].value;
      if (v && card.dataset[key] !== v) { return false; }
    }
    var min = minScore.value;
    if (min) {
      if (card.dataset.score === '' || Number(card.dataset.score) < Number(min)) { return false; }
    }
    return true;
  }
  function num(card, key) { return Number(card.dataset[key]) || 0; }
  function sortCards() {
    var mode = sortSel.value;
    cards.sort(function (a, b) {
      var d = 0;
      if (mode === 'oldest') { d = num(a, 'published') - num(b, 'published'); }
      else if (mode === 'match') { d = num(a, 'matchRank') - num(b, 'matchRank'); }
      else { d = num(b, 'published') - num(a, 'published'); }
      return d || (num(a, 'matchRank') - num(b, 'matchRank'));
    });
    cards.forEach(function (card) { list.appendChild(card); });
  }
  function apply() {
    var counts = { all: 0, unseen: 0, seen: 0 };
    var shown = 0;
    cards.forEach(function (card) {
      var ok = matchesFilters(card);
      var s = isSeen(card);
      if (ok) { counts.all++; counts[s ? 'seen' : 'unseen']++; }
      var visible = ok && (view === 'all' || (view === 'seen') === s);
      card.hidden = !visible;
      if (visible) { shown++; card.querySelector('.rank').textContent = '#' + shown; }
    });
    viewButtons.forEach(function (b) {
      var v = b.dataset.view;
      b.setAttribute('aria-pressed', v === view ? 'true' : 'false');
      b.querySelector('.count').textContent = String(counts[v]);
    });
    byId('visible-count').textContent = String(shown);
    byId('empty').hidden = shown !== 0;
  }
  function resort() { sortCards(); apply(); }

  list.addEventListener('click', function (ev) {
    var button = ev.target.closest('.seen-toggle');
    if (!button) { return; }
    var card = button.closest('article.job');
    if (isSeen(card)) { delete seen[card.dataset.uid]; } else { seen[card.dataset.uid] = new Date().toISOString(); }
    persist();
    paintSeen(card);
    apply();
  });
  viewButtons.forEach(function (b) {
    b.addEventListener('click', function () { view = b.dataset.view; apply(); });
  });
  q.addEventListener('input', apply);
  minScore.addEventListener('change', apply);
  Object.keys(selects).forEach(function (k) { selects[k].addEventListener('change', apply); });
  sortSel.addEventListener('change', resort);
  byId('reset-filters').addEventListener('click', function () {
    q.value = '';
    Object.keys(selects).forEach(function (k) { selects[k].value = ''; });
    minScore.value = '';
    sortSel.value = 'newest';
    resort();
  });
  cards.forEach(paintSeen);
  resort();
})();
"""


def _options(values: list[tuple[str, str]], all_label: str) -> str:
    opts = [f'<option value="">{esc(all_label)}</option>']
    opts += [f'<option value="{esc(v)}">{esc(label)}</option>' for v, label in values]
    return "".join(opts)


def render_html(data: ReportData) -> str:
    """Render the report. Cards start newest-first (the default sort); the script then
    applies the default Unseen view and re-sorts / filters in the browser."""
    now = datetime.fromisoformat(data.generated_at) if data.generated_at else None
    if now is not None and now.tzinfo is None:
        now = None
    match_rank = {id(e): i for i, e in enumerate(data.entries, 1)}   # data.entries is ranked by match
    ordered = sort_newest_first(data.entries)
    cards = "".join(_card(i, e, match_rank[id(e)], now) for i, e in enumerate(ordered, 1))
    m = data.metrics
    metric_items = [
        ("Active jobs", m["active_jobs"]), (f"Published in last {data.max_age_days} days", m["recent_jobs"]),
        ("Geographically eligible", m["eligible_jobs"]),
        ("High relevance", m["relevance_high"]), ("Medium relevance", m["relevance_medium"]),
        ("Low relevance", m["relevance_low"]), ("Current AI evaluations", m["current_evaluations"]),
    ]
    metrics_html = "".join(f'<div class="metric"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in metric_items)

    inputs = data.inputs
    if inputs.profile_sha256:
        profile_line = f"candidate profile fingerprint {esc(inputs.profile_sha256[:12])}"
    else:
        profile_line = ('<span class="warn">no candidate profile given: saved AI evaluations cannot be verified '
                        'and are not shown (re-run with --profile)</span>')
    notes = []
    if m["stale_evaluations"]:
        notes.append(f'{m["stale_evaluations"]} job(s) have only stale saved evaluations (inputs changed); '
                     "they are listed as not evaluated.")
    if m["unverified_evaluations"]:
        notes.append(f'{m["unverified_evaluations"]} job(s) have unverified saved evaluations.')
    notes_html = "".join(f'<p class="inputs warn">{esc(n)}</p>' for n in notes)
    cutoff_day = data.cutoff.strftime("%Y-%m-%d %H:%M UTC") if data.cutoff else "n/a"
    cutoff_line = (f"Only jobs the employer published on or after {cutoff_day} (last {data.max_age_days} days) are "
                   f"included. Excluded before rendering: {m['excluded_too_old']} published earlier, "
                   f"{m['excluded_unknown_date']} with a missing or unverified publication date.")

    companies = sorted({e.job.get("company") or "" for e in data.entries}, key=str.casefold)
    company_opts = _options([(c, c) for c in companies], "All companies")
    places = sorted({e.job.get("location") or "" for e in data.entries} - {""}, key=str.casefold)
    place_opts = _options([(c, c) for c in places], "All locations")
    location_opts = _options([(c, LOCATION_LABELS[c]) for c in ELIGIBLE_LOCATIONS], "All location types")
    relevance_opts = _options([(c, c.capitalize()) for c in relevance_filter.CLASSIFICATIONS], "All relevance")
    rec_opts = _options([("Apply", "Apply"), ("Review", "Review"), ("Skip", "Skip"), ("none", "Not evaluated")],
                        "All AI recommendations")
    score_opts = _options([(str(v), f"{v}+") for v in (50, 60, 70, 80, 90)], "Any match score")
    total = len(data.entries)

    script_hash = base64.b64encode(hashlib.sha256(JS.encode("utf-8")).digest()).decode("ascii")
    csp = (f"default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; "
           "img-src 'none'; base-uri 'none'; form-action 'none'")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{esc(csp)}">
<meta name="referrer" content="no-referrer">
<title>Job Intelligence Report</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <h1>Job Intelligence Report</h1>
  <p class="inputs">Generated {esc(data.generated_at)} · model {esc(inputs.model)}{esc(data.model_source)} ·
     evaluation schema v{esc(inputs.schema_version)} · {profile_line}</p>
  <p class="inputs">AI scores are heuristic fit assessments, not hiring probabilities. An evaluation is shown only
     when its job description, location, model, profile and schema version all match the current inputs.</p>
  <p class="inputs" id="cutoff-note">{esc(cutoff_line)}</p>
  {notes_html}
  <div class="metrics">{metrics_html}</div>
  <div class="filters" role="search">
    <input id="q" class="search" type="search" placeholder="Search company or job title" aria-label="Search company or job title">
    <select id="f-company" aria-label="Company">{company_opts}</select>
    <select id="f-location-text" aria-label="Location">{place_opts}</select>
    <select id="f-location" aria-label="Location classification">{location_opts}</select>
    <select id="f-relevance" aria-label="Title relevance">{relevance_opts}</select>
    <select id="f-recommendation" aria-label="AI recommendation">{rec_opts}</select>
    <select id="f-min-score" aria-label="Minimum match score">{score_opts}</select>
    <button type="button" id="reset-filters" class="reset">Clear filters</button>
  </div>
  <div class="toolbar">
    <div class="views" role="group" aria-label="Seen status">
      <button type="button" data-view="all" aria-pressed="false">All jobs<span class="count">{total}</span></button>
      <button type="button" data-view="unseen" aria-pressed="true">Unseen<span class="count">{total}</span></button>
      <button type="button" data-view="seen" aria-pressed="false">Seen<span class="count">0</span></button>
    </div>
    <label class="sort">Sort
      <select id="f-sort" aria-label="Sort order">
        <option value="newest" selected>Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="match">Highest match</option>
      </select>
    </label>
  </div>
  <p class="muted small"><span id="visible-count">{total}</span> of {total} eligible jobs shown.
     Incompatible locations are excluded. Seen status is saved in this browser only.</p>
  <section class="group" id="job-list" aria-label="Jobs">
    {cards}
  </section>
  <p class="empty" id="empty"{"" if not total else " hidden"}>No jobs match the current view and filters.</p>
</main>
<script>{JS}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def write_report(text: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, output)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.generate_report",
                                description="Write an offline HTML job report from the local database (read-only).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database (default data/jobs.db)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="output HTML file (default reports/job_intelligence.html)")
    p.add_argument("--profile", type=Path,
                   help="candidate profile used for evaluations (hashed locally, never included); "
                        "without it saved evaluations are shown as unverified")
    p.add_argument("--model", help="model whose evaluations count as current "
                                   "(default: $CLAUDE_EVALUATOR_MODEL or the evaluator default)")
    p.add_argument("--max-age-days", type=_positive_int, default=DEFAULT_MAX_AGE_DAYS,
                   help=f"exclude jobs the employer published more than N days ago (default {DEFAULT_MAX_AGE_DAYS})")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="collector snapshot directory read for employer publication dates "
                        "(default: the directory containing --db, i.e. data/)")
    return p


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_hash = None
    if args.profile is not None:
        try:
            profile_hash = profile_sha256(load_profile(args.profile))
        except ProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    model = resolve_model(args.model)
    model_source = "" if args.model else " (default)"
    inputs = CurrentInputs(model=model, schema_version=EVALUATION_SCHEMA_VERSION, profile_sha256=profile_hash)
    data_dir = args.data_dir if args.data_dir is not None else args.db.parent
    snapshot_warnings: list[str] = []
    snapshot_dates = posting_dates.load_snapshot_published_dates(data_dir, snapshot_warnings)
    try:
        conn = open_readonly(args.db)
        try:
            data = build_report_data(conn, inputs, model_source=model_source, max_age_days=args.max_age_days,
                                     snapshot_dates=snapshot_dates)
        finally:
            conn.close()
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except sqlite3.Error as exc:
        print(f"error: cannot read database: {exc}", file=sys.stderr)
        return 2
    write_report(render_html(data), args.output)
    m = data.metrics
    print(f"Report written: {_display_path(args.output)}")
    print(f"  {m['eligible_jobs']} eligible of {m['active_jobs']} active jobs; "
          f"{m['current_evaluations']} current AI evaluation(s), {m['stale_evaluations']} stale, "
          f"{m['unverified_evaluations']} unverified")
    print(f"  Posting-date cutoff: {m['recent_jobs']} of {m['active_jobs']} active jobs published in the last "
          f"{data.max_age_days} days (on/after {data.cutoff.isoformat(timespec='seconds')}); excluded "
          f"{m['excluded_too_old']} older and {m['excluded_unknown_date']} with a missing or unverified "
          f"publication date (records kept in the database)")
    for warning in snapshot_warnings:
        print(f"  Note: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
