"""Offline tests for the HTML Job Intelligence Report. Temporary databases only; no network."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from src import claude_evaluator as ce
from src import database, evaluation_store as store
from src import generate_report as gr
from src.candidate_profile import load_profile
from tests.test_claude_evaluator import SYNTHETIC_PROFILE, valid_evaluation
from tests.test_evaluation_store import V1_INDEX, V1_TABLE

MODEL = "claude-opus-5-5"
PROFILE = load_profile(SYNTHETIC_PROFILE)
PROFILE_HASH = store.profile_sha256(PROFILE)

# (ats_job_id, company, title, location)
JOBS = [
    ("1", "Alpha Orbital", "Satellite Operations Engineer", "New York, NY"),     # high, compatible
    ("2", "Beta Space", "Payload Systems Engineer", "Remote"),                  # high, review
    ("3", "Gamma Sensors", "Software Engineer", "Boston, MA (Hybrid)"),         # medium, boston_hybrid
    ("4", "Alpha Orbital", "Marketing Manager", "Brooklyn, NY"),                # low, compatible
    ("5", "Delta Aero", "Spacecraft Thermal Engineer", "Seattle, WA"),          # high, incompatible
    ("6", "Beta Space", "Office Coordinator", "New York, NY"),                  # medium, compatible
]


# Reference time for report data; fixture jobs were published two days earlier (inside the 30-day window).
NOW = datetime.now(timezone.utc).replace(microsecond=0)
RECENTLY_PUBLISHED = (NOW - timedelta(days=2)).isoformat()


def job_row(ats_job_id, company, title, location, description=None, published_at=RECENTLY_PUBLISHED):
    return {
        "company": company, "ats_provider": "greenhouse", "ats_job_id": ats_job_id, "title": title,
        "location": location, "description": description or f"Description for {title}.",
        "application_url": f"https://example.invalid/jobs/{ats_job_id}", "updated_at": None,
        "published_at": published_at,
    }


class ReportTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.db_path = self.tmp / "jobs.db"
        conn = database.initialize_database(self.db_path)
        for company in sorted({j[1] for j in JOBS}):
            cid = database.upsert_company(conn, company, "greenhouse", company.lower().replace(" ", ""))
            database.upsert_jobs(conn, cid, [job_row(*j) for j in JOBS if j[1] == company])
        conn.close()
        self.inputs = gr.CurrentInputs(model=MODEL, schema_version=ce.EVALUATION_SCHEMA_VERSION,
                                       profile_sha256=PROFILE_HASH)

    def rw(self) -> sqlite3.Connection:
        conn = database.connect(self.db_path)
        self.addCleanup(conn.close)
        return conn

    def job(self, ats_job_id: str) -> sqlite3.Row:
        return self.rw().execute("SELECT * FROM jobs WHERE ats_job_id = ?", (ats_job_id,)).fetchone()

    def save(self, ats_job_id: str, evaluation: dict | None = None, **key_overrides) -> int:
        """Save an evaluation keyed exactly as the evaluator would for the job's current state."""
        job = self.job(ats_job_id)
        conn = self.rw()
        store.ensure_schema(conn)
        values = dict(job_id=job["id"], description_sha256=store.sha256_text(job["description"]),
                      profile_sha256=PROFILE_HASH, model=MODEL, schema_version=ce.EVALUATION_SCHEMA_VERSION,
                      location_key=store.normalize_location(job["location"]))
        values.update(key_overrides)
        evaluated_at = values.pop("evaluated_at", None)
        return store.save_evaluation(conn, store.CacheKey(**values), evaluation or valid_evaluation(),
                                     MODEL, evaluated_at=evaluated_at).id

    def data(self, inputs: gr.CurrentInputs | None = None) -> gr.ReportData:
        conn = gr.open_readonly(self.db_path)
        try:
            return gr.build_report_data(conn, inputs or self.inputs, generated_at=NOW.isoformat())
        finally:
            conn.close()

    def entry(self, data: gr.ReportData, ats_job_id: str) -> gr.JobEntry:
        return next(e for e in data.entries if e.job["ats_job_id"] == ats_job_id)

    def order(self, data: gr.ReportData) -> list[str]:
        return [e.job["ats_job_id"] for e in data.entries]

    def run_main(self, *extra: str) -> tuple[int, str, str, str]:
        output = self.tmp / "out" / "report.html"
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = gr.main(["--db", str(self.db_path), "--output", str(output), "--model", MODEL, *extra])
        text = output.read_text(encoding="utf-8") if output.exists() else ""
        return code, text, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Eligibility, metrics and ranking
# --------------------------------------------------------------------------- #

class RankingTests(ReportTestCase):
    def test_incompatible_jobs_excluded(self):
        data = self.data()
        self.assertNotIn("5", self.order(data))
        self.assertEqual({e.location["classification"] for e in data.entries},
                         {"compatible", "review", "boston_hybrid"})

    def test_metrics(self):
        m = self.data().metrics
        self.assertEqual(m["active_jobs"], 6)
        self.assertEqual(m["eligible_jobs"], 5)
        self.assertEqual((m["relevance_high"], m["relevance_medium"], m["relevance_low"]), (2, 2, 1))
        self.assertEqual(m["current_evaluations"], 0)

    def test_inactive_jobs_excluded(self):
        self.rw().execute("UPDATE jobs SET is_active = 0 WHERE ats_job_id = '1'")
        data = self.data()
        self.assertNotIn("1", self.order(data))
        self.assertEqual(data.metrics["active_jobs"], 5)

    def test_unevaluated_ranked_by_relevance(self):
        # high (1, 2) -> medium (3 score 3.0, 6 score 0.0) -> low (4)
        self.assertEqual(self.order(self.data()), ["1", "2", "3", "6", "4"])

    def test_evaluated_first_by_score_then_unevaluated_by_relevance(self):
        self.save("4", valid_evaluation(overall_fit_score=91, recommendation="Review"))  # low relevance
        self.save("6", valid_evaluation(overall_fit_score=55))
        data = self.data()
        self.assertEqual(self.order(data), ["4", "6", "1", "2", "3"])
        self.assertEqual([e.score for e in data.entries], [91, 55, None, None, None])
        self.assertEqual(data.metrics["current_evaluations"], 2)

    def test_unevaluated_never_has_score_or_recommendation(self):
        self.save("1")
        for e in self.data().entries:
            if e.status != gr.CURRENT:
                self.assertIsNone(e.score)
                self.assertIsNone(e.recommendation)
                self.assertIsNone(e.evaluation)

    def test_location_policy_applied_to_displayed_recommendation(self):
        self.save("2", valid_evaluation(recommendation="Apply"))   # "Remote" -> review
        e = self.entry(self.data(), "2")
        self.assertEqual(e.recommendation, "Review")
        self.assertTrue(e.evaluation["policy_adjustments"])

    def test_tie_break_is_deterministic(self):
        self.save("1", valid_evaluation(overall_fit_score=70))
        self.save("6", valid_evaluation(overall_fit_score=70))
        self.assertEqual(self.order(self.data())[:2], ["1", "6"])  # high relevance before medium


# --------------------------------------------------------------------------- #
# Current vs stale vs unverified
# --------------------------------------------------------------------------- #

class StalenessTests(ReportTestCase):
    def status(self, ats_job_id: str, inputs: gr.CurrentInputs | None = None) -> gr.JobEntry:
        return self.entry(self.data(inputs), ats_job_id)

    def test_matching_evaluation_is_current(self):
        self.save("1")
        e = self.status("1")
        self.assertEqual(e.status, gr.CURRENT)
        self.assertEqual(e.score, 78)

    def test_description_change_is_stale(self):
        self.save("1")
        self.rw().execute("UPDATE jobs SET description = 'Rewritten posting.' WHERE ats_job_id = '1'")
        e = self.status("1")
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("job description changed", e.stale_reasons)
        self.assertIsNone(e.score)

    def test_location_change_is_stale(self):
        self.save("1")
        self.rw().execute("UPDATE jobs SET location = 'Brooklyn, NY' WHERE ats_job_id = '1'")
        e = self.status("1")
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("job location changed", e.stale_reasons)

    def test_cosmetic_location_change_stays_current(self):
        self.save("1")
        self.rw().execute("UPDATE jobs SET location = '  new york,  NY ' WHERE ats_job_id = '1'")
        self.assertEqual(self.status("1").status, gr.CURRENT)

    def test_model_change_is_stale(self):
        self.save("1")
        e = self.status("1", gr.CurrentInputs("claude-sonnet-5-5", ce.EVALUATION_SCHEMA_VERSION, PROFILE_HASH))
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("different model", e.stale_reasons)

    def test_profile_change_is_stale(self):
        self.save("1")
        e = self.status("1", gr.CurrentInputs(MODEL, ce.EVALUATION_SCHEMA_VERSION, "f" * 64))
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("different candidate profile", e.stale_reasons)

    def test_schema_version_change_is_stale(self):
        self.save("1")
        e = self.status("1", gr.CurrentInputs(MODEL, ce.EVALUATION_SCHEMA_VERSION + 1, PROFILE_HASH))
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("older evaluation schema version", e.stale_reasons)

    def test_no_profile_means_unverified_not_current(self):
        self.save("1")
        data = self.data(gr.CurrentInputs(MODEL, ce.EVALUATION_SCHEMA_VERSION, None))
        e = self.entry(data, "1")
        self.assertEqual(e.status, gr.UNVERIFIED)
        self.assertIsNone(e.score)
        self.assertEqual(data.metrics["current_evaluations"], 0)
        self.assertEqual(data.metrics["unverified_evaluations"], 1)

    def test_older_matching_row_used_when_newest_is_stale(self):
        first = self.save("1", valid_evaluation(overall_fit_score=60), evaluated_at="2026-11-01T00:00:00+00:00")
        self.save("1", valid_evaluation(overall_fit_score=90), profile_sha256="f" * 64,
                  evaluated_at="2026-11-02T00:00:00+00:00")
        e = self.status("1")
        self.assertEqual(e.status, gr.CURRENT)
        self.assertEqual(e.score, 60)
        self.assertTrue(first)

    def test_newest_matching_row_wins(self):
        self.save("1", valid_evaluation(overall_fit_score=60), evaluated_at="2026-11-01T00:00:00+00:00")
        self.save("1", valid_evaluation(overall_fit_score=85), evaluated_at="2026-11-02T00:00:00+00:00")
        self.assertEqual(self.status("1").score, 85)

    def test_invalid_saved_evaluation_not_current(self):
        self.save("1")
        self.rw().execute("UPDATE evaluations SET evaluation_json = ?",
                          (json.dumps(valid_evaluation(explanation="")),))
        e = self.status("1")
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("saved evaluation failed validation", e.stale_reasons)

    def test_v1_table_without_location_is_stale(self):
        conn = self.rw()
        conn.execute(V1_TABLE)
        conn.execute(V1_INDEX)
        job = self.job("1")
        conn.execute(
            """INSERT INTO evaluations (job_id, description_sha256, profile_sha256, model, response_model,
                   schema_version, evaluation_json, recommendation, overall_fit_score, evaluated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'Apply', 78, '2026-11-01T00:00:00+00:00')""",
            (job["id"], store.sha256_text(job["description"]), PROFILE_HASH, MODEL, MODEL,
             ce.EVALUATION_SCHEMA_VERSION, json.dumps(valid_evaluation())))
        e = self.status("1")
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("location not recorded (older evaluation)", e.stale_reasons)

    def test_stale_and_unverified_rendered_as_not_evaluated(self):
        self.save("1")
        self.rw().execute("UPDATE jobs SET description = 'Changed.' WHERE ats_job_id = '1'")
        page = gr.render_html(self.data())
        card = _card_for(page, "Satellite Operations Engineer")
        self.assertIn(">Not evaluated</span>", card)
        self.assertIn('title="A saved evaluation exists but is stale: job description changed."', card)
        self.assertNotIn("/100 fit", card)
        self.assertNotIn("<details", card)
        self.assertIn('data-recommendation="none"', card)

        page = gr.render_html(self.data(gr.CurrentInputs(MODEL, ce.EVALUATION_SCHEMA_VERSION, None)))
        card = _card_for(page, "Satellite Operations Engineer")
        self.assertIn(">Not evaluated</span>", card)
        self.assertIn("cannot be verified", card)
        self.assertNotIn("/100 fit", card)

# --------------------------------------------------------------------------- #
# Missing table / read-only behaviour
# --------------------------------------------------------------------------- #

class ReadOnlyTests(ReportTestCase):
    def test_missing_evaluations_table(self):
        self.assertFalse(store.table_exists(self.rw()))
        data = self.data()
        self.assertEqual(len(data.entries), 5)
        self.assertTrue(all(e.status == gr.NOT_EVALUATED for e in data.entries))
        page = gr.render_html(data)
        self.assertEqual(page.count('>Not evaluated</span>'), 5)
        self.assertNotIn("<details", page)

    def test_main_does_not_modify_database(self):
        self.save("1")
        before = self.db_path.read_bytes()
        mtime = self.db_path.stat().st_mtime_ns
        code, _, _, err = self.run_main("--profile", str(SYNTHETIC_PROFILE))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.db_path.read_bytes(), before)
        self.assertEqual(self.db_path.stat().st_mtime_ns, mtime)
        leftovers = [p.name for p in self.tmp.iterdir() if p.name.startswith("jobs.db-")]
        self.assertEqual(leftovers, [])

    def test_main_never_creates_evaluations_table(self):
        code, page, _, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertFalse(store.table_exists(self.rw()))
        self.assertIn("Not evaluated", page)

    def test_connection_is_read_only(self):
        conn = gr.open_readonly(self.db_path)
        self.addCleanup(conn.close)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE x (id INTEGER)")

    def test_missing_database_not_created(self):
        missing = self.tmp / "nope.db"
        with redirect_stderr(io.StringIO()) as err:
            code = gr.main(["--db", str(missing), "--output", str(self.tmp / "r.html")])
        self.assertEqual(code, 2)
        self.assertFalse(missing.exists())
        self.assertFalse((self.tmp / "r.html").exists())
        self.assertNotIn(str(self.tmp), err.getvalue())

    def test_no_network_or_api_clients(self):
        with mock.patch("anthropic.Anthropic", side_effect=AssertionError("API client constructed")), \
             mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
            code, _, _, err = self.run_main("--profile", str(SYNTHETIC_PROFILE))
        self.assertEqual(code, 0, err)

    def test_invalid_profile_exit_code(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        code, page, _, _ = self.run_main("--profile", str(bad))
        self.assertEqual(code, 2)
        self.assertEqual(page, "")


# --------------------------------------------------------------------------- #
# HTML: escaping, filters, privacy
# --------------------------------------------------------------------------- #

def _visible_text(fragment: str) -> str:
    """Text a reader sees (tags and attributes removed, entities decoded, whitespace collapsed)."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def _card_for(page: str, title_fragment: str) -> str:
    for card in re.findall(r"<article class=\"job\".*?</article>", page, flags=re.S):
        if title_fragment in card:
            return card
    raise AssertionError(f"no card containing {title_fragment!r}")


class HtmlTests(ReportTestCase):
    def test_job_and_ai_text_escaped(self):
        conn = self.rw()
        conn.execute("UPDATE jobs SET title = ?, location = ? WHERE ats_job_id = '1'",
                     ('Engineer <script>alert("t")</script>', 'New York, NY'))
        conn.execute("UPDATE companies SET name = ? WHERE name = 'Alpha Orbital'",
                     ('Alpha" onmouseover="alert(1)',))
        self.save("1", valid_evaluation(
            key_responsibilities=['<img src=x onerror=alert(1)>'],
            missing_qualifications=[{"qualification": "</details><script>steal()</script>",
                                     "status": "not_established", "evidence": "x"}],
            matched_qualifications=[{"qualification": "<b>bold</b> & more", "evidence": "a"}],
            transferable_skills=[{"skill": '"><svg onload=alert(2)>', "relevance": "x"}],
        ))
        page = gr.render_html(self.data())
        for raw in ('<script>alert("t")</script>', "<img src=x", "<script>steal()", "<b>bold</b>",
                    'Alpha" onmouseover', "<svg"):
            self.assertNotIn(raw, page)
        self.assertIn("&lt;script&gt;alert(&quot;t&quot;)&lt;/script&gt;", page)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", page)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt; &amp; more", page)
        self.assertIn("&lt;/details&gt;&lt;script&gt;steal()&lt;/script&gt;", page)
        self.assertEqual(page.count("<script>"), 1)  # only the report's own filter script
        self.assertEqual(page.count("</details>"), page.count("<details>"))

    def test_unsafe_application_links_dropped(self):
        conn = self.rw()
        conn.execute("UPDATE jobs SET application_url = 'javascript:alert(1)' WHERE ats_job_id = '1'")
        conn.execute("UPDATE jobs SET application_url = 'data:text/html,hi' WHERE ats_job_id = '2'")
        page = gr.render_html(self.data())
        self.assertNotIn("javascript:", page)
        self.assertNotIn("data:text", page)
        self.assertIn("No valid application link", _card_for(page, "Satellite Operations Engineer"))
        self.assertIn('href="https://example.invalid/jobs/3"', page)
        self.assertIn('rel="noopener noreferrer"', page)

    def test_safe_url(self):
        self.assertEqual(gr.safe_url("https://a.example/x?y=1"), "https://a.example/x?y=1")
        for bad in ("javascript:alert(1)", " JAVASCRIPT:x", "data:x", "//evil.example", "", None, "https://"):
            with self.subTest(url=bad):
                self.assertIsNone(gr.safe_url(bad))

    def test_csp_hash_matches_inline_script(self):
        page = gr.render_html(self.data())
        script = re.search(r"<script>(.*?)</script>", page, flags=re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        csp = html.unescape(re.search(r'http-equiv="Content-Security-Policy" content="([^"]*)"', page).group(1))
        self.assertIn(f"script-src 'sha256-{digest}'", csp)
        self.assertIn("default-src 'none'", csp)

    def test_filter_attributes_and_options(self):
        self.save("1", valid_evaluation(recommendation="Apply"))
        page = gr.render_html(self.data())
        card = _card_for(page, "Satellite Operations Engineer")
        self.assertIn('data-company="Alpha Orbital"', card)
        self.assertIn('data-location="compatible"', card)
        self.assertIn('data-relevance="high"', card)
        self.assertIn('data-recommendation="Apply"', card)
        self.assertIn('data-search="alpha orbital satellite operations engineer"', card)  # company + title
        self.assertIn('data-uid="greenhouse|Alpha Orbital|1"', card)  # provider | employer | posting id
        self.assertIn('data-location-text="New York, NY"', card)
        self.assertIn('data-score="78"', card)
        self.assertIn(f'data-published="{int((NOW - timedelta(days=2)).timestamp() * 1000)}"', card)
        self.assertIn('data-match-rank="1"', card)
        boston = _card_for(page, "Software Engineer")
        self.assertIn('data-location="boston_hybrid"', boston)
        self.assertIn('data-recommendation="none"', boston)
        self.assertIn('data-score=""', boston)  # not evaluated: excluded by any minimum score
        self.assertIn('<option value="Boston, MA (Hybrid)">', page)  # location filter by place
        self.assertIn('<option value="70">70+</option>', page)
        for select_id in ("f-company", "f-location-text", "f-location", "f-relevance", "f-recommendation",
                          "f-min-score", "f-sort"):
            self.assertIn(f'id="{select_id}"', page)
        self.assertIn('<option value="boston_hybrid">', page)
        self.assertIn('<option value="none">Not evaluated</option>', page)
        self.assertNotIn('<option value="Delta Aero">', page)  # incompatible-only company

    def test_evaluated_card_collapsed_by_default(self):
        self.save("1")
        card = _card_for(gr.render_html(self.data()), "Satellite Operations Engineer")
        self.assertEqual(card.count("<details>"), 1)      # native element, no "open" attribute
        self.assertNotIn("<details open", card)
        self.assertNotRegex(card, r"<details[^>]*\bopen\b")
        self.assertIn("<summary>More details</summary>", card)

    def test_collapsed_card_shows_only_requested_information(self):
        self.save("1", valid_evaluation(recommendation="Apply", overall_fit_score=78))
        card = _card_for(gr.render_html(self.data()), "Satellite Operations Engineer")
        collapsed = card.split("<details>", 1)[0] + card.split("</details>", 1)[1]
        text = _visible_text(collapsed)
        expected = ["#1", "Alpha Orbital", "Satellite Operations Engineer", "78", "/100 fit", "Posted",
                    (NOW - timedelta(days=2)).strftime("%Y-%m-%d"), "· 2 days ago", "New York, NY",
                    "Compatible", "Title relevance: high", "Application page", "Mark as seen"]
        for item in expected:
            self.assertIn(item, text)
            text = text.replace(item, " ", 1)
        self.assertEqual(text.strip(), "", f"unexpected visible text in collapsed card: {text.strip()!r}")
        self.assertEqual(_visible_text(card.split("<summary>", 1)[1].split("</summary>", 1)[0]), "More details")

    def test_expanded_section_has_only_four_categories(self):
        self.save("1")
        card = _card_for(gr.render_html(self.data()), "Satellite Operations Engineer")
        details = card.split("<details>", 1)[1].split("</details>", 1)[0]
        headings = re.findall(r"<h4>(.*?)</h4>", details)
        self.assertEqual(headings, ["Key responsibilities", "Matched skills", "Missing qualifications",
                                    "Transferable skills"])
        for text in ("Integrate and test optical satellite payloads", "B.S. in Aerospace Engineering",
                     "M.S. degree", "Confirmed missing", "Thermal vacuum testing", "Not established",
                     "Technical documentation"):
            self.assertIn(text, details)
        self.assertIn("st-confirmed_missing", details)
        self.assertIn("st-not_established", details)

    def test_compact_card_order(self):
        self.save("1", valid_evaluation(recommendation="Apply"))
        card = _card_for(gr.render_html(self.data()), "Satellite Operations Engineer")
        markers = ['class="rank">#1', 'class="company">Alpha Orbital', "<h3>Satellite Operations Engineer",
                   "/100 fit", 'class="posted"', "<span>New York, NY", "Compatible", "Title relevance: high",
                   "Application page", 'class="seen-toggle"', "<details>", "Key responsibilities", "Matched skills", "Missing qualifications",
                   "Transferable skills"]
        positions = [card.index(m) for m in markers]
        self.assertEqual(positions, sorted(positions), dict(zip(markers, positions)))

    def test_recommendation_explanation_and_metadata_not_rendered(self):
        self.save("1", valid_evaluation(recommendation="Apply"))
        self.save("2", valid_evaluation(recommendation="Apply"))   # "Remote": policy downgrades to Review
        page = gr.render_html(self.data())
        ev = valid_evaluation()
        body = page.split("<main>", 1)[1]
        visible = _visible_text(body)
        for absent in ("Why this recommendation", ev["explanation"], ev["company_summary"],
                       "Profile education lists it.", "Not mentioned.", "Posting requires test reports.",
                       "Technical 80", "Aerospace 85", "Seniority 70", "Company 75", "Evaluated 20", "Location review required",
                       "deterministic filter", "Apply ->", "More details More"):
            self.assertNotIn(absent, visible)
        for card_name in ("Satellite Operations Engineer", "Payload Systems Engineer"):
            card_text = _visible_text(_card_for(page, card_name))
            for word in ("Apply", "Review", "Skip"):
                self.assertNotRegex(card_text, rf"\b{word}\b")
        # The recommendation still drives the filter, via a hidden data attribute only.
        self.assertIn('data-recommendation="Apply"', _card_for(page, "Satellite Operations Engineer"))
        self.assertIn('data-recommendation="Review"', _card_for(page, "Payload Systems Engineer"))

    def test_unevaluated_card_stays_compact(self):
        card = _card_for(gr.render_html(self.data()), "Payload Systems Engineer")
        text = _visible_text(card)
        for item in ("#2", "Beta Space", "Payload Systems Engineer", "Not evaluated", "Posted",
                     (NOW - timedelta(days=2)).strftime("%Y-%m-%d"), "· 2 days ago", "Remote", "Needs review",
                     "Title relevance: high", "Application page", "Mark as seen"):
            self.assertIn(item, text)
            text = text.replace(item, " ", 1)
        self.assertEqual(text.strip(), "")
        for absent in ("<details", "/100 fit", "Key responsibilities"):
            self.assertNotIn(absent, card)

    def test_evaluated_and_unevaluated_share_layout(self):
        self.save("1")
        page = gr.render_html(self.data())
        def structure(card):  # element classes in the collapsed part, colour modifiers removed
            classes = re.findall(r'<(?:div|span|p|h3|a) class="([^"]+)"', card.split("<details>", 1)[0])
            return [re.sub(r" (loc|rel)-\w+", "", c) for c in classes]
        evaluated = structure(_card_for(page, "Satellite Operations Engineer"))
        unevaluated = structure(_card_for(page, "Office Coordinator"))
        self.assertEqual([c.replace("ai score", "ai").replace("ai none", "ai") for c in evaluated],
                         [c.replace("ai score", "ai").replace("ai none", "ai") for c in unevaluated])

    def test_search_and_filter_script_still_wired(self):
        self.save("1", valid_evaluation(recommendation="Skip"))
        page = gr.render_html(self.data())
        script = re.search(r"<script>(.*?)</script>", page, flags=re.S).group(1)
        for needle in ("'q'", "'f-company'", "'f-location-text'", "'f-location'", "'f-relevance'",
                       "'f-recommendation'", "'f-min-score'", "'f-sort'", "dataset.search", "localStorage",
                       "card.dataset.uid", "var view = 'unseen'"):
            self.assertIn(needle, script)
        cards = re.findall(r"<article class=\"job\"(.*?)>", page, flags=re.S)
        self.assertEqual(len(cards), 5)
        for attrs in cards:
            for attr in ("data-uid=", "data-company=", "data-location=", "data-location-text=", "data-relevance=",
                         "data-recommendation=", "data-score=", "data-published=", "data-match-rank=",
                         "data-search="):
                self.assertIn(attr, attrs)
        self.assertIn('<option value="Skip">Skip</option>', page)
        self.assertIn('data-recommendation="Skip"', _card_for(page, "Satellite Operations Engineer"))

    def test_v1_shaped_evaluation_is_never_current(self):
        v1 = {k: v for k, v in valid_evaluation().items() if k != "key_responsibilities"}
        self.save("1", v1, schema_version=1)
        e = self.entry(self.data(), "1")
        self.assertEqual(e.status, gr.STALE)
        self.assertIn("older evaluation schema version", e.stale_reasons)

    def test_responsibility_text_escaped(self):
        self.save("1", valid_evaluation(key_responsibilities=["<script>x()</script> integrate payloads"]))
        page = gr.render_html(self.data())
        self.assertNotIn("<script>x()", page)
        self.assertIn("&lt;script&gt;x()&lt;/script&gt; integrate payloads", page)
        self.assertEqual(page.count("<script>"), 1)

    def test_mobile_viewport_and_offline(self):
        page = gr.render_html(self.data())
        self.assertIn('name="viewport"', page)
        self.assertIn("@media (max-width:700px)", page)
        self.assertNotRegex(page, r'<(script|link)[^>]+(src|href)=')  # no external resources

    def test_no_sensitive_content(self):
        self.save("1")
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-SECRET-TEST"}):
            code, page, stdout, _ = self.run_main("--profile", str(SYNTHETIC_PROFILE))
        self.assertEqual(code, 0)
        for secret in ("sk-ant", "Synthetic State University", "calibration pipeline", PROFILE_HASH,
                       str(self.tmp), SYNTHETIC_PROFILE.name, str(Path.home())):
            self.assertNotIn(secret, page)
        self.assertNotIn(str(self.tmp), stdout)
        self.assertIn(PROFILE_HASH[:12], page)  # short fingerprint only

    def test_main_writes_report_and_shows_current_evaluation(self):
        self.save("1")
        code, page, stdout, err = self.run_main("--profile", str(SYNTHETIC_PROFILE))
        self.assertEqual(code, 0, err)
        self.assertIn("78<small>/100 fit</small>", page)
        self.assertIn("1 current AI evaluation(s)", stdout)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertFalse((self.tmp / "out" / "report.html.tmp").exists())

    def test_main_without_profile_warns(self):
        self.save("1")
        code, page, _, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("no candidate profile given", page)
        self.assertNotIn("/100 fit", page)

    def test_default_output_location(self):
        self.assertEqual(gr.DEFAULT_OUTPUT.relative_to(gr.PROJECT_ROOT).as_posix(), "reports/job_intelligence.html")


if __name__ == "__main__":
    unittest.main()
