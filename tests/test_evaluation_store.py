"""Offline tests for evaluation persistence/caching and the evaluator CLI's cache path.

No network: the Anthropic client is a mock injected through ``client_factory``,
and ``anthropic.Anthropic`` is patched to fail if anything constructs it.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import anthropic
import httpx2 as httpx

from src import claude_evaluator as ce
from src import database, evaluation_store as store
from src.candidate_profile import load_profile
from tests.test_claude_evaluator import SYNTHETIC_PROFILE, fake_client, fake_response, make_job, valid_evaluation


def _http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.invalid/v1/messages"))


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("anthropic.Anthropic", side_effect=AssertionError("real Anthropic client constructed"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "jobs.db"
        self.conn = database.initialize_database(self.db_path)
        self.addCleanup(self.conn.close)
        company_id = database.upsert_company(self.conn, "Example Orbital", "greenhouse", "exampleorbital")
        database.upsert_jobs(self.conn, company_id, [dict(make_job(), updated_at=None)])
        self.job_id = self.conn.execute("SELECT id FROM jobs WHERE ats_job_id = '4242'").fetchone()[0]

    def key(self, **overrides) -> store.CacheKey:
        values = dict(job_id=self.job_id, description_sha256="a" * 64, profile_sha256="b" * 64,
                      model="claude-opus-5-5", schema_version=1, location_key="new york, ny")
        values.update(overrides)
        return store.CacheKey(**values)


# Exact v1 DDL (before location_key) — used to test the upgrade path.
V1_TABLE = """
    CREATE TABLE evaluations (
        id INTEGER PRIMARY KEY,
        job_id INTEGER NOT NULL REFERENCES jobs(id),
        description_sha256 TEXT NOT NULL CHECK (length(description_sha256) = 64),
        profile_sha256 TEXT NOT NULL CHECK (length(profile_sha256) = 64),
        model TEXT NOT NULL CHECK (length(model) > 0),
        response_model TEXT,
        schema_version INTEGER NOT NULL,
        evaluation_json TEXT NOT NULL,
        recommendation TEXT NOT NULL CHECK (recommendation IN ('Apply', 'Review', 'Skip')),
        overall_fit_score INTEGER NOT NULL CHECK (overall_fit_score BETWEEN 0 AND 100),
        evaluated_at TEXT NOT NULL
    )
"""
V1_INDEX = """
    CREATE INDEX idx_evaluations_cache_key
        ON evaluations (job_id, description_sha256, profile_sha256, model, schema_version, evaluated_at)
"""


def create_v1_table(conn: sqlite3.Connection, job_id: int, key_kwargs: dict, rows: int = 2) -> list[tuple]:
    """Create a v1 evaluations table with ``rows`` legacy rows; return them."""
    conn.execute(V1_TABLE)
    conn.execute(V1_INDEX)
    for i in range(rows):
        conn.execute(
            """INSERT INTO evaluations (job_id, description_sha256, profile_sha256, model, response_model,
                   schema_version, evaluation_json, recommendation, overall_fit_score, evaluated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, key_kwargs["description_sha256"], key_kwargs["profile_sha256"], key_kwargs["model"],
             "claude-opus-5-5", key_kwargs["schema_version"], json.dumps(valid_evaluation()), "Apply", 78,
             f"2026-11-0{i + 1}T00:00:00+00:00"),
        )
    return [tuple(r) for r in conn.execute("SELECT * FROM evaluations ORDER BY id")]


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

class HashTests(unittest.TestCase):
    def test_sha256_text_matches_hashlib(self):
        self.assertEqual(store.sha256_text("abc"), hashlib.sha256(b"abc").hexdigest())

    def test_profile_hash_ignores_key_order_but_not_content(self):
        a = {"x": [1, 2], "y": "z"}
        self.assertEqual(store.profile_sha256(a), store.profile_sha256({"y": "z", "x": [1, 2]}))
        self.assertNotEqual(store.profile_sha256(a), store.profile_sha256({"x": [2, 1], "y": "z"}))
        self.assertEqual(len(store.profile_sha256(a)), 64)


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #

class MigrationTests(StoreTestCase):
    def test_creates_table_and_is_repeatable(self):
        self.assertFalse(store.table_exists(self.conn))
        store.ensure_schema(self.conn)
        store.save_evaluation(self.conn, self.key(), valid_evaluation(), "claude-opus-5-5")
        store.ensure_schema(self.conn)  # second run: no error, data intact
        store.ensure_schema(self.conn)
        self.assertEqual(store.count_evaluations(self.conn), 1)

    def test_migration_leaves_jobs_and_companies_untouched(self):
        before = (self.conn.execute("SELECT * FROM jobs").fetchall(), self.conn.execute("SELECT * FROM companies").fetchall())
        schema_before = {r[0]: r[1] for r in self.conn.execute("SELECT name, sql FROM sqlite_master WHERE name IN ('jobs','companies')")}
        store.ensure_schema(self.conn)
        after = (self.conn.execute("SELECT * FROM jobs").fetchall(), self.conn.execute("SELECT * FROM companies").fetchall())
        schema_after = {r[0]: r[1] for r in self.conn.execute("SELECT name, sql FROM sqlite_master WHERE name IN ('jobs','companies')")}
        self.assertEqual([tuple(r) for r in before[0]], [tuple(r) for r in after[0]])
        self.assertEqual([tuple(r) for r in before[1]], [tuple(r) for r in after[1]])
        self.assertEqual(schema_before, schema_after)

    def test_collector_initialization_still_works_after_migration(self):
        store.ensure_schema(self.conn)
        database.initialize_database(self.db_path).close()
        self.assertTrue(store.table_exists(self.conn))

    def test_incompatible_existing_table_is_rejected_and_not_modified(self):
        self.conn.execute("CREATE TABLE evaluations (id INTEGER PRIMARY KEY, note TEXT)")
        self.conn.execute("INSERT INTO evaluations (note) VALUES ('keep me')")
        with self.assertRaises(store.EvaluationStoreError):
            store.ensure_schema(self.conn)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(evaluations)")}
        self.assertEqual(cols, {"id", "note"})
        self.assertEqual(self.conn.execute("SELECT note FROM evaluations").fetchone()[0], "keep me")
        self.assertFalse(self.conn.in_transaction)

    def test_requires_jobs_table(self):
        conn = database.connect(Path(self._tmp.name) / "empty.db")
        self.addCleanup(conn.close)
        with self.assertRaises(store.EvaluationStoreError):
            store.ensure_schema(conn)
        self.assertFalse(store.table_exists(conn))

    def test_foreign_key_to_jobs_enforced(self):
        store.ensure_schema(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            store.save_evaluation(self.conn, self.key(job_id=99999), valid_evaluation(), None)
        self.assertEqual(store.count_evaluations(self.conn), 0)

    def test_check_constraints(self):
        store.ensure_schema(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            store.save_evaluation(self.conn, self.key(), valid_evaluation(recommendation="Maybe"), None)
        with self.assertRaises(sqlite3.IntegrityError):
            store.save_evaluation(self.conn, self.key(description_sha256="short"), valid_evaluation(), None)

    def test_no_profile_columns(self):
        store.ensure_schema(self.conn)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(evaluations)")}
        self.assertEqual(cols, set(store.EXPECTED_COLUMNS))
        self.assertFalse(any("profile" in c and c != "profile_sha256" for c in cols))


# --------------------------------------------------------------------------- #
# Cache lookup / history
# --------------------------------------------------------------------------- #

class CacheLookupTests(StoreTestCase):
    def setUp(self):
        super().setUp()
        store.ensure_schema(self.conn)

    def test_miss_without_table(self):
        conn = database.connect(Path(self._tmp.name) / "other.db")
        self.addCleanup(conn.close)
        self.assertIsNone(store.find_cached(conn, self.key()))

    def test_hit_on_exact_key(self):
        saved = store.save_evaluation(self.conn, self.key(), valid_evaluation(), "claude-opus-5-5")
        hit = store.find_cached(self.conn, self.key())
        self.assertEqual(hit.id, saved.id)
        self.assertEqual(hit.evaluation, valid_evaluation())
        self.assertEqual(hit.response_model, "claude-opus-5-5")

    def test_each_key_part_invalidates(self):
        store.save_evaluation(self.conn, self.key(), valid_evaluation(), None)
        for change in (dict(description_sha256="c" * 64), dict(profile_sha256="d" * 64),
                       dict(model="claude-sonnet-5-5"), dict(schema_version=2), dict(location_key="remote")):
            with self.subTest(change=change):
                self.assertIsNone(store.find_cached(self.conn, self.key(**change)))

    def test_location_key_stored(self):
        store.save_evaluation(self.conn, self.key(location_key="remote"), valid_evaluation(), None)
        self.assertEqual(self.conn.execute("SELECT location_key FROM evaluations").fetchone()[0], "remote")
        self.assertIsNotNone(store.find_cached(self.conn, self.key(location_key="remote")))
        self.assertIsNone(store.find_cached(self.conn, self.key(location_key="new york, ny")))

    def test_history_preserved_and_latest_wins(self):
        first = store.save_evaluation(self.conn, self.key(), valid_evaluation(overall_fit_score=40),
                                      None, evaluated_at="2026-11-01T00:00:00+00:00")
        store.save_evaluation(self.conn, self.key(profile_sha256="e" * 64), valid_evaluation(), None)
        latest = store.save_evaluation(self.conn, self.key(), valid_evaluation(overall_fit_score=90),
                                       None, evaluated_at="2026-11-02T00:00:00+00:00")
        self.assertEqual(store.count_evaluations(self.conn, self.job_id), 3)
        self.assertEqual(store.find_cached(self.conn, self.key()).id, latest.id)
        row = self.conn.execute("SELECT overall_fit_score FROM evaluations WHERE id = ?", (first.id,)).fetchone()
        self.assertEqual(row[0], 40)

    def test_unreadable_row_is_a_miss(self):
        store.save_evaluation(self.conn, self.key(), valid_evaluation(), None)
        self.conn.execute("UPDATE evaluations SET evaluation_json = '{broken'")
        self.assertIsNone(store.find_cached(self.conn, self.key()))


class NormalizeLocationTests(unittest.TestCase):
    def test_identical_and_cosmetic_variants_match(self):
        n = store.normalize_location
        self.assertEqual(n("New York, NY"), n("New York, NY"))
        self.assertEqual(n("New York, NY"), n("  new  york,\tNY "))

    def test_real_changes_differ(self):
        n = store.normalize_location
        self.assertNotEqual(n("Remote"), n("New York, NY"))
        self.assertNotEqual(n("New York, NY"), n("New York, NY; Remote"))
        self.assertNotEqual(n("Remote"), n("Remote (US)"))

    def test_missing_location(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertEqual(store.normalize_location(value), "")
        self.assertNotEqual(store.normalize_location(None), store.normalize_location("Remote"))


class LocationMigrationTests(StoreTestCase):
    """Upgrading a v1 evaluations table (no location_key) must not lose or rewrite rows."""

    def setUp(self):
        super().setUp()
        k = self.key()
        self.legacy = create_v1_table(self.conn, self.job_id, vars(k))

    def test_upgrade_adds_column_and_preserves_rows(self):
        store.ensure_schema(self.conn)
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(evaluations)")]
        self.assertIn("location_key", cols)
        rows = [tuple(r) for r in self.conn.execute("SELECT * FROM evaluations ORDER BY id")]
        self.assertEqual(len(rows), len(self.legacy))
        for old, new in zip(self.legacy, rows):
            self.assertEqual(new[:len(old)], old)      # every original value unchanged
            self.assertIsNone(new[cols.index("location_key")])

    def test_upgrade_is_repeatable(self):
        store.ensure_schema(self.conn)
        store.ensure_schema(self.conn)
        self.assertEqual(store.count_evaluations(self.conn), len(self.legacy))

    def test_indexes_replaced(self):
        store.ensure_schema(self.conn)
        indexes = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'evaluations'")}
        self.assertIn("idx_evaluations_cache_key_v2", indexes)
        self.assertNotIn("idx_evaluations_cache_key", indexes)

    def test_legacy_rows_never_hit(self):
        self.assertIsNone(store.find_cached(self.conn, self.key()))   # before upgrade
        store.ensure_schema(self.conn)
        self.assertIsNone(store.find_cached(self.conn, self.key()))   # after upgrade
        self.assertIsNone(store.find_cached(self.conn, self.key(location_key="")))

    def test_find_cached_on_v1_table_does_not_migrate(self):
        store.find_cached(self.conn, self.key())
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(evaluations)")}
        self.assertNotIn("location_key", cols)

    def test_new_rows_after_upgrade_hit_and_history_kept(self):
        store.ensure_schema(self.conn)
        saved = store.save_evaluation(self.conn, self.key(), valid_evaluation(), None)
        self.assertEqual(store.find_cached(self.conn, self.key()).id, saved.id)
        self.assertEqual(store.count_evaluations(self.conn), len(self.legacy) + 1)

    def test_failed_upgrade_rolls_back(self):
        with mock.patch.object(store, "_CREATE_INDEX", "CREATE INDEX broken ON evaluations (no_such_column)"):
            with self.assertRaises(sqlite3.OperationalError):
                store.ensure_schema(self.conn)
        self.assertFalse(self.conn.in_transaction)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(evaluations)")}
        self.assertNotIn("location_key", cols)    # ALTER rolled back with the transaction
        self.assertEqual([tuple(r) for r in self.conn.execute("SELECT * FROM evaluations ORDER BY id")],
                         self.legacy)

# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #

class CliCacheTests(StoreTestCase):
    def setUp(self):
        super().setUp()
        self.conn.close()  # the CLI opens its own connections
        key_patch = mock.patch.object(ce, "load_api_key", return_value="sk-ant-test-NOT-REAL")
        self.load_key = key_patch.start()
        self.addCleanup(key_patch.stop)
        self.factory = mock.MagicMock()
        self.setup_tracking()
        self.profile = load_profile(SYNTHETIC_PROFILE)

    def run_cli(self, *extra: str, profile: Path = SYNTHETIC_PROFILE):
        argv = ["--provider", "greenhouse", "--posting-id", "4242", "--profile", str(profile),
                "--db", str(self.db_path), *extra]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ce.main(argv, client_factory=self.factory)
        return code, out.getvalue(), err.getvalue()

    def requests_made(self) -> int:
        """Total messages.create calls across every client the CLI constructed."""
        return sum(c.messages.create.call_count + c.beta.messages.create.call_count for c in self._created)

    def setup_tracking(self, response_text: str | None = None, error: Exception | None = None,
                       stop_reason: str = "end_turn"):
        self._created = []
        text = json.dumps(valid_evaluation()) if response_text is None else response_text

        def factory(*_):
            client = fake_client(fake_response(text, stop_reason=stop_reason), error=error)
            self._created.append(client)
            return client
        self.factory.side_effect = factory

    def db(self) -> sqlite3.Connection:
        conn = database.connect(self.db_path)
        self.addCleanup(conn.close)
        return conn

    def stored_count(self) -> int:
        return store.count_evaluations(self.db())

    # -- miss then hit -------------------------------------------------------

    def test_first_call_misses_and_stores_second_call_hits_without_request(self):
        self.setup_tracking()
        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0, err)
        first = json.loads(out)
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(first["cache"]["saved"])
        self.assertEqual(self.requests_made(), 1)
        self.assertEqual(self.stored_count(), 1)

        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0, err)
        second = json.loads(out)
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["cache"]["evaluation_id"], first["cache"]["evaluation_id"])
        self.assertEqual(self.requests_made(), 1)          # no new request
        self.assertEqual(self.factory.call_count, 1)        # no client even constructed
        self.assertEqual(self.stored_count(), 1)
        self.assertEqual(second["recommendation"], first["recommendation"])

    def test_cache_hit_needs_no_api_key(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        self.load_key.return_value = None
        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.factory.call_count, 1)

    def test_stored_hashes_and_no_profile_contents(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        row = self.db().execute("SELECT * FROM evaluations").fetchone()
        self.assertEqual(row["description_sha256"], store.sha256_text(make_job()["description"]))
        self.assertEqual(row["profile_sha256"], store.profile_sha256(self.profile))
        self.assertEqual(row["model"], ce.DEFAULT_MODEL)
        self.assertEqual(row["schema_version"], ce.EVALUATION_SCHEMA_VERSION)
        self.assertTrue(row["evaluated_at"])
        dump = "\n".join(str(v) for v in tuple(row))
        for secret in ("Synthetic State University", "calibration pipeline", "sk-ant"):
            self.assertNotIn(secret, dump)
        self.assertEqual(json.loads(row["evaluation_json"]), valid_evaluation())

    # -- invalidation --------------------------------------------------------

    def test_profile_change_invalidates_and_keeps_history(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        other = Path(self._tmp.name) / "other_profile.json"
        data = json.loads(SYNTHETIC_PROFILE.read_text(encoding="utf-8"))
        data["technical_skills"].append("Rust")
        other.write_text(json.dumps(data), encoding="utf-8")
        code, out, _ = self.run_cli("--call-api", profile=other)
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 2)
        self.assertEqual(self.stored_count(), 2)
        # The original profile still hits its own cached row.
        code, out, _ = self.run_cli("--call-api")
        self.assertTrue(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 2)

    def test_description_change_invalidates(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        conn = self.db()
        company_id = conn.execute("SELECT id FROM companies").fetchone()[0]
        database.upsert_jobs(conn, company_id, [dict(make_job(description="Updated posting text."), updated_at=None)])
        code, out, _ = self.run_cli("--call-api")
        self.assertFalse(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 2)
        self.assertEqual(self.stored_count(), 2)

    def test_model_change_invalidates(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        code, out, _ = self.run_cli("--call-api", "--model", "claude-sonnet-5-5")
        self.assertFalse(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 2)

    def test_schema_version_change_invalidates(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        current = ce.EVALUATION_SCHEMA_VERSION
        with mock.patch.object(ce, "EVALUATION_SCHEMA_VERSION", current + 1):
            code, out, _ = self.run_cli("--call-api")
        self.assertFalse(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.stored_count(), 2)
        versions = {r[0] for r in self.db().execute("SELECT schema_version FROM evaluations")}
        self.assertEqual(versions, {current, current + 1})

    def test_v1_evaluation_rows_kept_but_not_reused(self):
        """Rows from schema v1 (pre-concise format) stay as history; v2 re-evaluates."""
        self.setup_tracking()
        conn = self.db()
        store.ensure_schema(conn)
        job_id = conn.execute("SELECT id FROM jobs").fetchone()[0]
        v1_eval = {k: v for k, v in valid_evaluation().items() if k != "key_responsibilities"}
        v1_key = store.CacheKey(job_id=job_id, description_sha256=store.sha256_text(make_job()["description"]),
                                profile_sha256=store.profile_sha256(self.profile), model=ce.DEFAULT_MODEL,
                                schema_version=1, location_key=store.normalize_location(make_job()["location"]))
        old = store.save_evaluation(conn, v1_key, v1_eval, ce.DEFAULT_MODEL)
        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0, err)
        self.assertFalse(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 1)
        row = conn.execute("SELECT schema_version, evaluation_json FROM evaluations WHERE id = ?", (old.id,)).fetchone()
        self.assertEqual(row[0], 1)
        self.assertEqual(json.loads(row[1]), v1_eval)
        self.assertEqual(sorted(r[0] for r in conn.execute("SELECT schema_version FROM evaluations")),
                         [1, ce.EVALUATION_SCHEMA_VERSION])

    def test_truncation_changes_description_hash(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        code, out, _ = self.run_cli("--call-api", "--max-description-chars", "10")
        self.assertFalse(json.loads(out)["cache"]["hit"])

    def set_location(self, location: str | None) -> None:
        """Change only the job's location (description unchanged)."""
        self.db().execute("UPDATE jobs SET location = ?", (location,))

    def test_location_change_invalidates_with_same_description(self):
        self.setup_tracking()
        self.set_location("Remote")
        code, out, _ = self.run_cli("--call-api")
        first = json.loads(out)
        self.assertFalse(first["cache"]["hit"])
        self.assertEqual(first["recommendation"], "Review")   # policy for "Remote"

        self.set_location("New York, NY")
        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0, err)
        second = json.loads(out)
        self.assertFalse(second["cache"]["hit"])
        self.assertEqual(second["recommendation"], "Apply")
        self.assertEqual(self.requests_made(), 2)
        rows = self.db().execute("SELECT location_key, description_sha256 FROM evaluations ORDER BY id").fetchall()
        self.assertEqual([r[0] for r in rows], ["remote", "new york, ny"])
        self.assertEqual(rows[0][1], rows[1][1])              # description hash unchanged
        # The request for the new location actually carried the new location.
        sent = self._created[-1].beta.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn('"location": "New York, NY"', sent)

    def test_identical_location_hits(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        code, out, _ = self.run_cli("--call-api")
        self.assertTrue(json.loads(out)["cache"]["hit"])
        self.set_location("  new york,  NY ")                  # cosmetic difference only
        code, out, _ = self.run_cli("--call-api")
        self.assertTrue(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 1)

    def test_returning_to_previous_location_reuses_its_row(self):
        self.setup_tracking()
        _, out, _ = self.run_cli("--call-api")                 # New York, NY
        ny_id = json.loads(out)["cache"]["evaluation_id"]
        self.set_location("Remote")
        self.run_cli("--call-api")
        self.set_location("New York, NY")
        _, out, _ = self.run_cli("--call-api")
        hit = json.loads(out)["cache"]
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["evaluation_id"], ny_id)
        self.assertEqual(self.requests_made(), 2)
        self.assertEqual(self.stored_count(), 2)

    def test_missing_location_is_its_own_key(self):
        self.setup_tracking()
        self.set_location(None)
        _, out, _ = self.run_cli("--call-api")
        self.assertFalse(json.loads(out)["cache"]["hit"])
        _, out, _ = self.run_cli("--call-api")
        self.assertTrue(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.db().execute("SELECT location_key FROM evaluations").fetchone()[0], "")

    def test_dry_run_reports_location_miss(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        self.set_location("Remote")
        code, out, _ = self.run_cli()
        data = json.loads(out)
        self.assertFalse(data["cache"]["hit"])
        self.assertTrue(data["would_call_api"])
        self.assertEqual(self.requests_made(), 1)

    def test_effort_is_not_part_of_key(self):
        self.setup_tracking()
        self.run_cli("--call-api", "--effort", "medium")
        code, out, _ = self.run_cli("--call-api", "--effort", "high")
        self.assertTrue(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 1)

    def test_cli_upgrades_v1_table_and_keeps_legacy_rows(self):
        self.setup_tracking()
        conn = self.db()
        legacy = create_v1_table(conn, self.job_id, vars(self.key(
            description_sha256=store.sha256_text(make_job()["description"]),
            profile_sha256=store.profile_sha256(self.profile), model=ce.DEFAULT_MODEL)))
        # Dry run: read-only, no migration, legacy rows cannot hit.
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(out)["cache"]["hit"])
        self.assertNotIn("location_key", {r[1] for r in conn.execute("PRAGMA table_info(evaluations)")})
        # --call-api: migrates, misses (legacy location unknown), appends a new row.
        code, out, err = self.run_cli("--call-api")
        self.assertEqual(code, 0, err)
        self.assertFalse(json.loads(out)["cache"]["hit"])
        rows = [tuple(r) for r in conn.execute("SELECT * FROM evaluations ORDER BY id")]
        self.assertEqual(len(rows), len(legacy) + 1)
        for old, new in zip(legacy, rows):
            self.assertEqual(new[:len(old)], old)
        self.assertEqual(self.requests_made(), 1)

    def test_invalid_cached_row_is_not_reused(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        self.db().execute("UPDATE evaluations SET evaluation_json = ?",
                          (json.dumps(valid_evaluation(explanation="")),))
        code, out, _ = self.run_cli("--call-api")
        self.assertFalse(json.loads(out)["cache"]["hit"])
        self.assertEqual(self.requests_made(), 2)

    # -- --force -------------------------------------------------------------

    def test_force_bypasses_cache_and_keeps_history(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        code, out, err = self.run_cli("--call-api", "--force")
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertFalse(result["cache"]["hit"])
        self.assertTrue(result["cache"]["forced"])
        self.assertEqual(self.requests_made(), 2)
        self.assertEqual(self.stored_count(), 2)
        # Later normal run reuses the newest row.
        code, out, _ = self.run_cli("--call-api")
        hit = json.loads(out)["cache"]
        self.assertTrue(hit["hit"])
        self.assertEqual(hit["evaluation_id"], result["cache"]["evaluation_id"])

    def test_force_without_call_api_is_still_a_dry_run(self):
        self.setup_tracking()
        code, out, _ = self.run_cli("--force")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(data["dry_run"])
        self.assertTrue(data["would_call_api"])
        self.factory.assert_not_called()
        self.assertFalse(store.table_exists(self.db()))

    # -- failures are never cached ---------------------------------------------

    def test_failures_are_not_cached(self):
        cases = {
            "api_error": dict(error=anthropic.RateLimitError("slow", response=_http_response(429), body=None)),
            "timeout": dict(error=anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.invalid"))),
            "malformed": dict(response_text="not json"),
            "invalid": dict(response_text=json.dumps(valid_evaluation(overall_fit_score=500))),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                self.setup_tracking(**kwargs)
                code, out, _ = self.run_cli("--call-api")
                self.assertEqual(code, 1)
                self.assertEqual(out, "")
                self.assertEqual(self.stored_count(), 0)

    def test_refusal_is_not_cached(self):
        self.setup_tracking(response_text="", stop_reason="refusal")
        code, _, err = self.run_cli("--call-api")
        self.assertEqual(code, 1)
        self.assertIn("refusal", err)
        self.assertEqual(self.stored_count(), 0)

    # -- dry run and safeguards ------------------------------------------------

    def test_dry_run_never_creates_table_or_requests(self):
        self.setup_tracking()
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertFalse(data["cache"]["hit"])
        self.assertTrue(data["would_call_api"])
        self.factory.assert_not_called()
        self.assertFalse(store.table_exists(self.db()))

    def test_dry_run_reports_cache_hit_without_printing_evaluation(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        code, out, _ = self.run_cli()
        data = json.loads(out)
        self.assertTrue(data["dry_run"])
        self.assertTrue(data["cache"]["hit"])
        self.assertFalse(data["would_call_api"])
        self.assertNotIn("explanation", data)
        self.assertEqual(self.factory.call_count, 1)

    def test_dry_run_with_cache_hit_ready_without_key(self):
        self.setup_tracking()
        self.run_cli("--call-api")
        self.load_key.return_value = None
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["ready"])

    def test_cache_miss_without_key_makes_no_request(self):
        self.setup_tracking()
        self.load_key.return_value = None
        code, _, err = self.run_cli("--call-api")
        self.assertEqual(code, 2)
        self.assertIn("ANTHROPIC_API_KEY", err)
        self.factory.assert_not_called()

    def test_real_profile_still_refused(self):
        self.setup_tracking()
        code, _, err = self.run_cli("--call-api", profile=ce.DEFAULT_PROFILE)
        self.assertEqual(code, 2)
        self.factory.assert_not_called()

    def test_incompatible_table_blocks_call_without_request(self):
        self.setup_tracking()
        conn = self.db()
        conn.execute("CREATE TABLE evaluations (id INTEGER PRIMARY KEY, note TEXT)")
        code, _, err = self.run_cli("--call-api")
        self.assertEqual(code, 2)
        self.assertIn("incompatible", err)
        self.factory.assert_not_called()

    def test_no_batch_mode(self):
        options = {a for action in ce.build_parser()._actions for a in action.option_strings}
        self.assertFalse(options & {"--all", "--batch", "--company"})
        self.assertFalse(any("batch" in o for o in options))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ce.main(["--provider", "greenhouse", "--profile", str(SYNTHETIC_PROFILE), "--db", str(self.db_path),
                     "--call-api"], client_factory=self.factory)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
