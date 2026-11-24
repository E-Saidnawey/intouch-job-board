# Aerospace Job Intelligence Agent — job collector

Collects public job postings from aerospace companies' applicant-tracking-system
(ATS) job boards, saves a JSON snapshot per company, and optionally keeps a
history in SQLite. Pure Python standard library; no dependencies, no API keys.

## Supported ATS providers

All providers use public, unauthenticated read-only APIs. No API keys, no
credentials, nothing is posted to them.

| `ats_provider` | API endpoint (GET)                                                       | `board_token` example |
|----------------|--------------------------------------------------------------------------|-----------------------|
| `greenhouse`   | `https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true` | `muonspace`           |
| `ashby`        | `https://api.ashbyhq.com/posting-api/job-board/{board_token}`             | `slingshot-aerospace` |
| `lever`        | `https://api.lever.co/v0/postings/{board_token}?mode=json`                | site in `jobs.lever.co/{site}` |

Every provider's postings are normalized to the same schema, so snapshots and
the database don't care which ATS a job came from:

| Normalized field  | Greenhouse source                | Ashby source                                         | Lever source |
|-------------------|----------------------------------|------------------------------------------------------|--------------|
| `company`         | config `name`                    | config `name`                                        | config `name` |
| `ats_provider`    | `"greenhouse"`                   | `"ashby"`                                            | `"lever"` |
| `ats_job_id`      | `id` (as string)                 | `id` (UUID string)                                   | `id` (UUID string) |
| `title`           | `title`                          | `title`                                              | `text` |
| `location`        | `location.name`                  | `location`                                           | `categories.location` |
| `description`     | `content` (escaped HTML → text)  | `descriptionHtml` (HTML → text); `descriptionPlain` if no HTML | `description` + each `lists[]` heading/items + `additional` (HTML → text; `*Plain` fallbacks) |
| `application_url` | `absolute_url`                   | `applyUrl`, falling back to `jobUrl`                 | `applyUrl`, falling back to `hostedUrl` |
| `updated_at`      | `updated_at`                     | `updatedAt` if present, otherwise `null`             | `updatedAt` if present (not in the public API), otherwise `null` |
| `published_at`    | `first_published`                | `publishedAt`                                        | `createdAt` (epoch ms → ISO-8601 UTC) |

`id` and `title` are required; a posting missing either fails that company's
collection. Missing optional fields become `null` (description: `""`).
Ashby's `publishedAt` is a publication date, not an update time, so it is
deliberately **not** used for `updated_at`; it is stored as `published_at` instead.
`published_at` is the employer's publication date and is never derived from
`updated_at`. No dates are ever invented.

Lever specifics: the response is a top-level JSON list (an unknown site returns
HTTP 404), and the public API exposes no separate publish time, so `createdAt`, the
employer's posting-creation time, is used as `published_at`. It is never later than
the real publication, so the report's 30-day cutoff can only err towards hiding a
job, never towards showing a stale one. A `createdAt` that is not a plausible epoch
timestamp is treated as unknown.

## Requirements

- Python 3.10+ (developed on 3.14)
- Run every command from the project root.
- Collecting jobs, the database and the report use only the standard library. The
  Claude evaluators and the test suite also need the packages in `requirements.txt`:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Setup: personal config files

Personal files are git-ignored. Each has a public template to copy and fill in:

| Create (git-ignored)                   | From template                          | Needed for                          |
|----------------------------------------|----------------------------------------|-------------------------------------|
| `config/companies.json`                | `config/companies.example.json`        | collecting jobs                     |
| `config/candidate_profile.json`        | `config/candidate_profile.example.json`| local profile validation / matching |
| `config/claude_candidate_profile.json` | `config/candidate_profile.example.json`| Claude evaluation and the report (sanitized: no names, employers or programs) |
| `.env`                                 | `.env.example`                         | `--call-api` only (`ANTHROPIC_API_KEY`) |

```powershell
Copy-Item config\companies.example.json config\companies.json
Copy-Item config\candidate_profile.example.json config\candidate_profile.json
Copy-Item config\candidate_profile.example.json config\claude_candidate_profile.json
Copy-Item .env.example .env
```

`data/`, `reports/`, `logs/` and `*.log` are git-ignored as well. Tests that check your
personal files are skipped when those files don't exist, so a fresh clone runs the
full suite with no setup beyond `requirements.txt`.

## Configuration

Companies are listed in `config/companies.json` (git-ignored; start from
[`config/companies.example.json`](config/companies.example.json), whose placeholder
entries are disabled). Without it, `collect_jobs` exits with code 2 and points to the template:

```json
{
  "companies": [
    { "name": "Muon Space", "ats_provider": "greenhouse", "board_token": "muonspace", "enabled": true }
  ]
}
```

| Field          | Required | Notes                                                                   |
|----------------|----------|-------------------------------------------------------------------------|
| `name`         | yes      | Display name; also used for the snapshot filename and the DB record.    |
| `ats_provider` | yes      | `greenhouse`, `ashby` or `lever`. Enabled entries must use a supported provider. |
| `board_token`  | yes      | Board slug: `muonspace` in `boards.greenhouse.io/muonspace`, `slingshot-aerospace` in `jobs.ashbyhq.com/slingshot-aerospace`; for Lever, the site identifier: `acme` in `jobs.lever.co/acme`. Add a Lever company only after confirming `https://api.lever.co/v0/postings/{site}?mode=json` returns its postings. |
| `enabled`      | no       | `true` (default) or `false`. `--all` skips disabled companies.          |

The config is validated before any network call (types, supported providers,
duplicate names / sources / snapshot filenames). Invalid config exits with code 2.

## Collecting jobs

```powershell
# All enabled companies, persisted to SQLite
python -m src.collect_jobs --all --db data/jobs.db

# One company (by name or board token, case-insensitive; repeatable)
python -m src.collect_jobs --company "Muon Space" --db data/jobs.db
python -m src.collect_jobs --company "Planet Labs" --company leolabsinc
python -m src.collect_jobs --company "Slingshot Aerospace" --db data/jobs.db   # Ashby

# No flags: legacy behaviour, Muon Space only
python -m src.collect_jobs
```

Options:

| Option            | Purpose                                                           |
|-------------------|-------------------------------------------------------------------|
| `--all`           | Collect every enabled company.                                    |
| `--company NAME`  | Collect the named company (repeatable). Exclusive with `--all`.   |
| `--db PATH`       | Also upsert into a SQLite database (e.g. `data/jobs.db`).         |
| `--config PATH`   | Use a different config file (default `config/companies.json`).   |
| `--data-dir PATH` | Snapshot directory (default `data/`).                             |
| `--output PATH`   | Snapshot path override; only valid for a single company.          |
| `--allow-empty`   | Accept a board that returns zero jobs (otherwise a failure).      |

### Outputs

- **JSON snapshots**: `data/<company_slug>_jobs.json`, e.g.
  `muon_space_jobs.json`, `planet_labs_jobs.json`, `ast_spacemobile_jobs.json`,
  `leolabs_jobs.json`, `slingshot_aerospace_jobs.json`. Each contains `collected_at` (UTC), `source`, `job_count`
  and normalized `jobs`. Snapshots are written atomically and only after a
  fully valid fetch, so a failed run never overwrites a good snapshot.
- **SQLite** (`--db`): tables `companies` and `jobs`. Jobs are keyed on
  `(company_id, ats_job_id)`. `first_seen_at` is set once; `last_seen_at` updates
  every run; `last_changed_at` updates only when a source field (title,
  location, description, application URL, source `updated_at`) changes. Jobs
  missing from a later run are not deactivated (yet). Each company is written
  in its own transaction. `source_published_at` holds the employer publication
  date; it is added to older databases as a nullable column (additive
  `ALTER TABLE`, existing rows kept), is not a change-detection field, and a
  missing value never erases a stored one.

### Failure handling and exit codes

Each company is collected independently: an HTTP error, network failure,
invalid JSON, malformed response, empty board or database error for one
company is reported on stderr and the remaining companies still run.

| Exit code | Meaning                                              |
|-----------|------------------------------------------------------|
| 0         | Every requested company succeeded.                   |
| 1         | One or more companies failed (see stderr / summary). |
| 2         | Invalid configuration or command-line arguments.     |

## Candidate profile (local only)

Resume-to-job matching (not implemented yet) will read a structured profile
from `config/candidate_profile.json`. That file is **git-ignored** and must
never be committed; the public template is
[`config/candidate_profile.example.json`](config/candidate_profile.example.json).

Create your profile from the template, then edit it:

```powershell
Copy-Item config\candidate_profile.example.json config\candidate_profile.json
```

Validate it (offline; nothing is sent anywhere, and only section counts are printed):

```powershell
python -m src.candidate_profile
python -m src.candidate_profile --profile path/to/other_profile.json
```

Exit codes: `0` valid, `1` invalid (every problem is listed), `2` file missing or not valid JSON.

| Section                | Type                                                                  |
|------------------------|-----------------------------------------------------------------------|
| `target_roles`         | non-empty list of strings                                             |
| `preferred_locations`  | object: `primary` (non-empty list of strings), optional `secondary`  |
| `years_experience`     | object mapping discipline → years (number, 0–60)                      |
| `education`            | non-empty list of `{degree, field, institution, status}` plus optional integer `year`; `status` is `completed` or `in_progress` |
| `aerospace_experience` | non-empty list of strings                                             |
| `technical_skills`     | non-empty list of strings                                             |
| `transferable_skills`  | non-empty list of strings                                             |
| `experience_evidence`  | *optional* list of `{skill, evidence}` objects (both non-empty strings): generalized examples of demonstrated capabilities |

`schema_version` (currently `1`) is optional. Strings must be non-empty,
list entries must be unique, and anything that looks like an email address or
phone number is rejected. Keep the profile to non-sensitive professional
background: no contact details, personal identifiers, employer program names,
clearance or classified information.

Confirm the profile is ignored before committing anything:

```powershell
git check-ignore -v config/candidate_profile.json config/claude_candidate_profile.json config/companies.json .env
```

## Claude job evaluator (single job, opt-in)

`src/claude_evaluator.py` asks Claude for a heuristic fit assessment of **one**
stored job against a **synthetic** candidate profile. It needs the `anthropic`
and `python-dotenv` packages and `ANTHROPIC_API_KEY` in `.env` (never printed).

```powershell
# Dry run (default): prints job metadata and whether the request is ready; sends nothing
python -m src.claude_evaluator --provider greenhouse --posting-id 4242 --profile tests/fixtures/synthetic_candidate_profile.json

# One real API request
python -m src.claude_evaluator --provider greenhouse --posting-id 4242 --profile tests/fixtures/synthetic_candidate_profile.json --call-api
```

- `--profile` is required; `config/candidate_profile.json` is refused (synthetic profiles only for now).
- `--model` (or `CLAUDE_EVALUATOR_MODEL`) picks the model, default `claude-opus-5-5`;
  `--effort`, `--timeout`, `--max-retries` (max 3) and an optional `--max-description-chars` cap are available.
  The full description is sent by default.
- Output is schema-validated JSON in a **concise format** (evaluation schema v2):
  five 0–100 scores (heuristic, **not** hiring probabilities), recommendation
  (`Apply` / `Review` / `Skip`), `location_review_required`, and:

  | Field                    | Limit                                                              |
  |--------------------------|--------------------------------------------------------------------|
  | `key_responsibilities`   | 1–3 bullets, ≤ 15 words each, from the posting                     |
  | `matched_qualifications` | ≤ 4, label ≤ 10 words + profile evidence ≤ 25 words                |
  | `missing_qualifications` | ≤ 3, each `confirmed_missing` or `not_established`, with evidence  |
  | `transferable_skills`    | ≤ 3, label ≤ 10 words + relevance ≤ 25 words                       |
  | `explanation`            | why this recommendation: 1–2 sentences, ≤ 45 words                 |
  | `company_summary`        | one sentence, ≤ 30 words                                           |

  Claude still receives the full job description. It is told to list only the most
  important items, most important first, and never to invent qualifications. Structured
  outputs can't enforce counts or word limits, so a fresh response is trimmed locally:
  lists keep their first N items and over-long text ends with "…". The result is then
  validated strictly. Malformed structure is still rejected, never repaired.
- The deterministic location filter wins: `review` locations can't be `Apply`
  (become `Review`), `incompatible` locations can't be `Apply` (become `Skip`).
  Changes are listed in `policy_adjustments`.
- Job text is treated as untrusted data; no tools are given to the model. This CLI takes
  one job per run; see [Batch evaluation](#batch-evaluation-all-eligible-jobs-opt-in) for all eligible jobs.

### Evaluation cache

Successful evaluations are stored in an `evaluations` table in the same database
(created on the first `--call-api` run; dry runs only read). Each row links to its
`jobs` row and stores the validated result, the model identifier, the time, the
schema version, and SHA-256 hashes of the job description (as sent) and of the
candidate profile. **Profile contents are never stored.**

- A cached row is reused only if job, description hash, profile hash, model,
  schema version **and normalized job location** all match. A hit prints the stored
  result with **no API request** (and no API key needed); the dry run reports
  `cache.hit` / `would_call_api`.
- The location key is the job's location string with whitespace collapsed, trimmed
  and case-folded (missing → empty). So `"Remote"` → `"New York, NY"` forces a fresh
  evaluation even if the description is unchanged, while `"New York, NY"` and
  `" new york,  NY"` still hit. Going back to an earlier location reuses that
  location's row.
- Effort (`--effort`) is **not** part of the key: changing it alone reuses the cached row.
- `--force` (with `--call-api`) skips the cache and stores a fresh evaluation.
- Rows are never updated or deleted, so history survives posting, profile, model
  and schema changes. The newest matching row wins.
- Failed calls, refusals and malformed or invalid responses are never stored.
- The location policy is re-applied to cached results (the stored row holds Claude's raw output).
- Schema upgrade: databases created before the location key get a nullable
  `location_key` column added (`ALTER TABLE ... ADD COLUMN`, in one transaction, on the
  first `--call-api` run; dry runs never migrate). Existing rows are kept unchanged with
  `location_key = NULL` as history, but are never reused, because their location is
  unknown. A table missing the original columns is rejected and left untouched.
- Bump `EVALUATION_SCHEMA_VERSION` in `src/claude_evaluator.py` when the output schema or prompt changes.
  It is currently `2` (concise format). Version-1 evaluations stay in the table as history,
  but they are never reused or shown as current, so re-run `--call-api` for jobs you want
  in the new format.

Exit codes: `0` success / dry run ready, `1` API or response failure / dry run not ready, `2` bad input.

## Batch evaluation (all eligible jobs, opt-in)

`src/batch_evaluator.py` evaluates every **active, geographically eligible** job
(location `compatible`, `review` or `boston_hybrid`) whose title relevance is **HIGH
or MEDIUM**. It uses the single-job evaluator's prompt, schema, validation and the
same evaluation cache. Each job is sent as its own request, one at a time.

```powershell
# Dry run (default): counts and the list of jobs that would be evaluated; sends nothing
python -m src.batch_evaluator --profile config/claude_candidate_profile.json

# Evaluate every eligible HIGH/MEDIUM job that has no current cached evaluation
python -m src.batch_evaluator --profile config/claude_candidate_profile.json --call-api
```

- **Posting-date cutoff:** only jobs the employer published in the last 30 days are
  evaluated, the same rule the report uses (UTC, boundary inclusive). Dates come from
  the newest `*_jobs.json` snapshots first (`--data-dir`, default the database's
  directory), then the database's `source_published_at`. Jobs with a missing or
  unverified publication date are skipped too. `updated_at` and first/last-seen times
  are never substituted. `--max-age-days N` changes the window. Skipped jobs and
  their existing evaluations stay in the database. Adding the date to the job record
  does not change cache keys or prompts, so earlier evaluations are still reused.
- Jobs with a current cached evaluation are skipped, with no request. Only new or
  changed postings are sent, where a change means description, location, profile,
  model or schema version, exactly as in the single-job cache.
  Running the command again therefore costs nothing for jobs already done.
- Each validated evaluation is appended to `evaluations` as soon as it arrives.
  Earlier rows are never updated or deleted, so stopping mid-run loses nothing.
- A failed job (timeout, rate limit, refusal, malformed or invalid response,
  save error) is logged and the batch moves on. Failures are never stored, so the next
  run retries them. The batch stops early only on authentication, permission or
  unknown-model errors, or after `--max-consecutive-failures` failures in a row
  (default 5). Jobs left over are counted as `not_attempted`.
- The final line logs `evaluated`, `cached`, `failed`, `skipped` (outside the posting-date
  window, undated, ineligible location, low relevance or incomplete record) and `not_attempted`.
- No job count is capped by default. `--limit N` optionally caps requests in one run.
- The profile path is required. The real `config/candidate_profile.json` is refused;
  use the sanitized `config/claude_candidate_profile.json`. Nothing is submitted,
  emailed or contacted, and the key, profile, prompts and descriptions are never printed.
- `--model`, `--effort`, `--timeout`, `--max-retries`, `--max-description-chars`
  behave as in the single-job CLI. There is no `--force` in batch mode.

Exit codes: `0` all succeeded / dry run ready, `1` some evaluations failed, the batch
stopped or the dry run is not ready, `2` bad input, `130` interrupted (completed
evaluations are kept).

## Job Intelligence Report (HTML)

`src/generate_report.py` writes a single self-contained HTML file from the local
database. It is **offline and read-only**: SQLite is opened with `mode=ro`, nothing is
created or migrated, and no ATS or Claude requests are made.

Generate it (run from the project root; pass the same profile used for `--call-api`
evaluations so saved evaluations can be verified as current):

```powershell
python -m src.generate_report --profile config/claude_candidate_profile.json
```

Open it in your default browser:

```powershell
Start-Process reports\job_intelligence.html
```

(or double-click `reports/job_intelligence.html`; it works without a web server or internet connection).

| Option            | Purpose                                                                          |
|-------------------|----------------------------------------------------------------------------------|
| `--profile PATH`  | Profile whose evaluations count as current. Hashed locally; its contents never appear in the report. Without it, saved evaluations are marked *unverified* and not shown or ranked. |
| `--model NAME`    | Model whose evaluations count as current (default `$CLAUDE_EVALUATOR_MODEL` or `claude-opus-5-5`). |
| `--db PATH`       | Database (default `data/jobs.db`).                                               |
| `--output PATH`   | Output file (default `reports/job_intelligence.html`).                           |
| `--max-age-days N`| Posting-age cutoff in days (default 30).                                         |
| `--data-dir PATH` | Collector snapshots read for publication dates (default: the `--db` directory, i.e. `data/`). |

**30-day posting cutoff.** Before ranking and rendering, the report keeps only jobs
whose employer publication date (Greenhouse `first_published`, Ashby `publishedAt`) is
within the last 30 days (UTC, boundary inclusive). Dates come from the newest snapshots
in `data/` (written by every `collect_jobs` run, even without `--db`) and then from the
database. Jobs with a missing or unverified date (no UTC offset, unparseable) are
excluded; `updated_at`, `first_seen_at` and `last_seen_at` are never substituted. The
counts are printed on a `Posting-date cutoff:` line and shown in the report. Nothing
is deleted from SQLite.

What the report shows:

- Summary: active jobs, geographically eligible jobs, high/medium/low title relevance
  (among eligible jobs) and jobs with a current AI evaluation.
- Eligible jobs only (`compatible`, `review`, `boston_hybrid`); `incompatible` jobs are excluded.
- Ranking: jobs with a **current** AI evaluation first, by overall fit score (highest
  first); then all other jobs by title relevance (high → medium → low). Unevaluated jobs
  are labelled **Not evaluated** and never show an AI score.
- An evaluation is *current* only if its job description, normalized location, model,
  profile and schema version all match the current inputs. Stale ones are listed as
  "Not evaluated (saved evaluation is stale)" with the reason. Evaluations made with
  `--max-description-chars` (truncated description) are not treated as current.
- Every job uses the same compact card. Collapsed, it shows only the rank, company,
  job title, AI fit score (or **Not evaluated**), location with its compatibility badge,
  title relevance and the application link. Hovering "Not evaluated" explains why (no
  evaluation, stale with the reason, or unverified).
- Evaluated jobs add a native **More details** toggle (`<details>`, closed by default,
  works without JavaScript) containing only:
  key responsibilities (≤ 3), matched skills (≤ 4), missing qualifications (≤ 3, tagged
  *Confirmed missing* or *Not established*) and transferable skills (≤ 3).
- The report does not display the AI recommendation (Apply / Review / Skip), the
  "why" explanation, sub-scores, evidence notes, company summary, evaluation date or
  location-policy messages. These are still stored in the database and in the
  evaluator's JSON output; only the report hides them.
- Default view: **Unseen** jobs, **newest first**. Views All jobs / Unseen / Seen (with
  counts), sorts Newest first / Oldest first / Highest match, search across company and
  job title, and filters for company, location, location type, relevance, AI
  recommendation and minimum match score all run in the browser, offline. The shown
  count reflects the active filters. Each card shows its posting date and a
  **Mark as seen / Mark as unseen** button; seen state is kept in the browser's
  `localStorage` (key `jobIntelligence.seenJobs.v1`), keyed by
  `ats_provider|company|ats_job_id`, so it survives report regeneration in the same
  browser. If storage is unavailable the buttons still work for the open page. The recommendation filter still works: each card carries its
  (location-policy-adjusted) recommendation as a hidden attribute. All job and AI text is HTML-escaped, only `http(s)` application links
  are rendered, and a Content-Security-Policy allows only the report's own script.
- If the `evaluations` table doesn't exist, every job is shown as Not evaluated.

`reports/` is git-ignored: the report can contain personalized AI evaluations.

## InTouch AI workflow

[`intouch/aerospace_job_intelligence.example.iml`](intouch/aerospace_job_intelligence.example.iml)
is a cleaned InTouch AI workflow that runs the whole pipeline in order, logging each
run to `logs/intouch_<UTC-run-id>.log`:

1. `collect-jobs`: `src.collect_jobs --all --db <root>\data\jobs.db`
2. `evaluate-jobs`: `src.batch_evaluator --profile config/claude_candidate_profile.json --call-api` (**paid** Claude requests)
3. `generate-report`: `src.generate_report --profile config/claude_candidate_profile.json`
4. `verify-report`: fails if `reports/job_intelligence.html` is missing or empty
5. `send-report-email`: InTouch message task that attaches the report
6. `email-confirmed`: logs that the message task returned successfully

Before importing a copy, replace both placeholders:

- `C:\path\to\intouch-job-board` with the absolute path of your clone. It appears in
  every task's `root=` line and command arguments, and in the email attachment.
- `you@example.com` with your address.

The workflow stores no credentials; SMTP is configured in InTouch and the API key
comes from `.env`. No schedule is included: add one in InTouch after a test run
succeeds. Personal `*.iml` files are git-ignored; only `*.example.iml` is tracked.

Check a workflow file (offline; it is parsed and compiled, never run):

```powershell
python -m src.intouch_workflow                       # the template (placeholders only)
python -m src.intouch_workflow path\to\your.iml      # your filled-in workflow
```

It reports every missing or wrong piece and exits `0` when complete, `1` with problems,
`2` if the file is unreadable. It checks:
- the six tasks are present once each, in order;
- every inline script compiles, names its own step and uses the same root, log folder and run marker;
- the commands use the project's `.venv` Python and include the required flags
  (`--all --db` with the database the evaluator and report read, `--call-api`, the sanitized profile);
- the commands are accepted by each tool's own argument parser;
- each task's timeouts let the script log a timeout;
- verify-report and the email attachment both point to the report path;
- the file contains no credentials, API keys or schedule.

`tests/test_intouch_workflow.py` runs the same checks on the template, breaks each
piece in turn to confirm it is reported, and executes the task scripts in a temporary
folder with the pipeline commands replaced by a harmless `print`.

## Tests

Tests use mocked HTTP responses and temporary files/databases; no network access.

```powershell
python -m unittest discover -s tests -v
```

## Adding another provider

1. Create `src/<provider>.py` with a `fetch_<provider>_jobs(board_token, company=None, timeout=30)`
   function and its own exception type, following `src/greenhouse.py` and
   `src/ashby.py` / `src/lever.py`. Reuse `get_json`, `clean_str` and `html_to_text` from
   [`src/ats_common.py`](src/ats_common.py) and return the normalized schema above.
2. Register it in `_PROVIDERS` in [`src/providers.py`](src/providers.py).
   Config validation picks it up automatically.
3. Add companies with `"ats_provider": "<provider>"` to the config.

## Project layout

```
config/companies.example.json  company list template (real config/companies.json is git-ignored)
config/candidate_profile.example.json  public profile template (real profile is git-ignored)
.env.example              template for the git-ignored .env (ANTHROPIC_API_KEY)
requirements.txt          packages for the Claude evaluators and tests
src/ats_common.py         shared HTTP JSON fetch + HTML-to-text helpers
src/greenhouse.py         Greenhouse API client + normalization
src/ashby.py              Ashby API client + normalization
src/lever.py              Lever API client + normalization
src/providers.py          provider dispatcher
src/company_config.py     config loading / validation / selection
src/database.py           SQLite schema and upserts
src/collect_jobs.py       CLI entry point
src/location_filter.py    deterministic location classifier + read-only report
src/relevance_filter.py   weighted title-relevance classifier + read-only report
src/candidate_profile.py  local candidate profile loader / validator
src/claude_evaluator.py   single-job Claude evaluator CLI (dry run by default)
src/batch_evaluator.py    batch evaluation of eligible HIGH/MEDIUM jobs (dry run by default)
src/intouch_workflow.py   offline checker for InTouch AI workflow (.iml) files
intouch/aerospace_job_intelligence.example.iml  cleaned InTouch AI workflow template
src/evaluation_store.py   evaluations table, hashing and cache
src/generate_report.py    offline HTML Job Intelligence Report (read-only)
src/posting_dates.py      publication-date parsing and the 30-day report cutoff
reports/                  generated HTML reports (git-ignored)
tests/                    unittest suites (offline)
data/                     snapshots and jobs.db (git-ignored)
```
