"""Deterministic, weighted job-title relevance classifier (no network, no LLM).

    classify_relevance("Mission Operations Engineer")
    -> {"classification": "high", "score": 7.0,
        "reason": "phrase 'mission operations' +5; role 'engineer' +2 | total 7"}

Method
------
1. Normalize: lowercase, '&' -> 'and', periods dropped, other punctuation ->
   spaces, simple synonyms ('ops' -> 'operations') and plurals ('sensors' ->
   'sensor'). Matching is on whole tokens, never substrings, so 'sales' does not
   match 'Salesforce' and 'hr' does not match 'Shreveport'.
2. Phrases are matched first, longest first. Tokens inside a matched phrase are
   consumed and never scored again as single words (no double counting).
3. Remaining tokens are scored as domain words, generic technical words,
   roles or negative signals. Each distinct term counts once, and only the
   single strongest role counts.
4. Negative signals (marketing, legal, ...) are damped when the title has a
   technical role, so 'Sales Engineer' is not automatically low.
5. HIGH needs both a total of at least HIGH_THRESHOLD *and* aerospace domain
   evidence of at least MIN_DOMAIN_FOR_HIGH, so generic words alone ('Senior
   Systems Operations Engineer') can't make a title high.
   LOW needs a total of at most LOW_THRESHOLD. A title with no recognized terms
   (or no title) is MEDIUM for manual review.

Seniority words (senior, staff, principal, lead, ...) are neutral.
All weights and thresholds are module-level constants below.

Read-only report:
    python -m src.relevance_filter --db data/jobs.db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from src import location_filter

HIGH = "high"
MEDIUM = "medium"
LOW = "low"
CLASSIFICATIONS = (HIGH, MEDIUM, LOW)

# --------------------------------------------------------------------------- #
# Tunable weights and thresholds
# --------------------------------------------------------------------------- #

HIGH_THRESHOLD = 4.5        # total score needed for "high" ...
MIN_DOMAIN_FOR_HIGH = 2.0   # ... plus at least this much aerospace domain evidence
LOW_THRESHOLD = 0.0         # total score at or below this is "low"
NEGATIVE_DAMPING = 0.25     # negatives are multiplied by this when a technical role is present

# Aerospace / mission phrases: strong domain evidence.
DOMAIN_PHRASES: dict[str, float] = {
    "mission operations": 5, "satellite operations": 5, "spacecraft operations": 5,
    "payload operations": 5, "flight operations": 5, "signal processing": 5,
    "image processing": 4, "systems integration": 5, "sensor characterization": 5,
    "payload performance": 5, "mission systems": 5, "satellite systems": 5,
    "spacecraft systems": 5, "space systems": 5, "radar systems": 5, "rf systems": 5,
    "ground systems": 5, "ground station": 5, "ground software": 4, "flight software": 5,
    "flight dynamics": 5, "mission planning": 5, "remote sensing": 5,
    "orbit determination": 5, "attitude determination": 5, "attitude control": 5,
    "guidance navigation and control": 5, "calibration design": 5,
    # EO/IR sensing ("EO/IR", "EO-IR", "Electro-Optical", "Electro-Optic(s)", "Multi-Spectral").
    "eo ir": 5, "electro optical": 5, "electro optic": 5, "electro optics": 5, "multi spectral": 3,
    "integration and verification": 5, "integration and test": 5, "i and t": 5,
    "verification and validation": 5, "v and v": 5, "hardware in the loop": 4,
    "mission assurance": 4, "test automation": 4,
    # Contextual exception: QA automation is engineering, not administrative QA.
    "quality assurance automation": 3, "qa automation": 3,
}

# Aerospace / instrument words: domain evidence.
DOMAIN_WORDS: dict[str, float] = {
    # strong
    "satellite": 3, "spacecraft": 3, "payload": 3, "radar": 3, "telemetry": 3, "sensor": 3,
    "optical": 3, "optics": 3, "infrared": 3, "calibration": 3, "avionics": 3, "propulsion": 3,
    "orbital": 3, "orbit": 3, "gnc": 3, "rf": 3, "fpga": 3, "antenna": 3, "instrument": 3,
    "instrumentation": 3, "photonics": 3, "imaging": 3, "hyperspectral": 3, "multispectral": 3,
    "eoir": 3, "electrooptical": 3, "electrooptic": 3, "mwir": 3, "lwir": 3, "swir": 3,
    "sar": 3, "lidar": 3, "anomaly": 3, "constellation": 3, "astrodynamics": 3, "aerospace": 3,
    "spaceflight": 3, "microwave": 3, "geospatial": 3,
    # moderate
    "space": 2, "flight": 2, "mission": 2, "integration": 2, "verification": 2,
    "validation": 2, "testing": 2, "test": 2, "automation": 2, "signal": 2,
    "firmware": 2, "embedded": 2, "simulation": 2, "controls": 2, "guidance": 2,
    "navigation": 2, "characterization": 2, "metrology": 2, "thermal": 2, "launch": 2,
}

# Generic technical words: add to the score but are NOT domain evidence.
GENERIC_WORDS: dict[str, float] = {
    "operations": 1, "systems": 1, "software": 1, "hardware": 1.5, "electrical": 1.5,
    "mechanical": 1.5, "manufacturing": 1.5, "data": 1, "engineering": 1, "reliability": 1,
    "network": 1, "infrastructure": 1, "design": 1, "research": 1, "performance": 1,
    "ground": 1, "product": 0.5, "technical": 0.5, "cloud": 0.5, "security": 0.5,
}

# Roles: only the single strongest role found counts.
ROLE_PHRASES: dict[str, float] = {"technical lead": 2, "tech lead": 2}
ROLE_WORDS: dict[str, float] = {
    "engineer": 2, "scientist": 2, "physicist": 2, "analyst": 1.5, "developer": 1.5,
    "architect": 1.5, "researcher": 1.5, "specialist": 1, "technician": 1, "programmer": 1,
    "operator": 1, "lead": 1, "manager": 0, "director": 0,
}
TECHNICAL_ROLES = {"engineer", "scientist", "physicist", "developer", "architect",
                   "researcher", "programmer", "technical lead", "tech lead"}

# Negative signals: clearly unrelated business functions.
NEGATIVE_PHRASES: dict[str, float] = {
    "human resources": -6, "talent acquisition": -6, "business development": -6,
    "social media": -6, "public relations": -6, "customer success": -4,
    "account executive": -6, "account manager": -4, "executive assistant": -6,
    "people operations": -6, "office manager": -6, "government relations": -6,
    "investor relations": -6, "supply chain": -3, "regulatory affairs": -6,
    "subcontract administration": -6, "subcontracts administration": -6,
    "quality assurance": -1,  # admin QA; "quality assurance automation" overrides
}
NEGATIVE_WORDS: dict[str, float] = {
    "marketing": -6, "sales": -6, "finance": -6, "financial": -6, "accounting": -6,
    "accountant": -6, "legal": -6, "attorney": -6, "counsel": -6, "paralegal": -6,
    "lawyer": -6, "recruiting": -6, "recruiter": -6, "recruitment": -6, "procurement": -6,
    "subcontracts": -6, "subcontract": -6, "subcontracting": -6, "regulatory": -4,
    "purchasing": -6, "buyer": -4, "hr": -6, "compliance": -6, "capture": -6, "proposal": -6,
    "payroll": -6, "tax": -6, "treasury": -6, "contracts": -6, "receptionist": -6,
    "events": -4, "brand": -4, "copywriter": -6, "bookkeeper": -6, "facilities": -4,
    "benefits": -4, "qa": -1,
}

# Seniority is neutral; listed so the reason can say so.
SENIORITY_WORDS = {"senior", "sr", "staff", "principal", "lead", "chief", "junior", "jr",
                   "associate", "head", "distinguished", "i", "ii", "iii", "iv", "1", "2", "3"}

SYNONYMS = {
    "ops": "operations", "operation": "operations", "system": "systems", "eng": "engineer",
    "engr": "engineer", "sw": "software", "hw": "hardware", "sr": "senior",
    "satcom": "satellite", "comms": "communications", "dev": "developer",
}

# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

_ALL_PHRASES = {**DOMAIN_PHRASES, **ROLE_PHRASES, **NEGATIVE_PHRASES}
_PHRASES_BY_LENGTH = sorted(_ALL_PHRASES, key=lambda p: (-len(p.split()), p))
_VOCAB = (set(DOMAIN_WORDS) | set(GENERIC_WORDS) | set(ROLE_WORDS) | set(NEGATIVE_WORDS)
          | {w for p in _ALL_PHRASES for w in p.split()})


def _tokens(title: str) -> list[str]:
    text = title.lower().replace("&", " and ").replace(".", "").replace("'", "")
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    out = []
    for tok in tokens:
        tok = SYNONYMS.get(tok, tok)
        if tok not in _VOCAB and len(tok) > 3 and tok.endswith("s") and tok[:-1] in _VOCAB:
            tok = tok[:-1]  # simple plural: sensors -> sensor
        out.append(tok)
    return out


def _fmt(weight: float) -> str:
    return f"{weight:+g}"


def classify_relevance(title: str | None) -> dict:
    """Score a job title. Returns classification, score and a matched-terms reason."""
    if not isinstance(title, str) or not title.strip():
        return {"classification": MEDIUM, "score": 0.0, "reason": "title missing; manual review"}

    tokens = _tokens(title)
    consumed = [False] * len(tokens)
    domain: list[tuple[str, str, float]] = []    # (kind, term, weight)
    generic: list[tuple[str, str, float]] = []
    roles: list[tuple[str, float]] = []
    negatives: list[tuple[str, float]] = []

    # 1. Phrases, longest first. Every occurrence is consumed (so its words are
    #    never re-scored individually) but each phrase is counted only once.
    for phrase in _PHRASES_BY_LENGTH:
        words = phrase.split()
        n = len(words)
        matched = False
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == words and not any(consumed[i:i + n]):
                consumed[i:i + n] = [True] * n
                matched = True
        if not matched:
            continue
        if phrase in DOMAIN_PHRASES:
            domain.append(("phrase", phrase, DOMAIN_PHRASES[phrase]))
        elif phrase in ROLE_PHRASES:
            roles.append((phrase, ROLE_PHRASES[phrase]))
        else:
            negatives.append((phrase, NEGATIVE_PHRASES[phrase]))

    # 2. Remaining single words, each distinct term once.
    seen: set[str] = set()
    seniority: list[str] = []
    for tok, used in zip(tokens, consumed):
        if used or tok in seen:
            continue
        seen.add(tok)
        if tok in DOMAIN_WORDS:
            domain.append(("word", tok, DOMAIN_WORDS[tok]))
        elif tok in GENERIC_WORDS:
            generic.append(("generic", tok, GENERIC_WORDS[tok]))
        elif tok in NEGATIVE_WORDS:
            negatives.append((tok, NEGATIVE_WORDS[tok]))
        elif tok in ROLE_WORDS:
            roles.append((tok, ROLE_WORDS[tok]))
        elif tok in SENIORITY_WORDS:
            seniority.append(tok)

    # 3. Only the strongest role counts ('lead' only when nothing better is present).
    best_role = max(roles, key=lambda r: r[1]) if roles else None
    technical = any(r in TECHNICAL_ROLES for r, _ in roles)
    damping = NEGATIVE_DAMPING if technical else 1.0

    domain_score = sum(w for _, _, w in domain)
    generic_score = sum(w for _, _, w in generic)
    role_score = best_role[1] if best_role else 0.0
    negative_score = sum(w for _, w in negatives) * damping
    score = round(domain_score + generic_score + role_score + negative_score, 2)

    parts = [f"phrase '{t}' {_fmt(w)}" for k, t, w in domain if k == "phrase"]
    parts += [f"domain '{t}' {_fmt(w)}" for k, t, w in domain if k == "word"]
    parts += [f"generic '{t}' {_fmt(w)}" for _, t, w in generic]
    if best_role:
        parts.append(f"role '{best_role[0]}' {_fmt(best_role[1])}")
    for term, w in negatives:
        parts.append(f"negative '{term}' {_fmt(w * damping)}"
                     + (" (damped: technical role)" if damping != 1.0 else ""))

    if not parts:
        return {"classification": MEDIUM, "score": 0.0,
                "reason": "no recognized terms" + (f" (seniority only: {', '.join(seniority)})" if seniority else "")
                          + "; manual review"}

    if score >= HIGH_THRESHOLD and domain_score >= MIN_DOMAIN_FOR_HIGH:
        classification = HIGH
    elif score <= LOW_THRESHOLD:
        classification = LOW
    else:
        classification = MEDIUM

    reason = "; ".join(parts) + f" | total {score:g}"
    if score >= HIGH_THRESHOLD and classification != HIGH:
        reason += " (no aerospace domain term, capped at medium)"
    return {"classification": classification, "score": score, "reason": reason}


# --------------------------------------------------------------------------- #
# Read-only CLI report
# --------------------------------------------------------------------------- #

ELIGIBLE_LOCATIONS = (location_filter.COMPATIBLE, location_filter.REVIEW, location_filter.BOSTON_HYBRID)


def load_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """One row per ATS posting (company + ats_job_id); identical titles stay separate."""
    return conn.execute(
        """
        SELECT j.id, c.name AS company, c.ats_provider, j.ats_job_id, j.title, j.location,
               j.application_url
        FROM jobs j JOIN companies c ON c.id = j.company_id
        WHERE j.is_active = 1
        ORDER BY c.name, j.title, j.ats_job_id
        """
    ).fetchall()


def prioritize(jobs) -> tuple[list[dict], Counter]:
    """Classify location then relevance. Returns (eligible rows, location counts)."""
    location_counts: Counter = Counter()
    eligible: list[dict] = []
    for job in jobs:
        loc = location_filter.classify_location(job["location"])
        location_counts[loc["classification"]] += 1
        if loc["classification"] not in ELIGIBLE_LOCATIONS:
            continue
        rel = classify_relevance(job["title"])
        eligible.append({
            "company": job["company"], "ats_provider": job["ats_provider"],
            "ats_job_id": job["ats_job_id"], "title": job["title"], "location": job["location"],
            "location_class": loc["classification"], **rel,
        })
    eligible.sort(key=lambda r: (CLASSIFICATIONS.index(r["classification"]), -r["score"],
                                 r["company"], r["title"], r["ats_job_id"]))
    return eligible, location_counts


def build_report(jobs, show: tuple[str, ...] = (HIGH, MEDIUM)) -> str:
    eligible, location_counts = prioritize(jobs)
    counts = Counter(r["classification"] for r in eligible)
    total = sum(location_counts.values())
    excluded = total - len(eligible)

    lines = [
        f"{total} active jobs; {len(eligible)} with eligible locations "
        f"({', '.join(ELIGIBLE_LOCATIONS)}), {excluded} excluded as {location_filter.INCOMPATIBLE}.",
        "Relevance of eligible jobs:",
    ]
    lines += [f"  {name:<7} {counts.get(name, 0):>5}" for name in CLASSIFICATIONS]
    for name in show:
        group = [r for r in eligible if r["classification"] == name]
        lines += ["", f"== {name} ({len(group)}) =="]
        for r in group:
            lines.append(
                f"  [{r['score']:>5g}] {r['company']} | {r['title']} | "
                f"{r['location'] or '(no location)'} ({r['location_class']}) | "
                f"{r['ats_provider']}:{r['ats_job_id']}"
            )
            lines.append(f"          {r['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prioritize location-eligible jobs by title relevance (read-only)."
    )
    parser.add_argument("--db", type=Path, required=True, help="SQLite database, e.g. data/jobs.db")
    parser.add_argument("--show", default=f"{HIGH},{MEDIUM}",
                        help="Comma-separated relevance classes to list (default: high,medium)")
    args = parser.parse_args(argv)

    show = tuple(s.strip() for s in args.show.split(",") if s.strip())
    unknown = [s for s in show if s not in CLASSIFICATIONS]
    if unknown:
        parser.error(f"unknown classification(s) in --show: {', '.join(unknown)}")
    if not args.db.is_file():
        print(f"ERROR: database not found: {args.db}", file=sys.stderr)
        return 1

    try:
        conn = location_filter.open_readonly(args.db)
        try:
            jobs = load_jobs(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"ERROR: cannot read {args.db}: {exc}", file=sys.stderr)
        return 1

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(build_report(jobs, show))
    return 0


if __name__ == "__main__":
    sys.exit(main())
