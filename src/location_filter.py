"""Deterministic job-location classifier (no network, no LLM).

    classify_location("New York, NY")       -> {"classification": "compatible", ...}
    classify_location("Boston, MA/Hybrid")  -> {"classification": "boston_hybrid", ...}

Classifications, best first:
    compatible     NYC (any borough) or explicitly US-wide remote
    boston_hybrid  Boston area, explicitly hybrid
    review         missing/ambiguous: unspecified remote, country-only US, etc.
    incompatible   anywhere else in the US or abroad

How it works: a location string is split into *segments* on ';' (multiple
postings locations), and each segment into *components* on ',', '/', '()',
'|', spaced dashes, ' or ', ' and ' and '&'. Components are normalized
(lowercase, periods removed, punctuation -> spaces) and compared as whole
phrases against small vocabularies, so 'York' never matches 'Yorkshire' and
'New York' never matches 'New York Mills'. The best segment wins.

Read-only report:
    python -m src.location_filter --db data/jobs.db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

COMPATIBLE = "compatible"
BOSTON_HYBRID = "boston_hybrid"
REVIEW = "review"
INCOMPATIBLE = "incompatible"
CLASSIFICATIONS = (COMPATIBLE, BOSTON_HYBRID, REVIEW, INCOMPATIBLE)  # best first

# --------------------------------------------------------------------------- #
# Vocabularies (normalized phrases)
# --------------------------------------------------------------------------- #

REMOTE_WORDS = {"remote", "fully remote", "100 remote", "wfh", "work from home", "telecommute", "virtual"}
HYBRID_WORDS = {"hybrid"}
OTHER_MODIFIERS = {"on site", "onsite", "in office", "office", "in person", "full time", "part time"}
_MODIFIER_PHRASES = sorted(REMOTE_WORDS | HYBRID_WORDS | OTHER_MODIFIERS, key=lambda p: -len(p.split()))

US_COUNTRY = {"us", "usa", "united states", "united states of america", "america", "u s", "u s a"}

# "new york" itself is handled separately: it is NYC unless an unknown place
# (another city) precedes it, as in "Albany, New York".
NYC_NAMES = {
    "nyc", "new york city", "manhattan", "brooklyn", "queens", "bronx", "the bronx",
    "staten island", "long island city",
}

BOSTON_NAMES = {"boston"}
# Neighbouring cities count as Boston-area only when paired with MA/Massachusetts.
BOSTON_SUBURBS = {
    "cambridge", "somerville", "waltham", "burlington", "woburn", "lexington", "bedford",
    "quincy", "newton", "brookline", "watertown", "medford", "needham", "wellesley",
    "framingham", "billerica", "andover", "chelmsford", "lowell", "wilmington", "marlborough",
    "boxborough", "devens", "natick", "braintree", "dedham", "everett", "chelsea", "malden",
}
MASSACHUSETTS = {"ma", "mass", "massachusetts"}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "new york state", "north carolina", "north dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia", "washington dc", "dc",
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy", "mass",
}
NEW_YORK_STATE = {"ny", "new york", "new york state"}

# Places that say nothing definite about where the job is.
VAGUE = {
    "other", "anywhere", "global", "worldwide", "international", "multiple locations",
    "multiple", "various", "various locations", "flexible", "tbd", "to be determined",
    "hq", "headquarters", "north america", "americas", "nationwide", "location", "locations",
    "n a", "na", "none", "not specified",
}

_SEGMENT_SPLIT = re.compile(r"[;\n]+")
_COMPONENT_SPLIT = re.compile(r"\s+(?:-|–|—|or|and)\s+|[,/()|&:\[\]]|–|—", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = text.lower().replace(".", "").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _strip_area_words(phrase: str) -> str:
    """'greater boston area' -> 'boston', 'new york city metro area' -> 'new york city'."""
    phrase = re.sub(r"^greater ", "", phrase)
    return re.sub(r" (?:metropolitan area|metro area|metro|area|region)$", "", phrase)


def _parse_segment(segment: str) -> tuple[set[str], list[str]]:
    """Return (modifier flags, place components) for one segment."""
    flags: set[str] = set()
    places: list[str] = []
    for raw in _COMPONENT_SPLIT.split(segment):
        comp = f" {_normalize(raw)} "
        for phrase in _MODIFIER_PHRASES:  # longest first, whole-phrase only
            if f" {phrase} " in comp:
                comp = comp.replace(f" {phrase} ", " ")
                if phrase in REMOTE_WORDS:
                    flags.add("remote")
                elif phrase in HYBRID_WORDS:
                    flags.add("hybrid")
        comp = _strip_area_words(" ".join(comp.split()))
        if comp:
            places.append(comp)
    return flags, places


def _place_kinds(places: list[str]) -> list[str]:
    """Label each place component: nyc, boston, us, us_state, vague or other."""
    has_ma = any(p in MASSACHUSETTS for p in places)
    kinds: list[str] = []
    for place in places:
        base = place.removesuffix(" ny").removesuffix(" ma")  # "new york ny", "boston ma"
        if place in NYC_NAMES or base in NYC_NAMES:
            kinds.append("nyc")
        elif base == "new york" or place == "new york":
            kinds.append("state" if "other" in kinds else "nyc")
        elif base in BOSTON_NAMES or (base in BOSTON_SUBURBS and (has_ma or place.endswith(" ma"))):
            kinds.append("boston")
        elif place in US_COUNTRY:
            kinds.append("us")
        elif place in US_STATES:
            kinds.append("state")
        elif place in VAGUE:
            kinds.append("vague")
        else:
            kinds.append("other")  # a city, foreign country or region we don't special-case
    return kinds


def _classify_segment(segment: str) -> tuple[str, str]:
    flags, places = _parse_segment(segment)
    kinds = set(_place_kinds(places))
    remote, hybrid = "remote" in flags, "hybrid" in flags

    if "nyc" in kinds:
        return COMPATIBLE, "New York City location"

    if remote:
        if "state" in kinds or "boston" in kinds:
            return REVIEW, "remote, but tied to a specific US location outside NYC"
        if "us" in kinds:
            return COMPATIBLE, "explicitly US remote"
        if "other" in kinds:
            return INCOMPATIBLE, "remote outside the US"
        return REVIEW, "remote with unspecified geography"

    if "boston" in kinds:
        if hybrid:
            return BOSTON_HYBRID, "Boston area, explicitly hybrid"
        return INCOMPATIBLE, "Boston area but not explicitly hybrid"
    if "other" in kinds or ("state" in kinds and not set(places) <= NEW_YORK_STATE | US_COUNTRY):
        if "state" in kinds or "us" in kinds:
            return INCOMPATIBLE, "US location outside NYC"
        return INCOMPATIBLE, "location outside NYC (other US city or abroad)"
    if "state" in kinds:
        return REVIEW, "New York State without a city"
    if "us" in kinds:
        return REVIEW, "United States only, remote status not stated"
    return REVIEW, "ambiguous location"


def classify_location(location: str | None) -> dict[str, str]:
    """Classify a posting's location string. See module docstring for rules."""
    if not isinstance(location, str) or not _normalize(location):
        return {"classification": REVIEW, "reason": "location missing"}

    segments = [s.strip() for s in _SEGMENT_SPLIT.split(location) if _normalize(s)]
    results = [(seg, *_classify_segment(seg)) for seg in segments]
    seg, classification, reason = min(results, key=lambda r: CLASSIFICATIONS.index(r[1]))
    if len(results) > 1:
        reason = f"{reason} ({seg})"
    return {"classification": classification, "reason": reason}


# --------------------------------------------------------------------------- #
# Read-only CLI report
# --------------------------------------------------------------------------- #

def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open an existing database read-only; never creates or modifies the file."""
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT c.name AS company, j.title, j.location
        FROM jobs j JOIN companies c ON c.id = j.company_id
        WHERE j.is_active = 1
        ORDER BY c.name, j.title, j.id
        """
    ).fetchall()


def build_report(jobs, show: tuple[str, ...] = (COMPATIBLE, REVIEW)) -> str:
    classified = [(job, classify_location(job["location"])) for job in jobs]
    counts = Counter(result["classification"] for _, result in classified)

    lines = [f"Location classification for {len(classified)} active jobs:"]
    lines += [f"  {name:<14} {counts.get(name, 0):>5}" for name in CLASSIFICATIONS]
    for name in show:
        group = [(job, r) for job, r in classified if r["classification"] == name]
        lines += ["", f"== {name} ({len(group)}) =="]
        for job, result in group:
            line = f"  {job['company']} | {job['title']} | {job['location'] or '(no location)'}"
            if name == REVIEW:
                line += f"  [{result['reason']}]"
            lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify job locations in the jobs database (read-only).")
    parser.add_argument("--db", type=Path, required=True, help="SQLite database, e.g. data/jobs.db")
    parser.add_argument(
        "--show", default=f"{COMPATIBLE},{REVIEW}",
        help=f"Comma-separated classifications to list (default: compatible,review; choices: {','.join(CLASSIFICATIONS)})",
    )
    args = parser.parse_args(argv)

    show = tuple(s.strip() for s in args.show.split(",") if s.strip())
    unknown = [s for s in show if s not in CLASSIFICATIONS]
    if unknown:
        parser.error(f"unknown classification(s) in --show: {', '.join(unknown)}")
    if not args.db.is_file():
        print(f"ERROR: database not found: {args.db}", file=sys.stderr)
        return 1

    try:
        conn = open_readonly(args.db)
        try:
            jobs = load_jobs(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"ERROR: cannot read {args.db}: {exc}", file=sys.stderr)
        return 1

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # job titles may contain non-ASCII characters
    print(build_report(jobs, show))
    return 0


if __name__ == "__main__":
    sys.exit(main())
