"""Load and validate the local candidate profile (config/candidate_profile.json).

Local only: this module reads a JSON file and checks its structure. It makes no
network requests and sends the profile nowhere.

    python -m src.candidate_profile                       # validate the default profile
    python -m src.candidate_profile --profile path.json   # validate another file

Exit codes: 0 valid, 1 invalid, 2 file missing or unreadable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = PROJECT_ROOT / "config" / "candidate_profile.json"
EXAMPLE_PROFILE = PROJECT_ROOT / "config" / "candidate_profile.example.json"

STRING_LIST_SECTIONS = ("target_roles", "aerospace_experience", "technical_skills", "transferable_skills")
REQUIRED_SECTIONS = (
    "target_roles", "preferred_locations", "years_experience", "education",
    "aerospace_experience", "technical_skills", "transferable_skills",
)
OPTIONAL_SECTIONS = ("schema_version", "experience_evidence")
EDUCATION_FIELDS = ("degree", "field", "institution", "status")
EDUCATION_OPTIONAL_FIELDS = ("year",)  # completion (or expected completion) year
EDUCATION_STATUSES = ("completed", "in_progress")
EDUCATION_YEAR_RANGE = (1950, 2100)
EVIDENCE_FIELDS = ("skill", "evidence")

# Guard against contact details accidentally pasted into the profile.
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", re.IGNORECASE)
_PHONE = re.compile(r"(?:\+?\d[\s().-]*){10,}")


class ProfileError(Exception):
    """The profile file is missing, unreadable or invalid. ``errors`` lists every problem."""

    def __init__(self, message: str, errors: list[str] | None = None):
        self.errors = errors or [message]
        super().__init__(message if not errors else message + "\n  - " + "\n  - ".join(errors))


class ProfileFileError(ProfileError):
    """The file could not be read or parsed as JSON."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_text(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}: expected a non-empty string, got {type(value).__name__} {value!r:.40}")
    elif _EMAIL.search(value) or _PHONE.search(value):
        errors.append(f"{where}: looks like contact information (email/phone); remove it")


def _check_string_list(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{where}: expected a list of strings, got {type(value).__name__}")
        return
    if not value:
        errors.append(f"{where}: must contain at least one entry")
    seen: set[str] = set()
    for i, item in enumerate(value):
        _check_text(item, f"{where}[{i}]", errors)
        if isinstance(item, str):
            key = item.strip().lower()
            if key in seen:
                errors.append(f"{where}[{i}]: duplicate entry {item!r}")
            seen.add(key)


def validate_profile(data: Any) -> list[str]:
    """Return a list of human-readable problems; an empty list means the profile is valid."""
    if not isinstance(data, dict):
        return [f"profile: expected a JSON object, got {type(data).__name__}"]

    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in data:
            errors.append(f"{section}: required section is missing")

    for section in STRING_LIST_SECTIONS:
        if section in data:
            _check_string_list(data[section], section, errors)

    if "preferred_locations" in data:
        locations = data["preferred_locations"]
        if not isinstance(locations, dict):
            errors.append("preferred_locations: expected an object with 'primary' and optional 'secondary' lists")
        else:
            if "primary" not in locations:
                errors.append("preferred_locations.primary: required list is missing")
            else:
                _check_string_list(locations["primary"], "preferred_locations.primary", errors)
            if "secondary" in locations and locations["secondary"] != []:
                _check_string_list(locations["secondary"], "preferred_locations.secondary", errors)
            for key in sorted(set(locations) - {"primary", "secondary"}):
                errors.append(f"preferred_locations.{key}: unknown key (allowed: primary, secondary)")

    if "years_experience" in data:
        years = data["years_experience"]
        if not isinstance(years, dict) or not years:
            errors.append("years_experience: expected a non-empty object mapping discipline -> years")
        else:
            for key, value in years.items():
                if not key.strip():
                    errors.append("years_experience: discipline names must be non-empty")
                elif not _is_number(value) or value < 0 or value > 60:
                    errors.append(f"years_experience.{key}: expected a number between 0 and 60, got {value!r}")

    if "education" in data:
        education = data["education"]
        if not isinstance(education, list) or not education:
            errors.append("education: expected a non-empty list of objects")
        else:
            for i, entry in enumerate(education):
                where = f"education[{i}]"
                if not isinstance(entry, dict):
                    errors.append(f"{where}: expected an object, got {type(entry).__name__}")
                    continue
                for field in EDUCATION_FIELDS:
                    if field not in entry:
                        errors.append(f"{where}.{field}: required field is missing")
                    else:
                        _check_text(entry[field], f"{where}.{field}", errors)
                status = entry.get("status")
                if isinstance(status, str) and status.strip() and status not in EDUCATION_STATUSES:
                    errors.append(f"{where}.status: must be one of {', '.join(EDUCATION_STATUSES)}, got {status!r}")
                if "year" in entry:
                    year, (low, high) = entry["year"], EDUCATION_YEAR_RANGE
                    if not isinstance(year, int) or isinstance(year, bool) or not low <= year <= high:
                        errors.append(f"{where}.year: expected an integer year between {low} and {high}, got {year!r}")
                for key in sorted(set(entry) - set(EDUCATION_FIELDS) - set(EDUCATION_OPTIONAL_FIELDS)):
                    errors.append(f"{where}.{key}: unknown field "
                                  f"(allowed: {', '.join(EDUCATION_FIELDS + EDUCATION_OPTIONAL_FIELDS)})")

    # Optional: structured examples of demonstrated capabilities.
    if "experience_evidence" in data:
        evidence = data["experience_evidence"]
        if not isinstance(evidence, list):
            errors.append(f"experience_evidence: expected a list of objects, got {type(evidence).__name__}")
        else:
            for i, entry in enumerate(evidence):
                where = f"experience_evidence[{i}]"
                if not isinstance(entry, dict):
                    errors.append(f"{where}: expected an object with 'skill' and 'evidence', got {type(entry).__name__}")
                    continue
                for field in EVIDENCE_FIELDS:
                    if field not in entry:
                        errors.append(f"{where}.{field}: required field is missing")
                    else:
                        _check_text(entry[field], f"{where}.{field}", errors)
                for key in sorted(set(entry) - set(EVIDENCE_FIELDS)):
                    errors.append(f"{where}.{key}: unknown field (allowed: skill, evidence)")

    if "schema_version" in data and data["schema_version"] != 1:
        errors.append(f"schema_version: unsupported version {data['schema_version']!r} (expected 1)")
    return errors


def load_profile(path: str | Path = DEFAULT_PROFILE) -> dict[str, Any]:
    """Read and validate a profile. Raises ProfileFileError / ProfileError with clear messages."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ProfileFileError(
            f"Profile not found: {path}. Copy {EXAMPLE_PROFILE.name} to {path.name} and fill it in."
        ) from exc
    except OSError as exc:
        raise ProfileFileError(f"Cannot read profile {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProfileFileError(f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

    errors = validate_profile(data)
    if errors:
        raise ProfileError(f"Invalid candidate profile {path} ({len(errors)} problem(s)):", errors)
    return data


def summarize(profile: dict[str, Any]) -> str:
    """Section counts only, so the CLI output doesn't echo profile contents."""
    locations = profile["preferred_locations"]
    return "\n".join([
        f"  target_roles:         {len(profile['target_roles'])}",
        f"  preferred_locations:  {len(locations['primary'])} primary, {len(locations.get('secondary', []))} secondary",
        f"  years_experience:     {len(profile['years_experience'])} discipline(s)",
        f"  education:            {len(profile['education'])} entry/entries",
        f"  aerospace_experience: {len(profile['aerospace_experience'])}",
        f"  technical_skills:     {len(profile['technical_skills'])}",
        f"  transferable_skills:  {len(profile['transferable_skills'])}",
        f"  experience_evidence:  {len(profile.get('experience_evidence', []))} (optional)",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the local candidate profile (offline).")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="Profile JSON path.")
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except ProfileFileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ProfileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Candidate profile OK: {args.profile}\n{summarize(profile)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
