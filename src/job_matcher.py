"""Local, deterministic job-matching MVP (no network, no LLM).

Compares the local candidate profile (config/candidate_profile.json) with stored
job descriptions and produces heuristic fit scores. Scores are estimates of
overlap between documented qualifications and stated requirements; they are
NOT hiring probabilities.

Pipeline
--------
1. Candidate model: every profile text (technical/transferable skills,
   aerospace experience, experience evidence) is scanned against a qualification
   CATALOG of normalized aliases. Each capability remembers its source; capabilities
   backed by experience or evidence get full credit, skill-list-only ones slightly
   less. Target roles are interests, not experience, and give no credit.
2. Requirement extraction: the description is split into sections by header
   ("What you bring", "Qualifications", "Nice-to-have", "Responsibilities", ...).
   Only statements in requirement sections earn qualification credit; text in
   "About us", benefits, salary or EEO sections is ignored. If a description has
   no recognizable requirement section, requirement-like sentences ("experience
   with", "degree in", "N+ years") are used and flagged as inferred.
3. Scoring (each 0-100): technical (incl. degree), aerospace/sensor,
   responsibilities & seniority, location. Overall = weighted sum (WEIGHTS).
4. Gaps: skills the profile doesn't mention are "not established" (no credit,
   no assumption either way). Years or degree requirements the profile can be
   checked against are "confirmed gaps". Specialized skills (FPGA, embedded,
   RF hardware, ...) never receive credit unless the profile supports them.

Read-only report:
    python -m src.job_matcher --db data/jobs.db --profile config/candidate_profile.json
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import candidate_profile, location_filter, relevance_filter

# --------------------------------------------------------------------------- #
# Tunable weights and thresholds
# --------------------------------------------------------------------------- #

WEIGHTS = {"technical": 0.40, "aerospace": 0.30, "responsibility": 0.20, "location": 0.10}

APPLY_THRESHOLD = 70          # overall score needed for "Apply" (plus no blocking issues)
SKIP_THRESHOLD = 45           # overall score below this is "Skip"
MAX_REQUIRED_GAPS_FOR_APPLY = 1   # more unmet required items than this blocks "Apply"
NEUTRAL_SCORE = 50            # used when a component can't be assessed

REQUIRED_WEIGHT = 1.0
PREFERRED_WEIGHT = 0.5
RESPONSIBILITY_AEROSPACE_WEIGHT = 0.5   # aerospace items found only in responsibilities

EVIDENCE_CREDIT = 1.0         # capability backed by experience or experience_evidence
SKILL_LIST_CREDIT = 0.85      # capability only listed as a skill
IMPLIED_CREDIT = 0.85         # e.g. PostgreSQL -> SQL, TypeScript -> JavaScript
TRANSFERABLE_CREDIT = 0.4     # related capability (e.g. MATLAB -> Simulink)

LOCATION_SCORES = {
    location_filter.COMPATIBLE: 100,
    location_filter.BOSTON_HYBRID: 70,
    location_filter.REVIEW: 40,       # ambiguous: never treated as confirmed compatible
    location_filter.INCOMPATIBLE: 0,
}

SENIORITY_PENALTY_PER_YEAR = 7    # responsibility-score points per missing year
SENIORITY_MAX_PENALTY = 35
SENIORITY_APPLY_BLOCK_YEARS = 3   # a gap this large blocks "Apply"
OVERQUALIFIED_PENALTY = 10        # intern/junior roles when the profile has >= 3 years
TITLE_IMPLIED_YEARS = {
    "intern": 0, "internship": 0, "junior": 0, "jr": 0, "entry": 0, "associate": 1,
    "ii": 2, "iii": 5, "iv": 7, "senior": 5, "sr": 5, "lead": 6, "manager": 6, "staff": 7,
    "principal": 8, "director": 10, "head": 10, "vp": 12, "chief": 12,
}
JUNIOR_TITLE_WORDS = {"intern", "internship", "junior", "jr", "entry"}

ELIGIBLE_LOCATIONS = (location_filter.COMPATIBLE, location_filter.REVIEW, location_filter.BOSTON_HYBRID)
ELIGIBLE_RELEVANCE = (relevance_filter.HIGH, relevance_filter.MEDIUM)

APPLY, REVIEW, SKIP = "Apply", "Review", "Skip"
RECOMMENDATIONS = (APPLY, REVIEW, SKIP)

# --------------------------------------------------------------------------- #
# Qualification catalog
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Qualification:
    id: str
    label: str
    category: str                      # technical | aerospace | professional
    aliases: tuple[str, ...]
    specialized: bool = False          # never credited unless the profile supports it
    implied_by: tuple[str, ...] = ()   # candidate has one of these -> implied credit
    related: tuple[str, ...] = ()      # candidate has one of these -> transferable credit


def _q(id, label, category, aliases, **kw) -> Qualification:
    return Qualification(id, label, category, tuple(aliases), **kw)


CATALOG: tuple[Qualification, ...] = (
    # --- technical: languages, tools, analytics -------------------------------
    _q("python", "Python", "technical", ["python"]),
    _q("sql", "SQL", "technical", ["sql"], implied_by=("postgresql",)),
    _q("postgresql", "PostgreSQL", "technical", ["postgresql", "postgres"], related=("sql",)),
    _q("matlab", "MATLAB", "technical", ["matlab"]),
    _q("simulink", "Simulink", "technical", ["simulink"], related=("matlab",)),
    _q("cpp", "C++", "technical", ["cplusplus", "cpp"], related=("python",)),
    _q("c", "C", "technical", ["c", "c programming", "c language"], related=("python",)),
    _q("java", "Java", "technical", ["java"], related=("python",)),
    _q("javascript", "JavaScript", "technical", ["javascript", "nodejs", "node js"],
       implied_by=("typescript", "react")),
    _q("typescript", "TypeScript", "technical", ["typescript"]),
    _q("react", "React", "technical", ["react", "reactjs"]),
    _q("linux", "Linux", "technical", ["linux", "unix", "bash", "shell scripting"]),
    _q("git", "Git / version control", "technical", ["git", "github", "gitlab", "version control"]),
    _q("numpy", "NumPy", "technical", ["numpy"]),
    _q("pandas", "pandas", "technical", ["pandas"]),
    _q("scipy", "SciPy", "technical", ["scipy"]),
    _q("sklearn", "scikit-learn", "technical", ["scikit learn", "sklearn"]),
    _q("machine_learning", "Machine learning", "technical", ["machine learning", "ml", "ai ml"]),
    _q("deep_learning", "Deep learning", "technical",
       ["deep learning", "neural network", "neural networks", "pytorch", "tensorflow"],
       related=("machine_learning",)),
    _q("statistics", "Statistics / statistical modeling", "technical",
       ["statistics", "statistical", "statistical modeling", "statistical analysis", "probability"]),
    _q("data_analysis", "Data analysis", "technical",
       ["data analysis", "data analytics", "analytics", "data science", "data analyst"]),
    _q("signal_processing", "Signal processing", "technical",
       ["signal processing", "statistical signal processing", "digital signal processing", "dsp"]),
    _q("spectral_analysis", "Spectral analysis", "technical", ["spectral analysis", "fourier", "fft"]),
    _q("time_series", "Time-series analysis", "technical", ["time series"]),
    _q("regression", "Regression", "technical", ["regression", "nonlinear regression", "curve fitting"]),
    _q("clustering", "Clustering", "technical", ["clustering", "dbscan", "k means"]),
    _q("feature_extraction", "Feature extraction", "technical", ["feature extraction", "feature engineering"]),
    _q("estimation", "Parameter estimation", "technical", ["parameter estimation", "estimation theory"]),
    _q("kalman", "Kalman filtering / state estimation", "technical",
       ["kalman", "kalman filter", "kalman filtering", "state estimation"], related=("estimation",)),
    _q("optimization", "Optimization", "technical", ["optimization", "optimisation", "numerical optimization"]),
    _q("modeling_simulation", "Modeling and simulation", "technical",
       ["modeling and simulation", "modelling and simulation", "simulation", "simulations", "m and s"]),
    _q("automation", "Automation / scripting", "technical", ["automation", "test automation", "scripting"]),
    _q("cloud", "Cloud platforms", "technical", ["aws", "azure", "gcp", "cloud"]),
    _q("containers", "Docker / Kubernetes", "technical", ["docker", "kubernetes", "containers", "containerization"]),
    _q("labview", "LabVIEW", "technical", ["labview"], related=("automation",)),
    _q("fpga", "FPGA / HDL", "technical",
       ["fpga", "fpgas", "vhdl", "verilog", "systemverilog", "hdl", "xilinx", "vivado"], specialized=True),
    _q("embedded", "Embedded / firmware", "technical",
       ["embedded", "firmware", "rtos", "microcontroller", "microcontrollers", "bare metal"], specialized=True),
    _q("rf_hardware", "RF hardware / RF engineering", "technical",
       ["rf", "rf design", "rf hardware", "rf engineering", "microwave", "antenna design", "antennas",
        "link budget", "link budgets"], specialized=True),
    _q("pcb", "PCB / circuit design", "technical",
       ["pcb", "altium", "schematic capture", "circuit design", "board design"], specialized=True),
    _q("cad", "Mechanical CAD", "technical",
       ["cad", "solidworks", "catia", "creo", "siemens nx", "autocad"], specialized=True),
    _q("fea", "FEA / structural analysis", "technical",
       ["fea", "finite element", "finite element analysis", "ansys", "nastran", "abaqus"], specialized=True),
    # --- aerospace / sensor ----------------------------------------------------
    _q("satellite_systems", "Satellite / spacecraft systems", "aerospace",
       ["satellite", "satellites", "spacecraft", "space systems", "space vehicle", "space vehicles",
        "smallsat", "smallsats", "cubesat", "cubesats"]),
    _q("satellite_ops", "Satellite operations", "aerospace",
       ["satellite operations", "spacecraft operations", "mission operations", "flight operations",
        "satellite ops", "on orbit operations", "operations center"], related=("satellite_systems",)),
    _q("telemetry", "Telemetry analysis", "aerospace", ["telemetry", "tlm"]),
    _q("anomaly", "Anomaly detection / resolution", "aerospace",
       ["anomaly", "anomalies", "anomaly detection", "anomaly resolution", "anomaly investigation"]),
    _q("root_cause", "Root cause / failure investigation", "aerospace",
       ["root cause", "root cause analysis", "failure analysis", "failure investigation", "troubleshooting"],
       related=("anomaly",)),
    _q("infrared", "Infrared / EO-IR sensors", "aerospace",
       ["infrared", "ir sensor", "ir sensors", "eo ir", "eoir", "swir", "mwir", "lwir", "focal plane",
        "focal plane array", "focal plane arrays", "fpa", "thermal imaging"]),
    _q("eo_sensors", "Electro-optical / imaging sensors", "aerospace",
       ["electro optical", "electro optics", "optical sensor", "optical sensors", "optical payload",
        "optical payloads", "imaging sensor", "imaging sensors", "imaging systems"], implied_by=("infrared",)),
    _q("sensor_char", "Sensor characterization / performance", "aerospace",
       ["sensor characterization", "sensor performance", "sensor modeling", "sensor model", "sensor models",
        "detector characterization"]),
    _q("calibration", "Calibration", "aerospace", ["calibration", "calibrations", "radiometric calibration"]),
    _q("integration_test", "Systems integration and test", "aerospace",
       ["integration and test", "i and t", "systems integration", "system integration", "integration testing",
        "integration and testing", "ait", "aiv", "test and verification", "test campaign", "test campaigns"]),
    _q("environmental_test", "Environmental testing", "aerospace",
       ["environmental testing", "environmental test", "environmental tests", "thermal vacuum", "tvac",
        "vibration testing", "vibration test", "qualification testing", "environmental qualification"]),
    _q("verification", "Requirements verification / V&V", "aerospace",
       ["requirements verification", "verification and validation", "v and v", "verification"]),
    _q("requirements", "Requirements development", "aerospace",
       ["requirements management", "requirements development", "requirements definition",
        "requirements analysis", "doors"], related=("verification",)),
    _q("systems_eng", "Systems engineering", "aerospace",
       ["systems engineering", "system engineering"]),
    _q("mbse", "MBSE / SysML", "aerospace",
       ["mbse", "model based systems engineering", "sysml", "cameo"], related=("systems_eng",)),
    _q("hw_sw_integration", "Hardware-software integration / HIL", "aerospace",
       ["hardware software integration", "hardware in the loop", "hil", "hwil", "software in the loop"]),
    _q("attitude", "Attitude determination", "aerospace",
       ["attitude determination", "attitude control", "adcs", "quaternion", "quaternions", "star tracker",
        "star trackers"]),
    _q("gnc", "GNC", "aerospace", ["gnc", "guidance navigation and control", "guidance navigation"],
       related=("attitude",)),
    _q("orbital", "Orbital mechanics", "aerospace",
       ["orbital mechanics", "astrodynamics", "orbit determination", "orbital dynamics", "stk",
        "systems tool kit"], related=("satellite_ops",)),
    _q("remote_sensing", "Remote sensing / Earth observation", "aerospace",
       ["remote sensing", "earth observation", "hyperspectral", "multispectral", "imagery", "geospatial"],
       related=("infrared", "spectral_analysis")),
    _q("ground_systems", "Ground systems", "aerospace",
       ["ground systems", "ground system", "ground station", "ground stations", "ground software",
        "mission control", "command and control"], related=("satellite_ops", "telemetry")),
    _q("image_processing", "Image processing", "aerospace",
       ["image processing", "computer vision", "image analysis"], related=("signal_processing",)),
    _q("radar", "Radar / SAR", "aerospace", ["radar", "sar", "synthetic aperture radar"],
       specialized=True, related=("signal_processing",)),
    _q("propulsion", "Propulsion", "aerospace", ["propulsion", "rocket engine", "rocket engines", "thrusters"],
       specialized=True),
    _q("avionics", "Avionics", "aerospace", ["avionics", "flight computer", "flight computers"], specialized=True),
    _q("optical_design", "Optical design", "aerospace",
       ["optical design", "lens design", "zemax", "code v", "optical engineering"], specialized=True),
    _q("thermal_analysis", "Thermal analysis / design", "aerospace",
       ["thermal analysis", "thermal modeling", "thermal design", "thermal desktop"],
       specialized=True, related=("environmental_test",)),
    # --- professional ------------------------------------------------------------
    _q("communication", "Technical communication / reporting", "professional",
       ["technical writing", "technical reports", "technical reporting", "performance reports",
        "communication skills", "written and verbal communication", "stakeholders", "stakeholder",
        "presentations", "documentation"]),
)
CATALOG_BY_ID = {q.id: q for q in CATALOG}

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("c++", " cplusplus ").replace("c#", " csharp ").replace("&", " and ")
    text = re.sub(r"['’`�.]", "", text)  # you'll -> youll, B.S. -> bs, e.g. -> eg
    return " ".join(re.sub(r"[^a-z0-9+]+", " ", text).replace("+", " ").split())


_ALIASES = sorted(
    ((tuple(normalize(a).split()), q.id) for q in CATALOG for a in q.aliases),
    key=lambda pair: (-len(pair[0]), pair[0]),
)


def find_qualifications(text: str) -> list[str]:
    """Catalog ids mentioned in ``text`` (whole words, longest alias first, no overlaps)."""
    tokens = normalize(text).split()
    used = [False] * len(tokens)
    found: list[str] = []
    for alias, qid in _ALIASES:
        n = len(alias)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == alias and not any(used[i:i + n]):
                used[i:i + n] = [True] * n
                if qid not in found:
                    found.append(qid)
    return found


# --------------------------------------------------------------------------- #
# Candidate model
# --------------------------------------------------------------------------- #

_DEGREE_LEVELS = {"bachelor": 1, "master": 2, "phd": 3}
_FIELD_PHRASES = {
    "mechanical engineering": "mechanical", "mechanical": "mechanical",
    "aerospace": "aerospace", "aeronautical": "aerospace", "astronautical": "aerospace",
    "electrical": "electrical", "computer engineering": "computer", "computer science": "computer_science",
    "physics": "physics", "applied physics": "physics", "mathematics": "math", "applied mathematics": "math",
    "math": "math", "systems engineering": "systems", "optical engineering": "optics", "optics": "optics",
    "data science": "data", "data analytics": "data", "analytics": "data", "statistics": "data",
    "engineering": "engineering", "stem": "stem",
}
_ENGINEERING_FIELDS = {"mechanical", "aerospace", "electrical", "computer", "systems", "optics", "engineering"}


def _degree_fields(text: str) -> set[str]:
    padded = f" {normalize(text)} "
    return {canon for phrase, canon in _FIELD_PHRASES.items() if f" {phrase} " in padded}


def _degree_level(text: str) -> int | None:
    norm = normalize(text)
    tokens = set(norm.split())
    if tokens & {"phd", "doctorate", "doctoral"}:
        return 3
    if tokens & {"master", "masters", "msc", "mse"} or re.search(r"\bms\b(?! (office|excel|word|project|teams))", norm):
        return 2
    if tokens & {"bachelor", "bachelors", "bs", "ba", "bsc", "bse", "undergraduate"}:
        return 1
    return None


@dataclass
class Candidate:
    capabilities: dict[str, set[str]]         # catalog id -> {"experience", "skills"}
    years: float
    degree_level: int                          # highest completed level (0 = none documented)
    in_progress_level: int
    degree_fields: set[str]
    transferable: list[str] = field(default_factory=list)

    def credit(self, qid: str) -> tuple[float, str]:
        """(credit, basis) for one catalog id. basis explains where credit came from."""
        q = CATALOG_BY_ID[qid]
        sources = self.capabilities.get(qid)
        if sources:
            if "experience" in sources:
                return EVIDENCE_CREDIT, "experience/evidence"
            return SKILL_LIST_CREDIT, "skills list"
        for other in q.implied_by:
            if other in self.capabilities:
                return IMPLIED_CREDIT, f"implied by {CATALOG_BY_ID[other].label}"
        for other in q.related:
            if other in self.capabilities:
                return TRANSFERABLE_CREDIT, f"transferable from {CATALOG_BY_ID[other].label}"
        return 0.0, "not established by profile"


def build_candidate(profile: Mapping[str, Any]) -> Candidate:
    """Derive capabilities from the profile. Target roles are interests and give no credit."""
    capabilities: dict[str, set[str]] = {}

    def add(texts: Iterable[str], source: str) -> None:
        for text in texts:
            for qid in find_qualifications(text):
                capabilities.setdefault(qid, set()).add(source)

    add(profile.get("technical_skills", []), "skills")
    add(profile.get("transferable_skills", []), "skills")
    add(profile.get("aerospace_experience", []), "experience")
    for entry in profile.get("experience_evidence", []) or []:
        add([entry.get("skill", ""), entry.get("evidence", "")], "experience")

    completed, in_progress, fields = 0, 0, set()
    for entry in profile.get("education", []):
        level = _degree_level(entry.get("degree", "")) or 0
        if entry.get("status") == "completed":
            completed = max(completed, level)
            fields |= _degree_fields(entry.get("field", ""))
        else:
            in_progress = max(in_progress, level)
    if fields & _ENGINEERING_FIELDS:
        fields.add("engineering")

    years = max((v for v in profile.get("years_experience", {}).values()), default=0)
    return Candidate(capabilities, float(years), completed, in_progress, fields,
                     list(profile.get("transferable_skills", [])))


# --------------------------------------------------------------------------- #
# Requirement extraction
# --------------------------------------------------------------------------- #

REQUIRED, PREFERRED, RESPONSIBILITIES, ELIGIBILITY, OTHER, INTRO = (
    "required", "preferred", "responsibilities", "eligibility", "other", "intro")

_HEADER_RULES = (  # (section, phrases) checked in order on the normalized header
    (ELIGIBILITY, ("itar", "ear requirements", "export control", "security clearance", "clearance requirements")),
    (PREFERRED, ("preferred", "nice to have", "bonus", "desired", "good to have", "what makes you stand out",
                 "stand out", "pluses", "extra credit", "additional qualifications")),
    (REQUIRED, ("qualifications", "requirements", "what you bring", "what youll bring", "what youll need",
                "what you need", "youll need", "what were looking for", "what we are looking for",
                "who you are", "about you", "you have", "must have", "skills and experience", "your background",
                "minimum", "basic")),
    (RESPONSIBILITIES, ("responsibilities", "responsibility", "what youll do", "what you will do",
                        "impact youll own", "about the role", "about this role", "the role", "your role",
                        "duties", "day to day", "in this role", "what youll be doing", "the opportunity",
                        "job summary", "position summary")),
    (OTHER, ("about", "benefits", "compensation", "salary", "pay range", "perks", "equal opportunity", "eeo",
             "why join", "our mission", "taking care", "application deadline", "accommodations",
             "fair chance", "total rewards", "culture", "who we are", "li remote", "location")),
)
_INLINE_PREFERRED = re.compile(r"\b(preferred|a plus|nice to have|bonus|desired|ideally|is a plus|helpful)\b")
_REQUIREMENT_CUE = re.compile(
    r"\b(experience (with|in)|proficien\w*|knowledge of|familiar\w*|degree|years|required|must|skilled in|"
    r"expertise|background in|ability to)\b")
_OR_CUE = re.compile(r"\b(or|such as|eg|including|for example|one or more|any of)\b")
_ELIGIBILITY = re.compile(
    r"\b(security clearance|clearance|ts sci|top secret|secret|itar|us person|u s person|us citizen|"
    r"u s citizen|citizenship|export control\w*)\b")
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split())}
_YEARS = re.compile(
    r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?"
    r"(?:\(\d{1,2}\)\s*)?(?:years|yrs)\b", re.IGNORECASE)


@dataclass
class Statement:
    text: str
    section: str          # required | preferred | responsibilities | eligibility
    inferred: bool = False


def _header_section(text: str) -> str | None:
    norm = normalize(text)
    if not norm or len(norm.split()) > 8 or find_qualifications(text):
        return None
    padded = f" {norm} "
    for section, phrases in _HEADER_RULES:
        if any(f" {p} " in padded for p in phrases):
            return section
    return None


def extract_statements(description: str) -> tuple[list[Statement], str]:
    """Split a description into requirement/responsibility statements.

    Returns (statements, mode) where mode is "sections", "inferred" or "none".
    """
    section = INTRO
    collected: list[Statement] = []
    loose: list[str] = []
    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = line.startswith(("-", "•", "*", "·"))
        body = line.lstrip("-•*· ").strip()
        if not bullet:
            head, sep, rest = body.partition(":")
            header = _header_section(head) if (sep or not body.endswith(".")) else None
            if header:
                section = header
                if not rest.strip():
                    continue
                body = rest.strip()
        for sentence in re.split(r"(?<=[.;!?])\s+", body):
            if not normalize(sentence):
                continue
            if section in (REQUIRED, PREFERRED):
                level = PREFERRED if section == PREFERRED or _INLINE_PREFERRED.search(normalize(sentence)) else REQUIRED
                collected.append(Statement(sentence, level))
            elif section in (RESPONSIBILITIES, ELIGIBILITY):
                collected.append(Statement(sentence, section))
            elif section == INTRO:
                loose.append(sentence)
            if section != ELIGIBILITY and _ELIGIBILITY.search(normalize(sentence)) and section not in (REQUIRED, PREFERRED):
                collected.append(Statement(sentence, ELIGIBILITY))

    if any(s.section in (REQUIRED, PREFERRED) for s in collected):
        return collected, "sections"
    inferred = [Statement(s, PREFERRED if _INLINE_PREFERRED.search(normalize(s)) else REQUIRED, inferred=True)
                for s in loose if _REQUIREMENT_CUE.search(normalize(s))]
    statements = collected + inferred
    return statements, ("inferred" if inferred else "none")


def _years_in(text: str) -> int | None:
    values = []
    for match in _YEARS.finditer(text):
        token = match.group(1).lower()
        value = _NUMBER_WORDS.get(token, None) if not token.isdigit() else int(token)
        if value is not None and 0 < value <= 30:
            values.append(value)
    return max(values) if values else None


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class _Unit:
    """One requirement: a single qualification or an any-of group."""
    ids: tuple[str, ...]
    level: str
    category: str

    @property
    def label(self) -> str:
        return " or ".join(CATALOG_BY_ID[i].label for i in self.ids)


def _clamp(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def _score(units: list[tuple[_Unit, float, float]]) -> int | None:
    """units: (unit, weight, credit) -> 0-100, or None when there is nothing to score."""
    total = sum(w for _, w, _ in units)
    if not total:
        return None
    return _clamp(100 * sum(w * c for _, w, c in units) / total)


def _title_years(title: str) -> tuple[int | None, bool]:
    tokens = normalize(title).split()
    implied = [TITLE_IMPLIED_YEARS[t] for t in tokens if t in TITLE_IMPLIED_YEARS]
    phrase = " ".join(tokens)
    if "vice president" in phrase:
        implied.append(TITLE_IMPLIED_YEARS["vp"])
    junior = bool(set(tokens) & JUNIOR_TITLE_WORDS)
    return (max(implied) if implied else None), junior


def evaluate_job(job: Mapping[str, Any], candidate: Candidate,
                 location_result: Mapping[str, str] | None = None,
                 relevance_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one job (mapping with title, location, description, ...) against the candidate."""
    title = job.get("title") if isinstance(job.get("title"), str) else ""
    location_result = location_result or location_filter.classify_location(job.get("location"))
    relevance_result = relevance_result or relevance_filter.classify_relevance(title)
    description = job.get("description")

    notes: list[str] = []
    if not isinstance(description, str):
        desc_status = "malformed" if description is not None else "missing"
        statements, mode = [], "none"
    elif not normalize(description):
        desc_status, statements, mode = "missing", [], "none"
    else:
        desc_status = "ok"
        statements, mode = extract_statements(description)

    # --- requirement units --------------------------------------------------
    units: dict[tuple[str, ...], _Unit] = {}
    responsibility_ids: list[str] = []
    eligibility: list[str] = []
    years_required = years_preferred = None
    degree_reqs: list[tuple[str, int | None, set[str], bool]] = []

    for st in statements:
        norm = normalize(st.text)
        if st.section == ELIGIBILITY:
            if _ELIGIBILITY.search(norm) and not eligibility:
                eligibility.append(st.text.strip())
            continue
        ids = find_qualifications(st.text)
        if st.section == RESPONSIBILITIES:
            responsibility_ids += [i for i in ids if i not in responsibility_ids]
            continue
        years = _years_in(st.text)
        if years is not None:
            if st.section == REQUIRED:
                years_required = max(years_required or 0, years)
            else:
                years_preferred = max(years_preferred or 0, years)
        level_needed = _degree_level(st.text) if re.search(r"\b(degree|bachelor|master|phd|bs|ms|ba)\b", norm) else None
        if level_needed or re.search(r"\bdegree\b", norm):
            degree_reqs.append((st.section, level_needed or 1, _degree_fields(st.text),
                                bool(re.search(r"\b(related|equivalent|similar|technical field)\b", norm))))
        if not ids:
            continue
        groups: list[tuple[str, ...]]
        if _OR_CUE.search(norm) and len(ids) > 1:
            by_cat: dict[str, list[str]] = {}
            for i in ids:
                by_cat.setdefault(CATALOG_BY_ID[i].category, []).append(i)
            groups = [tuple(v) for v in by_cat.values()]
        else:
            groups = [(i,) for i in ids]
        for g in groups:
            existing = units.get(g)
            if existing is None or (existing.level == PREFERRED and st.section == REQUIRED):
                units[g] = _Unit(g, st.section, CATALOG_BY_ID[g[0]].category)

    # --- credit each unit -----------------------------------------------------
    matched: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    transferable: list[str] = []
    tech_scored: list[tuple[_Unit, float, float]] = []
    aero_scored: list[tuple[_Unit, float, float]] = []
    resp_scored: list[tuple[_Unit, float, float]] = []
    required_gaps = 0

    def credit_unit(unit: _Unit) -> tuple[float, str, str]:
        best = max(((candidate.credit(i), i) for i in unit.ids), key=lambda x: x[0][0])
        (credit, basis), qid = best
        return credit, basis, qid

    for key in sorted(units, key=lambda k: (units[k].level != REQUIRED, units[k].label)):
        unit = units[key]
        weight = REQUIRED_WEIGHT if unit.level == REQUIRED else PREFERRED_WEIGHT
        credit, basis, qid = credit_unit(unit)
        if credit >= IMPLIED_CREDIT:
            matched.append({"qualification": CATALOG_BY_ID[qid].label, "level": unit.level, "basis": basis})
        elif credit > 0:
            transferable.append(f"{unit.label} ({unit.level}): {basis}")
        else:
            specialized = any(CATALOG_BY_ID[i].specialized for i in unit.ids)
            missing.append({"qualification": unit.label, "level": unit.level, "gap": "not_established",
                            "detail": "specialized skill not supported by profile" if specialized
                            else "not mentioned in profile"})
            if unit.level == REQUIRED:
                required_gaps += 1
        target = {"technical": tech_scored, "aerospace": aero_scored}.get(unit.category, resp_scored)
        target.append((unit, weight, credit))

    for qid in responsibility_ids:
        unit = _Unit((qid,), RESPONSIBILITIES, CATALOG_BY_ID[qid].category)
        credit, _, _ = credit_unit(unit)
        resp_scored.append((unit, 1.0, credit))
        if unit.category == "aerospace" and (qid,) not in units:
            aero_scored.append((unit, RESPONSIBILITY_AEROSPACE_WEIGHT, credit))

    # --- degree requirements (part of technical) --------------------------------
    for section, level_needed, fields, related in degree_reqs:
        weight = REQUIRED_WEIGHT if section == REQUIRED else PREFERRED_WEIGHT
        unit = _Unit(("degree",), section, "technical")
        names = {1: "Bachelor's", 2: "Master's", 3: "PhD"}
        label = f"{names.get(level_needed, 'Degree')} degree" + (f" in {', '.join(sorted(fields))}" if fields else "")
        field_ok = (not fields or bool(candidate.degree_fields & fields)
                    or bool(fields & {"engineering", "stem"} and candidate.degree_fields))
        if candidate.degree_level >= level_needed and field_ok:
            matched.append({"qualification": label, "level": section, "basis": "education"})
            credit = EVIDENCE_CREDIT
        elif candidate.degree_level >= level_needed and related and candidate.degree_fields:
            transferable.append(f"{label} ({section}): related field accepted; profile degree field differs")
            credit = TRANSFERABLE_CREDIT * 1.5
        elif candidate.degree_level < level_needed and candidate.in_progress_level >= level_needed:
            missing.append({"qualification": label, "level": section, "gap": "not_established",
                            "detail": "degree at this level is in progress"})
            credit = TRANSFERABLE_CREDIT
        else:
            detail = ("degree field differs from profile" if candidate.degree_level >= level_needed
                      else "profile documents a lower completed degree level")
            missing.append({"qualification": label, "level": section, "gap": "confirmed_gap", "detail": detail})
            credit = 0.0
            if section == REQUIRED:
                required_gaps += 1
        tech_scored.append((unit, weight, credit))

    # --- seniority ------------------------------------------------------------
    implied_years, junior_title = _title_years(title)
    needed_years = years_required if years_required is not None else implied_years
    years_source = "description" if years_required is not None else ("title" if implied_years is not None else None)
    seniority_gap = max(0.0, (needed_years or 0) - candidate.years)
    seniority_penalty = min(SENIORITY_MAX_PENALTY, seniority_gap * SENIORITY_PENALTY_PER_YEAR)
    if seniority_gap > 0:
        missing.append({
            "qualification": f"{needed_years}+ years experience" + (" (implied by title)" if years_source == "title" else ""),
            "level": REQUIRED, "gap": "confirmed_gap",
            "detail": f"profile documents {candidate.years:g} years"})
    elif junior_title and candidate.years >= 3:
        seniority_penalty = OVERQUALIFIED_PENALTY
        notes.append("title suggests an entry-level role")
    if years_preferred and years_preferred > candidate.years and (years_required or 0) < years_preferred:
        missing.append({"qualification": f"{years_preferred}+ years experience", "level": PREFERRED,
                        "gap": "confirmed_gap", "detail": f"profile documents {candidate.years:g} years"})

    for item in eligibility:
        missing.append({"qualification": "Clearance / citizenship / export-control eligibility",
                        "level": REQUIRED, "gap": "not_established",
                        "detail": "eligibility is not recorded in the profile; verify in the posting"})
        break

    # --- component scores -----------------------------------------------------
    technical = _score(tech_scored)
    aerospace = _score(aero_scored)
    resp_base = _score(resp_scored)
    technical_score = NEUTRAL_SCORE if technical is None else technical
    aerospace_score = NEUTRAL_SCORE if aerospace is None else aerospace
    responsibility_score = _clamp((NEUTRAL_SCORE if resp_base is None else resp_base) - seniority_penalty)
    loc_class = location_result["classification"]
    location_score = LOCATION_SCORES.get(loc_class, 0)

    overall = _clamp(WEIGHTS["technical"] * technical_score + WEIGHTS["aerospace"] * aerospace_score
                     + WEIGHTS["responsibility"] * responsibility_score + WEIGHTS["location"] * location_score)

    # --- recommendation ---------------------------------------------------------
    blockers = []
    if loc_class != location_filter.COMPATIBLE:
        blockers.append(f"location is {loc_class} ({location_result['reason']})")
    if desc_status != "ok":
        blockers.append(f"description {desc_status}")
    elif mode == "none":
        blockers.append("no requirements could be extracted")
    if seniority_gap >= SENIORITY_APPLY_BLOCK_YEARS:
        blockers.append(f"seniority gap of {seniority_gap:g} years")
    if required_gaps > MAX_REQUIRED_GAPS_FOR_APPLY:
        blockers.append(f"{required_gaps} required qualifications not met")

    if overall < SKIP_THRESHOLD:
        recommendation = SKIP
    elif overall >= APPLY_THRESHOLD and not blockers:
        recommendation = APPLY
    else:
        recommendation = REVIEW

    # --- explanation --------------------------------------------------------------
    req_units = [u for u in units.values() if u.level == REQUIRED]
    req_matched = sum(1 for m in matched if m["level"] == REQUIRED)
    parts = []
    if desc_status != "ok":
        parts.append(f"Description {desc_status}; requirements not assessed.")
    elif mode == "none":
        parts.append("No requirement statements found in description.")
    else:
        if mode == "inferred":
            parts.append("Requirements inferred from unstructured text.")
        matched_names = ", ".join(m["qualification"] for m in matched[:6]) or "none"
        parts.append(f"Matched {req_matched}/{len(req_units) + len([d for d in degree_reqs if d[0] == REQUIRED])} "
                     f"required items ({matched_names}).")
    gaps = [m for m in missing if m["level"] == REQUIRED]
    if gaps:
        parts.append("Gaps: " + "; ".join(
            f"{g['qualification']} ({'confirmed' if g['gap'] == 'confirmed_gap' else 'not established'})"
            for g in gaps[:5]) + ".")
    if transferable:
        parts.append(f"Transferable: {len(transferable)} related item(s).")
    parts.append(f"Location {loc_class}.")
    parts += [n[0].upper() + n[1:] + "." for n in notes]
    if blockers and recommendation != SKIP and overall >= APPLY_THRESHOLD:
        parts.append("Not Apply: " + "; ".join(blockers) + ".")
    parts.append("Heuristic estimate, not a hiring probability.")

    return {
        "company": job.get("company"),
        "title": title,
        "location": job.get("location"),
        "ats_provider": job.get("ats_provider"),
        "ats_job_id": job.get("ats_job_id"),
        "application_url": job.get("application_url"),
        "location_classification": loc_class,
        "title_relevance": relevance_result.get("classification"),
        "overall_fit_score": overall,
        "technical_match_score": technical_score,
        "aerospace_match_score": aerospace_score,
        "responsibility_match_score": responsibility_score,
        "location_match_score": location_score,
        "matched_qualifications": matched,
        "missing_qualifications": missing,
        "transferable_skills": transferable,
        "explanation": " ".join(parts),
        "recommendation": recommendation,
        "requirements_extraction": mode,
        "description_status": desc_status,
        "seniority": {"required_years": needed_years, "source": years_source,
                      "profile_years": candidate.years, "penalty": seniority_penalty},
        "blockers": blockers,
    }


# --------------------------------------------------------------------------- #
# Selection, ranking and CLI
# --------------------------------------------------------------------------- #


def load_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT j.id, c.name AS company, c.ats_provider, j.ats_job_id, j.title, j.location,
               j.description, j.application_url
        FROM jobs j JOIN companies c ON c.id = j.company_id
        WHERE j.is_active = 1
        ORDER BY c.name, j.title, j.ats_job_id
        """
    ).fetchall()


def evaluate_jobs(jobs: Iterable[Mapping[str, Any]], candidate: Candidate) -> tuple[list[dict], Counter]:
    """Filter by location and title relevance, evaluate, and rank. Returns (results, funnel counts)."""
    funnel: Counter = Counter()
    results = []
    for job in jobs:
        funnel["active"] += 1
        loc = location_filter.classify_location(job["location"])
        if loc["classification"] not in ELIGIBLE_LOCATIONS:
            continue
        funnel["location_eligible"] += 1
        rel = relevance_filter.classify_relevance(job["title"])
        if rel["classification"] not in ELIGIBLE_RELEVANCE:
            continue
        funnel["evaluated"] += 1
        results.append(evaluate_job(dict(job), candidate, loc, rel))
    results.sort(key=lambda r: (-r["overall_fit_score"], RECOMMENDATIONS.index(r["recommendation"]),
                                str(r["company"]), r["title"], str(r["ats_provider"]), str(r["ats_job_id"])))
    return results, funnel


def _format_result(rank: int, r: dict) -> list[str]:
    def fmt_match(m):
        return f"{m['qualification']} ({m['level']}, {m['basis']})"

    def fmt_missing(m):
        gap = "confirmed gap" if m["gap"] == "confirmed_gap" else "not established"
        return f"{m['qualification']} ({m['level']}, {gap})"

    lines = [
        f"#{rank} [{r['overall_fit_score']:>3}] {r['recommendation']:<6} {r['company']} | {r['title']} | "
        f"{r['location'] or '(no location)'} ({r['location_classification']}) | {r['ats_provider']}:{r['ats_job_id']}",
        f"     scores: technical {r['technical_match_score']} | aerospace {r['aerospace_match_score']} | "
        f"responsibilities {r['responsibility_match_score']} | location {r['location_match_score']} | "
        f"title relevance {r['title_relevance']}",
        f"     matched: {'; '.join(fmt_match(m) for m in r['matched_qualifications']) or 'none'}",
        f"     missing: {'; '.join(fmt_missing(m) for m in r['missing_qualifications']) or 'none'}",
    ]
    if r["transferable_skills"]:
        lines.append(f"     transferable: {'; '.join(r['transferable_skills'])}")
    lines.append(f"     why: {r['explanation']}")
    if r.get("application_url"):
        lines.append(f"     url: {r['application_url']}")
    return lines


def build_report(results: list[dict], funnel: Counter, top: int = 15) -> str:
    lines = [f"Top {min(top, len(results))} of {len(results)} evaluated jobs "
             "(heuristic fit estimates, not hiring probabilities):", ""]
    for rank, r in enumerate(results[:top], start=1):
        lines += _format_result(rank, r) + [""]
    recs = Counter(r["recommendation"] for r in results)
    lines.append(
        f"Summary: {funnel['active']} active jobs -> {funnel['location_eligible']} location-eligible "
        f"({', '.join(ELIGIBLE_LOCATIONS)}) -> {funnel['evaluated']} evaluated (title relevance high/medium). "
        f"Recommendations: {', '.join(f'{k} {recs.get(k, 0)}' for k in RECOMMENDATIONS)}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank location- and title-eligible jobs against the local profile "
                                                 "(read-only, offline).")
    parser.add_argument("--db", type=Path, required=True, help="SQLite database, e.g. data/jobs.db")
    parser.add_argument("--profile", type=Path, default=candidate_profile.DEFAULT_PROFILE,
                        help="Candidate profile JSON (default: config/candidate_profile.json)")
    parser.add_argument("--top", type=int, default=15, help="Number of ranked jobs to print (default 15)")
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")

    try:
        profile = candidate_profile.load_profile(args.profile)
    except candidate_profile.ProfileFileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except candidate_profile.ProfileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

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

    results, funnel = evaluate_jobs(jobs, build_candidate(profile))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(build_report(results, funnel, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
