"""Load and validate the company source configuration (config/companies.json)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.providers import SUPPORTED_PROVIDERS


class ConfigError(Exception):
    """The configuration file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class CompanyConfig:
    name: str
    ats_provider: str
    board_token: str
    enabled: bool = True

    @property
    def snapshot_filename(self) -> str:
        """e.g. 'Muon Space' -> 'muon_space_jobs.json'."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")
        return f"{slug}_jobs.json"


def _require_str(entry: dict[str, Any], key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: '{key}' must be a non-empty string.")
    return value.strip()


def parse_config(data: Any) -> list[CompanyConfig]:
    """Validate decoded JSON and return company entries (enabled and disabled)."""
    if not isinstance(data, dict) or not isinstance(data.get("companies"), list):
        raise ConfigError("Config must be an object with a 'companies' list.")
    if not data["companies"]:
        raise ConfigError("Config 'companies' list is empty.")

    companies: list[CompanyConfig] = []
    seen_sources: set[tuple[str, str]] = set()
    seen_names: set[str] = set()
    seen_files: set[str] = set()

    for index, entry in enumerate(data["companies"]):
        where = f"companies[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: must be an object.")
        name = _require_str(entry, "name", where)
        where = f"{where} ({name})"
        provider = _require_str(entry, "ats_provider", where).lower()
        token = _require_str(entry, "board_token", where)
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{where}: 'enabled' must be true or false.")
        if enabled and provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                f"{where}: unsupported ats_provider {provider!r} "
                f"(supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}). Disable it or fix the provider."
            )

        company = CompanyConfig(name=name, ats_provider=provider, board_token=token, enabled=enabled)
        source = (provider, token.lower())
        if source in seen_sources:
            raise ConfigError(f"{where}: duplicate source {provider}/{token}.")
        if name.lower() in seen_names:
            raise ConfigError(f"{where}: duplicate company name.")
        if company.snapshot_filename in seen_files:
            raise ConfigError(f"{where}: snapshot filename {company.snapshot_filename} collides with another company.")
        seen_sources.add(source)
        seen_names.add(name.lower())
        seen_files.add(company.snapshot_filename)
        companies.append(company)

    return companies


def load_config(path: str | Path) -> list[CompanyConfig]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config {path}: {exc}") from exc
    return parse_config(data)


def select_companies(
    companies: list[CompanyConfig], names: list[str] | None = None, all_enabled: bool = False
) -> list[CompanyConfig]:
    """Pick enabled companies, or the named ones (matched by name or board token, case-insensitive)."""
    if all_enabled:
        selected = [c for c in companies if c.enabled]
        if not selected:
            raise ConfigError("No enabled companies in config.")
        return selected

    selected = []
    for requested in names or []:
        key = requested.strip().lower()
        match = next((c for c in companies if key in (c.name.lower(), c.board_token.lower())), None)
        if match is None:
            known = ", ".join(c.name for c in companies)
            raise ConfigError(f"Unknown company {requested!r}. Configured: {known}")
        if not match.enabled:
            raise ConfigError(f"Company {match.name!r} is disabled in config.")
        if match not in selected:
            selected.append(match)
    return selected
