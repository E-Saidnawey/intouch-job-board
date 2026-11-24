"""Dispatch job fetching to the right ATS provider.

Supported: greenhouse, ashby, lever. To add a provider: implement
``fetch_<provider>_jobs(board_token, company=...)`` returning normalized job
dicts in its own module (reusing src/ats_common.py), then register it in
``_PROVIDERS`` together with the exception type it raises. Config validation in
src/company_config.py picks up new providers from ``SUPPORTED_PROVIDERS``.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from src import ashby, greenhouse, lever


class CollectionError(Exception):
    """A single company's jobs could not be collected."""


class _Provider(NamedTuple):
    fetch: Callable[..., list[dict[str, Any]]]
    error: type[Exception]


def _greenhouse_fetch(board_token: str, company: str) -> list[dict[str, Any]]:
    # Looked up at call time so tests can patch src.greenhouse.fetch_greenhouse_jobs.
    return greenhouse.fetch_greenhouse_jobs(board_token, company=company)


def _ashby_fetch(board_token: str, company: str) -> list[dict[str, Any]]:
    # Looked up at call time so tests can patch src.ashby.fetch_ashby_jobs.
    return ashby.fetch_ashby_jobs(board_token, company=company)


def _lever_fetch(board_token: str, company: str) -> list[dict[str, Any]]:
    # Looked up at call time so tests can patch src.lever.fetch_lever_jobs.
    # For Lever the config board_token is the site identifier (jobs.lever.co/{site}).
    return lever.fetch_lever_jobs(board_token, company=company)


_PROVIDERS: dict[str, _Provider] = {
    "greenhouse": _Provider(fetch=_greenhouse_fetch, error=greenhouse.GreenhouseError),
    "ashby": _Provider(fetch=_ashby_fetch, error=ashby.AshbyError),
    "lever": _Provider(fetch=_lever_fetch, error=lever.LeverError),
}

SUPPORTED_PROVIDERS = frozenset(_PROVIDERS)


def fetch_jobs(ats_provider: str, board_token: str, company: str) -> list[dict[str, Any]]:
    """Fetch normalized jobs via the named provider.

    Provider-specific errors are re-raised as ``CollectionError`` so callers
    handle every provider the same way.
    """
    provider = _PROVIDERS.get(ats_provider)
    if provider is None:
        raise CollectionError(
            f"Unsupported ATS provider {ats_provider!r}; supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    try:
        return provider.fetch(board_token, company=company)
    except provider.error as exc:
        raise CollectionError(str(exc)) from exc
