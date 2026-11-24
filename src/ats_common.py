"""Shared helpers for ATS collectors: HTTP JSON fetching and text normalization.

Standard library only. Each provider module (greenhouse.py, ashby.py, lever.py) builds its
own URL, validates its own response shape and maps postings to the common
normalized job schema:

    company, ats_provider, ats_job_id, title, location, description,
    application_url, updated_at, published_at

``published_at`` is the employer's publication date (Greenhouse ``first_published``,
Ashby ``publishedAt``, Lever ``createdAt``) or None; it is never derived from ``updated_at``.
"""

from __future__ import annotations

import html
import json
import re
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = "aerospace-job-intelligence-agent/0.1 (+python-urllib)"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def get_json(url: str, timeout: float, error_cls: type[Exception]) -> Any:
    """GET ``url`` and return decoded JSON, raising ``error_cls`` on any failure."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        exc.close()  # release the underlying response/socket
        raise error_cls(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise error_cls(f"Network error fetching {url}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise error_cls(f"Timed out after {timeout}s fetching {url}") from exc
    except OSError as exc:
        raise error_cls(f"Connection error fetching {url}: {exc}") from exc

    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_cls(f"Invalid JSON from {url}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Field helpers
# --------------------------------------------------------------------------- #

def clean_str(value: Any) -> str | None:
    """Return a stripped non-empty string, or None for anything else."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


# --------------------------------------------------------------------------- #
# HTML -> plain text
# --------------------------------------------------------------------------- #

_BLOCK_TAGS = {
    "p", "div", "br", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "tr", "section", "article", "header", "footer", "blockquote", "hr",
}
_SKIP_TAGS = {"script", "style"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS and tag != "li":  # next <li> or </ul> breaks the line
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(raw: str | None, *, entity_escaped: bool = False) -> str:
    """Convert an HTML fragment to readable plain text.

    Set ``entity_escaped`` when the HTML itself arrives entity-escaped
    (Greenhouse's ``content``, e.g. ``&lt;p&gt;``); it is unescaped once first.
    Ashby's ``descriptionHtml`` is raw HTML and must not be pre-unescaped.
    """
    if not raw:
        return ""
    parser = _TextExtractor()
    parser.feed(html.unescape(raw) if entity_escaped else raw)
    parser.close()
    text = parser.text().replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    # Collapse runs of blank lines to a single blank line.
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()
