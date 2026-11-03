"""Parse raw files (Markdown with front matter, incident JSON) into domain models."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from incident_agent.domain.models import HistoricalIncident, SourceDocument

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_INCIDENTS = TypeAdapter(list[HistoricalIncident])


def parse_markdown(doc_id: str, text: str, uri: str | None = None) -> SourceDocument:
    """``key: value`` front matter -> metadata; first H1 -> title."""
    meta: dict[str, Any] = {}
    match = _FRONT_MATTER.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, sep, value = line.partition(":")
            if sep:
                meta[key.strip()] = value.strip()
        text = text[match.end() :]
    title = _TITLE.search(text)
    return SourceDocument(
        doc_id=doc_id,
        title=meta.get("title") or (title.group(1) if title else doc_id),
        body=text,
        doc_type=meta.get("doc_type", "runbook"),
        service=meta.get("service"),
        uri=uri,
    )


def iter_markdown_dir(directory: Path) -> Iterator[SourceDocument]:
    """Lazily parse every ``*.md`` file (sorted, so runs are reproducible)."""
    for path in sorted(directory.glob("*.md")):
        yield parse_markdown(path.stem, path.read_text(encoding="utf-8"), uri=path.as_posix())


def load_incidents(path: Path) -> list[HistoricalIncident]:
    """Validate a JSON array of incidents in one pass with a ``TypeAdapter``."""
    return _INCIDENTS.validate_python(json.loads(path.read_text(encoding="utf-8")))
