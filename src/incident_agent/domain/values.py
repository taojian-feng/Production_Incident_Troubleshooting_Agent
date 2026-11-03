"""Internal value objects: frozen, slotted dataclasses.

These types live on hot paths (thousands of log lines, every chunk in the
index) and never cross a trust boundary, so we use ``@dataclass(frozen=True,
slots=True)`` instead of pydantic:

* ``slots=True`` drops the per-instance ``__dict__`` (roughly 40-60% less memory
  per object), which matters when a log stream yields millions of events.
* ``frozen=True`` makes instances hashable and safe to share between
  concurrent branches of the agent graph without defensive copies.
* Validation happens once, at the boundary (``domain.models``), not on every
  internal construction.

Collections inside frozen objects are tuples / ``Mapping`` so the "frozen"
promise is not just shallow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

_VOLATILE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"  # uuids
    r"|\b0x[0-9a-f]+\b"  # hex addresses
    r"|\b(?=[0-9a-f]*\d)[0-9a-f]{6,}\b"  # request ids / hashes
    r"|\b\d+(?:\.\d+)?(?:ms|s)?\b",  # numbers, durations
    re.IGNORECASE,
)


def _freeze(mapping: Mapping[str, str] | None) -> Mapping[str, str]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True, slots=True)
class LogEvent:
    """One CloudWatch log event. Hashable, so it can live in sets for dedupe."""

    event_id: str
    timestamp: datetime
    log_group: str
    message: str
    log_stream: str = ""

    @property
    def signature(self) -> str:
        """Message with volatile tokens (ids, numbers) masked.

        Two events with the same signature are "the same error" for counting
        purposes, e.g. ``timeout after 3012ms`` and ``timeout after 2999ms``.
        """
        return _VOLATILE.sub("<*>", self.message).strip()


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text with the metadata needed to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    # Excluded from __hash__ (a mapping is not hashable); still part of __eq__.
    metadata: Mapping[str, str] = field(default_factory=lambda: _freeze(None), hash=False)

    def __post_init__(self) -> None:
        # Defensive: callers may pass a plain dict; freeze it so the dataclass
        # is immutable all the way down. object.__setattr__ is the documented
        # escape hatch for frozen dataclasses.
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def header(self) -> str:
        """Metadata header prepended before embedding (improves retrieval)."""
        title = self.metadata.get("title", self.doc_id)
        section = self.metadata.get("section", "")
        return f"[{title}{' > ' + section if section else ''}]"

    @property
    def text_for_embedding(self) -> str:
        """Header + body: the exact string sent to the embedding model."""
        return f"{self.header}\n{self.text}"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A chunk with its retrieval score and which retriever found it."""

    chunk: Chunk
    score: float
    retriever: Literal["keyword", "vector", "hybrid"]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one or more LLM calls. Supports ``+`` for aggregation."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    @property
    def total(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens
