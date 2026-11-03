"""Tokenisation and identifier extraction shared by indexing and query planning."""

from __future__ import annotations

import re
from typing import Final

# Identifiers must survive tokenisation intact: "ERR_GATEWAY_TIMEOUT",
# "INC0012345", "PAY-1432", "ORA-00060", "HTTP 503" are the strongest signals in
# an alert and are exactly what keyword search is for.
TICKET_RE: Final = re.compile(r"\b(INC\d{7}|[A-Z][A-Z0-9]{1,9}-\d{1,6})\b")
ERROR_CODE_RE: Final = re.compile(
    r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]{2,}|(?:ORA|SQL|ERR)-\d{4,5}|HTTP[ _]?[45]\d\d|E\d{4,})\b"
)
_TOKEN_RE: Final = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_\-][A-Za-z0-9]+)*|\d{3,}")

STOPWORDS: Final = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
        "will", "with", "this", "after", "before", "when", "while", "during", "not", "no",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens; hyphen/underscore compounds kept whole, stopwords dropped."""
    return [t for t in (m.lower() for m in _TOKEN_RE.findall(text)) if t not in STOPWORDS]


def extract_ticket_ids(text: str) -> tuple[str, ...]:
    """Ticket ids in first-seen order without duplicates (dict preserves order)."""
    candidates = (m for m in TICKET_RE.findall(text) if not ERROR_CODE_RE.fullmatch(m))
    return tuple(dict.fromkeys(candidates))


def extract_error_codes(text: str) -> tuple[str, ...]:
    """Error codes in first-seen order without duplicates."""
    return tuple(dict.fromkeys(ERROR_CODE_RE.findall(text)))
