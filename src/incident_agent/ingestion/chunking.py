"""Section-aware, overlapping chunking for RAG.

Runbooks and post-mortems are Markdown. We split on headings first so a chunk
never straddles two unrelated sections, then slide a fixed-size word window
with overlap inside each section. Every chunk carries metadata (title, section
path, doc type, service) that is both stored for filtering/citation and
prepended as a header before embedding.

Everything here is a generator: a 500 MB corpus is processed one section at a
time, never fully materialised in memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from incident_agent.domain.values import Chunk

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from incident_agent.domain.models import SourceDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Window size and overlap, in words (a cheap, model-agnostic token proxy)."""

    size: int = 180
    overlap: int = 40
    min_words: int = 8

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.size:
            msg = f"overlap ({self.overlap}) must be in [0, size={self.size})"
            raise ValueError(msg)

    @property
    def stride(self) -> int:
        """How far the window advances each step."""
        return self.size - self.overlap


@dataclass(frozen=True, slots=True)
class Section:
    """A run of text under one heading path, e.g. ``("Runbook", "Mitigation")``."""

    path: tuple[str, ...]
    text: str


def iter_sections(markdown: str) -> Iterator[Section]:
    """Yield sections with their full heading path (a breadcrumb).

    The heading stack is a list used as a stack; we store it as a tuple in the
    output because tuples are immutable and hashable (safe as dict keys).
    """
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> Iterator[Section]:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            yield Section(tuple(title for _, title in stack), text)

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue
        yield from flush()
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2)))
    yield from flush()


def _windows(words: list[str], cfg: ChunkingConfig) -> Iterator[list[str]]:
    """Sliding windows; the last window is kept only if it adds new words."""
    if len(words) <= cfg.size:
        yield words
        return
    for start in range(0, len(words), cfg.stride):
        window = words[start : start + cfg.size]
        yield window
        if start + cfg.size >= len(words):
            return


def chunk_document(doc: SourceDocument, cfg: ChunkingConfig | None = None) -> Iterator[Chunk]:
    """Split one document into overlapping, metadata-rich chunks.

    Chunk ids are deterministic (``doc_id#section-slug:n``) so re-ingesting an
    updated document overwrites its chunks instead of duplicating them.
    """
    cfg = cfg or ChunkingConfig()
    base_meta = {"title": doc.title, "doc_type": doc.doc_type}
    if doc.service:
        base_meta["service"] = doc.service
    if doc.uri:
        base_meta["uri"] = doc.uri

    for section in iter_sections(doc.body):
        # The H1 usually repeats the document title, which the header already carries.
        path = section.path[1:] if section.path[:1] == (doc.title,) else section.path
        section_name = " > ".join(path)
        slug = _SLUG.sub("-", section_name.lower()).strip("-") or "body"
        words = section.text.split()
        for n, window in enumerate(_windows(words, cfg)):
            if len(window) < cfg.min_words and n > 0:
                continue
            yield Chunk(
                chunk_id=f"{doc.doc_id}#{slug}:{n}",
                doc_id=doc.doc_id,
                text=" ".join(window),
                # dict unpacking builds a new dict per chunk: no shared mutable state
                metadata={**base_meta, "section": section_name, "position": str(n)},
            )


def chunk_documents(
    docs: Iterable[SourceDocument], cfg: ChunkingConfig | None = None
) -> Iterator[Chunk]:
    """Lazily chunk a stream of documents (generator of generators)."""
    for doc in docs:
        yield from chunk_document(doc, cfg)
