"""Ingestion pipeline: documents -> chunks -> embeddings -> both indexes.

Streaming end to end: ``chunk_documents`` is a generator and
``itertools.batched`` pulls fixed-size batches from it, so only one batch of
chunks and embeddings is in memory at a time. Batches are embedded
concurrently (bounded) because embedding calls are I/O-bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import batched
from typing import TYPE_CHECKING

from incident_agent.core.rate_limit import gather_bounded
from incident_agent.ingestion.chunking import ChunkingConfig, chunk_documents

if TYPE_CHECKING:
    from collections.abc import Iterable

    from incident_agent.domain.models import SourceDocument
    from incident_agent.domain.values import Chunk
    from incident_agent.llm.base import EmbeddingProvider
    from incident_agent.retrieval.keyword import BM25Index
    from incident_agent.retrieval.vector import FloatMatrix, InMemoryVectorIndex


@dataclass(frozen=True, slots=True)
class IngestStats:
    """What an ingestion run did."""

    documents: int
    chunks: int
    batches: int


@dataclass(slots=True)
class IngestionPipeline:
    """Keeps keyword and vector indexes in sync for every ingested document."""

    embedder: EmbeddingProvider
    keyword: BM25Index
    vector: InMemoryVectorIndex
    chunking: ChunkingConfig = ChunkingConfig()
    batch_size: int = 16
    concurrency: int = 4

    async def ingest(self, docs: Iterable[SourceDocument]) -> IngestStats:
        """Chunk, embed and index ``docs``; re-ingesting a doc upserts its chunks."""
        doc_ids: set[str] = set()

        def tracked() -> Iterable[SourceDocument]:
            for doc in docs:
                doc_ids.add(doc.doc_id)
                yield doc

        chunk_count = 0
        batch_count = 0
        # Process a window of `concurrency` batches at a time: bounded memory,
        # concurrent embedding calls within the window.
        for window in batched(
            batched(chunk_documents(tracked(), self.chunking), self.batch_size), self.concurrency
        ):
            embeddings = await gather_bounded(
                (partial(self.embedder.embed, [c.text_for_embedding for c in b]) for b in window),
                limit=self.concurrency,
            )
            for chunk_batch, matrix in zip(window, embeddings, strict=True):
                self._index(chunk_batch, matrix)
                chunk_count += len(chunk_batch)
                batch_count += 1
        return IngestStats(documents=len(doc_ids), chunks=chunk_count, batches=batch_count)

    def _index(self, chunks: tuple[Chunk, ...], matrix: FloatMatrix) -> None:
        self.keyword.add(chunks)
        self.vector.upsert(chunks, matrix)
