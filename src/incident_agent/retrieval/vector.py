"""Dense vector index: NumPy matrix + cosine similarity.

A Lambda-friendly vector store: embeddings live in a single ``float32`` matrix
that is persisted to S3 as an ``.npz`` + JSON-lines snapshot, loaded on cold
start and reloaded whenever the snapshot changes. Search is one vectorised
matrix-vector product (no Python loop over rows) followed by ``argpartition``
for O(n) top-k.

This comfortably serves tens of thousands of chunks per service domain. The
:class:`VectorStore` protocol is the seam for swapping in OpenSearch
Serverless or S3 Vectors when the corpus outgrows a single process.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from incident_agent.domain.values import Chunk, SearchHit

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

FloatMatrix = npt.NDArray[np.float32]


class VectorStore(Protocol):
    """Structural interface any vector backend must satisfy."""

    def upsert(self, chunks: Sequence[Chunk], embeddings: FloatMatrix) -> None:
        """Insert or replace chunks with their embeddings (rows aligned)."""
        ...

    def search(
        self, query: FloatMatrix, k: int = 10, *, filters: Mapping[str, str] | None = None
    ) -> list[SearchHit]:
        """Top-k most similar chunks."""
        ...


def _normalise(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32, copy=False)


class InMemoryVectorIndex:
    """Cosine-similarity index over L2-normalised rows."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._matrix: FloatMatrix = np.empty((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._row_of: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def upsert(self, chunks: Sequence[Chunk], embeddings: FloatMatrix) -> None:
        """Insert new rows and overwrite rows whose ``chunk_id`` already exists."""
        if embeddings.shape != (len(chunks), self.dim):
            msg = (
                f"expected embeddings of shape ({len(chunks)}, {self.dim}), got {embeddings.shape}"
            )
            raise ValueError(msg)
        vectors = _normalise(embeddings)
        new_rows: list[FloatMatrix] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            row = self._row_of.get(chunk.chunk_id)
            if row is None:
                self._row_of[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                new_rows.append(vec)
            else:
                self._chunks[row] = chunk
                self._matrix[row] = vec
        if new_rows:
            # One concatenate per batch, not per row: np.vstack in a loop is O(n^2).
            self._matrix = np.concatenate([self._matrix, np.stack(new_rows)])

    def search(
        self, query: FloatMatrix, k: int = 10, *, filters: Mapping[str, str] | None = None
    ) -> list[SearchHit]:
        """Top-k by cosine similarity, optionally restricted by metadata equality."""
        if not self._chunks:
            return []
        scores = self._matrix @ _normalise(query.reshape(-1)).astype(np.float32)
        if filters:
            mask = np.fromiter(
                (all(c.metadata.get(f) == v for f, v in filters.items()) for c in self._chunks),
                dtype=bool,
                count=len(self._chunks),
            )
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchHit(self._chunks[i], float(scores[i]), "vector")
            for i in top
            if np.isfinite(scores[i])
        ]

    # ------------------------------------------------------------------ persistence

    def to_bytes(self) -> tuple[bytes, bytes]:
        """Serialise to ``(npz_bytes, jsonl_bytes)``. No pickle: safe to load."""
        buf = io.BytesIO()
        np.savez_compressed(buf, matrix=self._matrix)
        meta = "\n".join(
            json.dumps(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "metadata": dict(c.metadata),
                }
            )
            for c in self._chunks
        )
        return buf.getvalue(), meta.encode()

    @classmethod
    def from_bytes(cls, npz: bytes, jsonl: bytes) -> InMemoryVectorIndex:
        """Inverse of :meth:`to_bytes`."""
        with np.load(io.BytesIO(npz), allow_pickle=False) as data:
            matrix = data["matrix"].astype(np.float32)
        chunks = [Chunk(**json.loads(line)) for line in jsonl.decode().splitlines() if line]
        if matrix.shape[0] != len(chunks):
            # The two files are written separately, so a reader can catch them mid-update.
            msg = f"snapshot mismatch: {matrix.shape[0]} vectors, {len(chunks)} chunks"
            raise ValueError(msg)
        index = cls(dim=matrix.shape[1])
        index._matrix = matrix
        index._chunks = chunks
        index._row_of = {c.chunk_id: i for i, c in enumerate(chunks)}
        return index

    def chunks(self) -> Iterable[Chunk]:
        """All indexed chunks."""
        return iter(self._chunks)
