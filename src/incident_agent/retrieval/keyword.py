"""BM25 keyword index built on plain dicts.

Data-structure choices:

* ``postings: dict[term, dict[chunk_id, tf]]`` - an inverted index. Looking up
  a term is O(1) average, and scoring touches only chunks that contain at least
  one query term instead of scanning the whole corpus.
* ``Counter`` computes term frequencies per chunk in one pass.
* ``heapq.nlargest`` returns top-k in O(n log k) instead of sorting all n.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from incident_agent.domain.values import Chunk, SearchHit
from incident_agent.retrieval.text import tokenize

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class BM25Index:
    """Okapi BM25 with an exact-identifier boost."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, exact_boost: float = 2.0) -> None:
        self.k1 = k1
        self.b = b
        self.exact_boost = exact_boost
        self._postings: defaultdict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        self._chunks: dict[str, Chunk] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def add(self, chunks: Iterable[Chunk]) -> int:
        """Index chunks; re-adding an existing ``chunk_id`` replaces it (upsert)."""
        added = 0
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                self.remove(chunk.chunk_id)
            terms = Counter(tokenize(chunk.text_for_embedding))
            for term, tf in terms.items():
                self._postings[term][chunk.chunk_id] = tf
            length = sum(terms.values())
            self._lengths[chunk.chunk_id] = length
            self._total_length += length
            self._chunks[chunk.chunk_id] = chunk
            added += 1
        return added

    def remove(self, chunk_id: str) -> None:
        """Drop a chunk from every posting list it appears in."""
        chunk = self._chunks.pop(chunk_id)
        for term in set(tokenize(chunk.text_for_embedding)):
            postings = self._postings.get(term)
            if postings is not None:
                postings.pop(chunk_id, None)
                if not postings:
                    del self._postings[term]
        self._total_length -= self._lengths.pop(chunk_id)

    def chunks(self) -> Iterable[Chunk]:
        """All indexed chunks (for snapshotting)."""
        return self._chunks.values()

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        filters: Mapping[str, str] | None = None,
        exact_terms: Iterable[str] = (),
    ) -> list[SearchHit]:
        """Top-k chunks by BM25; chunks containing an ``exact_terms`` id get boosted."""
        if not self._chunks:
            return []
        n_docs = len(self._chunks)
        avg_len = self._total_length / n_docs
        scores: defaultdict[str, float] = defaultdict(float)

        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for chunk_id, tf in postings.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[chunk_id] / avg_len)
                scores[chunk_id] += idf * tf * (self.k1 + 1) / norm

        for ident in {t.lower() for t in exact_terms}:
            for chunk_id in self._postings.get(ident, {}):
                scores[chunk_id] = scores[chunk_id] * self.exact_boost + self.exact_boost

        if filters:
            scores = defaultdict(
                float,
                {
                    cid: s
                    for cid, s in scores.items()
                    if all(self._chunks[cid].metadata.get(f) == v for f, v in filters.items())
                },
            )

        best = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        return [SearchHit(self._chunks[cid], score, "keyword") for cid, score in best]
