"""Hybrid retrieval: query planning + reciprocal rank fusion (design section 2.2).

The planner inspects the *structure* of the alert:

* explicit identifiers and little prose  -> KEYWORD (exact match wins)
* no identifiers                         -> VECTOR (semantic intent)
* both                                   -> HYBRID (run both, fuse ranks)

Fusion uses Reciprocal Rank Fusion, which combines rankings without needing
BM25 and cosine scores to be on the same scale.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from incident_agent.domain.enums import RetrievalMode
from incident_agent.domain.state import RetrievalPlan
from incident_agent.domain.values import SearchHit
from incident_agent.retrieval.text import extract_error_codes, extract_ticket_ids, tokenize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from incident_agent.domain.models import Alert
    from incident_agent.domain.values import Chunk
    from incident_agent.llm.base import EmbeddingProvider
    from incident_agent.retrieval.keyword import BM25Index
    from incident_agent.retrieval.vector import InMemoryVectorIndex

KEYWORD_ONLY_MAX_WORDS = 12


def plan_retrieval(alert: Alert) -> RetrievalPlan:
    """Decide keyword vs vector vs hybrid from the alert's structure alone.

    Deterministic and free: no LLM call is spent on routing.
    """
    text = alert.text
    tickets = extract_ticket_ids(text)
    codes = extract_error_codes(text)
    identifiers = tickets + codes
    prose_words = len(tokenize(text)) - len(identifiers)

    if identifiers and prose_words <= KEYWORD_ONLY_MAX_WORDS:
        mode = RetrievalMode.KEYWORD
    elif identifiers:
        mode = RetrievalMode.HYBRID
    else:
        mode = RetrievalMode.VECTOR
    return RetrievalPlan(
        mode=mode, query=text, identifiers=identifiers, error_codes=codes, ticket_ids=tickets
    )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], k: int = 60, top_n: int = 10
) -> list[SearchHit]:
    """Fuse ranked lists: ``score(d) = sum(1 / (k + rank_i(d)))``.

    A ``defaultdict(float)`` accumulates scores; a plain dict remembers the
    chunk object for each id (first occurrence wins).
    """
    fused: defaultdict[str, float] = defaultdict(float)
    chunk_of: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk.chunk_id] += 1.0 / (k + rank)
            chunk_of.setdefault(hit.chunk.chunk_id, hit.chunk)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [SearchHit(chunk_of[cid], score, "hybrid") for cid, score in ordered]


@dataclass(slots=True)
class HybridRetriever:
    """Runs the planned retrieval strategy over both indexes."""

    keyword: BM25Index
    vector: InMemoryVectorIndex
    embedder: EmbeddingProvider
    candidates: int = 20

    async def search(
        self, plan: RetrievalPlan, k: int = 8, filters: Mapping[str, str] | None = None
    ) -> list[SearchHit]:
        """Execute ``plan``; falls back to unfiltered search if filters empty the result."""
        hits = await self._search(plan, k, filters)
        if not hits and filters:
            hits = await self._search(plan, k, None)
        return hits

    async def _search(
        self, plan: RetrievalPlan, k: int, filters: Mapping[str, str] | None
    ) -> list[SearchHit]:
        keyword_hits: list[SearchHit] = []
        vector_hits: list[SearchHit] = []
        if plan.mode in {RetrievalMode.KEYWORD, RetrievalMode.HYBRID}:
            keyword_hits = self.keyword.search(
                plan.query, self.candidates, filters=filters, exact_terms=plan.identifiers
            )
        if plan.mode in {RetrievalMode.VECTOR, RetrievalMode.HYBRID}:
            (query_vec,) = await self.embedder.embed([plan.query])
            vector_hits = self.vector.search(query_vec, self.candidates, filters=filters)

        if plan.mode is RetrievalMode.KEYWORD:
            return keyword_hits[:k]
        if plan.mode is RetrievalMode.VECTOR:
            return vector_hits[:k]
        return reciprocal_rank_fusion([keyword_hits, vector_hits], top_n=k)
