"""Knowledge domain: hybrid search over runbooks, post-mortems and past incidents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from incident_agent.domain.enums import EvidenceSource
from incident_agent.tools.base import RawEvidence

if TYPE_CHECKING:
    from incident_agent.domain.models import Alert
    from incident_agent.domain.state import RetrievalPlan
    from incident_agent.retrieval.hybrid import HybridRetriever


@dataclass(slots=True)
class KnowledgeServer:
    """Wraps :class:`HybridRetriever`; prefers documents for the alerting service."""

    retriever: HybridRetriever
    top_k: int = 5

    @property
    def domain(self) -> EvidenceSource:
        """Knowledge domain."""
        return EvidenceSource.KNOWLEDGE

    async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
        """Service-scoped hybrid search, falling back to the whole corpus."""
        hits = await self.retriever.search(plan, k=self.top_k, filters={"service": alert.service})
        return [
            RawEvidence(
                source=EvidenceSource.KNOWLEDGE,
                ref=hit.chunk.chunk_id,
                title=f"{hit.chunk.metadata.get('title', hit.chunk.doc_id)}"
                f" > {hit.chunk.metadata.get('section', '')}",
                snippet=hit.chunk.text[:700],
                score=hit.score,
                uri=hit.chunk.metadata.get("uri"),
            )
            for hit in hits
        ]
