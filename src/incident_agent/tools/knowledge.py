"""Knowledge domain: hybrid search over runbooks, post-mortems and past incidents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from incident_agent.domain.enums import EvidenceSource
from incident_agent.tools.base import RawEvidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from incident_agent.domain.models import Alert
    from incident_agent.domain.state import RetrievalPlan
    from incident_agent.domain.values import Chunk, SearchHit
    from incident_agent.retrieval.hybrid import HybridRetriever


# Log patterns only record that an error was seen, so they must not displace a runbook's fix.
LOG_PATTERN_WEIGHT: Final = 0.5
MAX_LOG_PATTERNS: Final = 1
_ACTION_SECTIONS: Final = re.compile(r"mitigation|resolution|remediation", re.IGNORECASE)


def _evidence(chunk: Chunk, score: float) -> RawEvidence:
    return RawEvidence(
        source=EvidenceSource.KNOWLEDGE,
        ref=chunk.chunk_id,
        title=f"{chunk.metadata.get('title', chunk.doc_id)} > {chunk.metadata.get('section', '')}",
        snippet=chunk.text[:700],
        score=score,
        uri=chunk.metadata.get("uri"),
    )


def _is_log_pattern(hit: SearchHit) -> bool:
    return hit.chunk.metadata.get("doc_type") == "log_pattern"


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
        documents = [hit for hit in hits if not _is_log_pattern(hit)]
        patterns = [hit for hit in hits if _is_log_pattern(hit)][:MAX_LOG_PATTERNS]
        return [
            *(_evidence(hit.chunk, hit.score) for hit in documents),
            *self._action_sections(documents),
            *(_evidence(hit.chunk, hit.score * LOG_PATTERN_WEIGHT) for hit in patterns),
        ]

    def _action_sections(self, hits: Sequence[SearchHit]) -> list[RawEvidence]:
        """Add the fix from the best-matching document.

        An error code usually appears in a runbook's symptoms and not in its
        mitigation, so search alone can find the problem but miss the fix.
        """
        if not hits:
            return []
        best = hits[0]
        found = {hit.chunk.chunk_id for hit in hits}
        return [
            # Ranked just below the best match, so the evidence budget keeps the fix.
            _evidence(chunk, best.score * 0.999)
            for chunk in self.retriever.vector.chunks()
            if chunk.doc_id == best.chunk.doc_id
            and chunk.chunk_id not in found
            and _ACTION_SECTIONS.search(chunk.metadata.get("section", ""))
        ]
