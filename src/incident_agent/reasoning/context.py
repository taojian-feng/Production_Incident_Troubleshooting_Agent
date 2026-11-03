"""Context engineering: pick, number and budget the evidence the model sees.

The model only ever sees evidence that survived this filter, numbered
``E1..En``. Limiting the context to a few high-signal snippets per domain
reduces noise, cost and hallucinated pathways (design section 3).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from incident_agent.domain.models import Evidence

if TYPE_CHECKING:
    from collections.abc import Iterable

    from incident_agent.domain.enums import EvidenceSource
    from incident_agent.tools.base import RawEvidence


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Caps on what goes into the prompt."""

    per_source: int = 4
    max_chars: int = 9_000


def select_evidence(raw: Iterable[RawEvidence], budget: ContextBudget) -> list[Evidence]:
    """Top-``per_source`` items per domain (round-robin), then trim to ``max_chars``.

    Round-robin across domains guarantees logs, tickets and runbooks are all
    represented even when one domain returns many high-scoring hits.
    """
    by_source: defaultdict[EvidenceSource, list[RawEvidence]] = defaultdict(list)
    for item in raw:
        by_source[item.source].append(item)
    ranked = {
        src: sorted(items, key=lambda e: e.score, reverse=True)[: budget.per_source]
        for src, items in by_source.items()
    }

    interleaved: list[RawEvidence] = []
    for rank in range(budget.per_source):
        interleaved.extend(items[rank] for items in ranked.values() if rank < len(items))

    selected: list[Evidence] = []
    used = 0
    for item in interleaved:
        cost = len(item.title) + len(item.snippet)
        if used + cost > budget.max_chars:
            continue
        used += cost
        selected.append(
            Evidence(
                evidence_id=f"E{len(selected) + 1}",
                source=item.source,
                ref=item.ref,
                title=item.title,
                snippet=item.snippet,
                score=round(item.score, 4),
                uri=item.uri,
            )
        )
    return selected


def render_evidence(evidence: Iterable[Evidence]) -> str:
    """One line per item: ``[E3] (tickets) INC0012345 ... :: snippet``."""
    return "\n".join(
        f"[{e.evidence_id}] ({e.source.value}) {e.title} :: {e.snippet}" for e in evidence
    )
