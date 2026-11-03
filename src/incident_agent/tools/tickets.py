"""Tickets domain: historical incidents mirrored from ServiceNow / Jira.

Exact ticket ids in the alert are fetched directly (keyword path). Otherwise
we pull the service's recent incidents and rank them by term overlap with the
alert. Semantic search over incident *text* happens in the knowledge domain,
where incidents are also indexed as documents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from boto3.dynamodb.conditions import Key

from incident_agent.domain.enums import EvidenceSource
from incident_agent.domain.models import HistoricalIncident
from incident_agent.retrieval.text import tokenize
from incident_agent.tools.base import RawEvidence

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mypy_boto3_dynamodb.service_resource import Table

    from incident_agent.domain.models import Alert
    from incident_agent.domain.state import RetrievalPlan


class IncidentRepository(Protocol):
    """Storage seam for historical incidents."""

    def get_many(self, incident_ids: Sequence[str]) -> list[HistoricalIncident]:
        """Fetch by id; unknown ids are skipped."""
        ...

    def recent_for_service(self, service: str, limit: int) -> list[HistoricalIncident]:
        """Most recent incidents for a service, newest first."""
        ...

    def put(self, incident: HistoricalIncident) -> None:
        """Insert or replace one incident."""
        ...


@dataclass(slots=True)
class InMemoryIncidentRepository:
    """Dict-backed repository: O(1) lookups by id."""

    _by_id: dict[str, HistoricalIncident] = field(default_factory=dict)

    @classmethod
    def of(cls, incidents: Iterable[HistoricalIncident]) -> InMemoryIncidentRepository:
        """Build from an iterable (dict comprehension keyed by id)."""
        return cls({i.incident_id: i for i in incidents})

    def get_many(self, incident_ids: Sequence[str]) -> list[HistoricalIncident]:
        """Fetch by id; unknown ids are skipped."""
        return [self._by_id[i] for i in incident_ids if i in self._by_id]

    def recent_for_service(self, service: str, limit: int) -> list[HistoricalIncident]:
        """Most recent incidents for a service, newest first."""
        matching = (i for i in self._by_id.values() if i.service == service)
        return sorted(matching, key=lambda i: i.opened_at, reverse=True)[:limit]

    def put(self, incident: HistoricalIncident) -> None:
        """Insert or replace one incident."""
        self._by_id[incident.incident_id] = incident


@dataclass(slots=True)
class DynamoIncidentRepository:
    """DynamoDB table ``pk=incident_id`` with GSI ``by_service (service, opened_at)``."""

    table: Table
    service_index: str = "by_service"

    def get_many(self, incident_ids: Sequence[str]) -> list[HistoricalIncident]:
        """``get_item`` per id (ids per alert are few; BatchGet adds little here)."""
        found: list[HistoricalIncident] = []
        for incident_id in dict.fromkeys(incident_ids):
            item = self.table.get_item(Key={"incident_id": incident_id}).get("Item")
            if item is not None:
                found.append(HistoricalIncident.model_validate(item))
        return found

    def recent_for_service(self, service: str, limit: int) -> list[HistoricalIncident]:
        """Query the GSI newest-first."""
        response = self.table.query(
            IndexName=self.service_index,
            KeyConditionExpression=Key("service").eq(service),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [HistoricalIncident.model_validate(item) for item in response.get("Items", [])]

    def put(self, incident: HistoricalIncident) -> None:
        """Store as plain JSON types (datetimes as ISO strings)."""
        item = incident.model_dump(mode="json", exclude_none=True)
        self.table.put_item(Item=item)


def _to_evidence(incident: HistoricalIncident, score: float) -> RawEvidence:
    return RawEvidence(
        source=EvidenceSource.TICKETS,
        ref=incident.incident_id,
        title=f"{incident.incident_id} [{incident.severity}] {incident.title}",
        snippet=(
            f"{incident.incident_id}: root cause: {incident.root_cause} "
            f"| resolution: {incident.resolution}"
        )[:600],
        score=score,
    )


@dataclass(slots=True)
class IncidentHistoryServer:
    """Tickets domain server."""

    repo: IncidentRepository
    lookback: int = 50
    top_k: int = 3

    @property
    def domain(self) -> EvidenceSource:
        """Tickets domain."""
        return EvidenceSource.TICKETS

    async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
        """Exact ids first (score 1.0), then the best term-overlap matches."""
        return await asyncio.to_thread(self._gather, alert, plan)

    def _gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
        exact = self.repo.get_many(plan.ticket_ids)
        evidence = [_to_evidence(i, 1.0) for i in exact]
        seen = {i.incident_id for i in exact}

        query_terms = set(tokenize(alert.text))
        if not query_terms:
            return evidence

        scored: list[tuple[float, HistoricalIncident]] = []
        for incident in self.repo.recent_for_service(alert.service, self.lookback):
            if incident.incident_id in seen:
                continue
            terms = set(tokenize(f"{incident.title} {incident.summary} {incident.root_cause}"))
            # Jaccard-like overlap on sets: O(min(len(a), len(b))).
            overlap = len(query_terms & terms) / len(query_terms)
            if overlap > 0:
                scored.append((overlap, incident))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        evidence.extend(_to_evidence(i, s) for s, i in scored[: self.top_k])
        return evidence
