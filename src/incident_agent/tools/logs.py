"""Logs domain: CloudWatch Logs around the alert window."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from incident_agent.domain.enums import EvidenceSource
from incident_agent.ingestion.log_stream import (
    dedupe,
    is_error,
    iter_cloudwatch_events,
    top_signatures,
)
from incident_agent.tools.base import RawEvidence

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from mypy_boto3_logs import CloudWatchLogsClient

    from incident_agent.domain.models import Alert
    from incident_agent.domain.state import RetrievalPlan
    from incident_agent.domain.values import LogEvent


def _filter_pattern(plan: RetrievalPlan) -> str:
    """CloudWatch filter syntax: quoted terms are OR-ed with ``?``."""
    if plan.error_codes:
        return " ".join(f'?"{code}"' for code in plan.error_codes) + " ?ERROR ?Exception"
    return "?ERROR ?Exception ?FATAL"


def summarise_events(events: Iterable[LogEvent], top_k: int) -> list[RawEvidence]:
    """Turn a stream of events into the top-k error signatures as evidence."""
    errors = (e for e in dedupe(events) if is_error(e))
    return [
        RawEvidence(
            source=EvidenceSource.LOGS,
            ref=f"log:{event.event_id}",
            title=f"{count}x in {event.log_group}",
            snippet=f"{event.timestamp.isoformat()} {event.message[:400]}",
            score=float(count),
        )
        for _sig, count, event in top_signatures(errors, k=top_k)
    ]


@dataclass(slots=True)
class CloudWatchLogsServer:
    """Reads only the alert's own log group, inside a bounded time window."""

    client: CloudWatchLogsClient
    default_log_group_for: Callable[[str], str] = lambda service: f"/app/{service}"
    window: timedelta = timedelta(minutes=15)
    top_k: int = 5
    max_events: int = 5_000

    @property
    def domain(self) -> EvidenceSource:
        """Logs domain."""
        return EvidenceSource.LOGS

    async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
        """Top error signatures from ``[t - window, t + window/3]``."""
        group = alert.log_group or self.default_log_group_for(alert.service)
        start = alert.triggered_at - self.window
        end = alert.triggered_at + self.window / 3

        def run() -> list[RawEvidence]:
            events = iter_cloudwatch_events(
                self.client, group, start, end, _filter_pattern(plan), self.max_events
            )
            return summarise_events(events, self.top_k)

        return await asyncio.to_thread(run)


@dataclass(slots=True)
class StaticLogsServer:
    """Logs from an in-memory list (local demo / tests)."""

    events: list[LogEvent]
    top_k: int = 5

    @property
    def domain(self) -> EvidenceSource:
        """Logs domain."""
        return EvidenceSource.LOGS

    async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
        """Same summarisation as CloudWatch, over the static events for the service."""
        del plan
        scoped = (e for e in self.events if alert.service in e.log_group)
        return summarise_events(scoped, self.top_k)
