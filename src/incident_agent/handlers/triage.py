"""Lambda: EventBridge (CloudWatch alarm -> ALARM) or async API invoke -> triage report.

Heavy objects (clients, the loaded index) are reused across warm invocations.
Components are rebuilt when the S3 index snapshot changes, so new knowledge is
used without waiting for a cold start.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from typing import TYPE_CHECKING, Any

import boto3

from incident_agent.app import Components, build_aws_components
from incident_agent.config import Settings
from incident_agent.core.refresh import RefreshingValue
from incident_agent.domain.models import Alert, TriageReport
from incident_agent.observability.metrics import MetricsLogger, record_triage
from incident_agent.storage.repositories import DynamoReportStore, S3IndexSnapshot

if TYPE_CHECKING:
    from incident_agent.storage.repositories import ReportStore

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@functools.cache
def _settings() -> Settings:
    return Settings.from_env()


@functools.cache
def _refreshing_components() -> RefreshingValue[Components]:  # pragma: no cover - needs AWS
    settings = _settings()
    client = boto3.client("s3", region_name=settings.region)
    return RefreshingValue(
        version=S3IndexSnapshot(client, settings.knowledge_bucket).version,
        build=lambda: build_aws_components(settings),
    )


def _components() -> Components:  # pragma: no cover - exercised in AWS
    return _refreshing_components().get()


@functools.cache
def _report_store() -> ReportStore:  # pragma: no cover - exercised in AWS
    table = boto3.resource("dynamodb", region_name=_settings().region).Table(
        _settings().reports_table
    )
    return DynamoReportStore(table)


def parse_alert(event: dict[str, Any]) -> Alert:
    """Accept either an EventBridge alarm event or ``{"alert": {...}}`` from the API."""
    if event.get("detail-type") == "CloudWatch Alarm State Change":
        return Alert.from_cloudwatch_alarm(event)
    return Alert.model_validate(event["alert"])


async def triage(
    alert: Alert, components: Components, store: ReportStore, report_id: str | None = None
) -> TriageReport:
    """Run the agent, persist the report, emit metrics."""
    report, _state = await components.agent.run(alert)
    if report_id:
        report = report.model_copy(update={"report_id": report_id})
    store.save(report)
    record_triage(MetricsLogger({"Service": alert.service}), report, components.registry)
    return report


def _notify(report: TriageReport) -> None:  # pragma: no cover - exercised in AWS
    topic = _settings().notify_topic_arn
    if not topic:
        return
    first = report.steps[0].content if report.steps else ""
    boto3.client("sns", region_name=_settings().region).publish(
        TopicArn=topic,
        Subject=f"[{report.alert.severity}] triage ready: {report.alert.title}"[:100],
        Message=json.dumps(
            {"report_id": report.report_id, "status": report.status, "summary": first}
        ),
    )


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:  # pragma: no cover
    """Lambda entry point."""
    del context
    alert = parse_alert(event)
    report = asyncio.run(triage(alert, _components(), _report_store(), event.get("report_id")))
    _notify(report)
    logger.info("triaged %s -> %s (%s)", alert.alert_id, report.report_id, report.status)
    return {"report_id": report.report_id, "status": report.status}
