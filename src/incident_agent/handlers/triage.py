"""Lambda: EventBridge (CloudWatch alarm -> ALARM) or async API invoke -> triage report.

Clients are reused across warm invocations. Evidence comes from the three domain
MCP servers, which each hold their own permissions and data.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING, Any

import boto3

from incident_agent.app import build_aws_components
from incident_agent.config import Settings
from incident_agent.domain.models import Alert, TriageReport
from incident_agent.handlers.notification import format_report_email
from incident_agent.observability.metrics import MetricsLogger, record_triage
from incident_agent.storage.repositories import DynamoReportStore

if TYPE_CHECKING:
    from incident_agent.app import Runtime
    from incident_agent.storage.repositories import ReportStore

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@functools.cache
def _settings() -> Settings:
    return Settings.from_env()


@functools.cache
def _components() -> Runtime:  # pragma: no cover - exercised in AWS
    return build_aws_components(_settings())


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
    alert: Alert, components: Runtime, store: ReportStore, report_id: str | None = None
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
    subject, body = format_report_email(report)
    boto3.client("sns", region_name=_settings().region).publish(
        TopicArn=topic, Subject=subject, Message=body
    )


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:  # pragma: no cover
    """Lambda entry point."""
    del context
    alert = parse_alert(event)
    report = asyncio.run(triage(alert, _components(), _report_store(), event.get("report_id")))
    _notify(report)
    logger.info("triaged %s -> %s (%s)", alert.alert_id, report.report_id, report.status)
    return {"report_id": report.report_id, "status": report.status}
