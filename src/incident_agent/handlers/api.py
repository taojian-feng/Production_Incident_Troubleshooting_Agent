"""Lambda: HTTP API for the analyst UI.

Routes (API Gateway HTTP API, payload v2, IAM-authorised):

* ``POST /alerts``                  submit an alert; triage runs async -> 202 + report id
* ``GET  /reports/{report_id}``     fetch a report
* ``POST /reports/{report_id}/feedback``  thumbs up/down + optional correction
* ``POST /tickets/jira`` and ``/tickets/servicenow``  index a ticket

The Analyst UI uses ``/api/*`` routes, which accept a Cognito sign-in token instead of IAM.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ValidationError

from incident_agent.domain.models import (
    Alert,
    Feedback,
    HistoricalIncident,
    JiraWebhook,
    ReportSummary,
    ServiceNowIncident,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from incident_agent.storage.repositories import FeedbackStore, ReportStore

logger = logging.getLogger()

Response = dict[str, Any]

_REPORT_LIST_SIZE = 100


def _json(status: int, body: object) -> Response:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body, default=str),
    }


def _caller(event: dict[str, Any]) -> str | None:
    """Who the API Gateway authorizer verified: a signed-in analyst or an IAM principal."""
    authorizer = event.get("requestContext", {}).get("authorizer", {})
    email: str | None = authorizer.get("jwt", {}).get("claims", {}).get("email")
    principal: str | None = authorizer.get("iam", {}).get("userArn")
    return email or principal


@dataclass(slots=True)
class Api:
    """Pure request -> response logic; AWS clients are injected."""

    reports: ReportStore
    feedback: FeedbackStore
    start_triage: Callable[[Alert, str], None]
    publish_ticket: Callable[[HistoricalIncident], None]

    def handle(self, event: dict[str, Any]) -> Response:
        """Dispatch on ``routeKey`` via a dict (O(1), no if/elif ladder)."""
        routes: dict[str, Callable[[dict[str, Any]], Response]] = {
            "POST /alerts": self.submit_alert,
            "GET /reports/{report_id}": self.get_report,
            "POST /reports/{report_id}/feedback": self.post_feedback,
            "GET /api/reports": self.list_reports,
            "GET /api/reports/{report_id}": self.get_report_with_feedback,
            "POST /api/reports/{report_id}/feedback": self.post_feedback,
            "POST /tickets/jira": self.receive_jira,
            "POST /tickets/servicenow": self.receive_servicenow,
        }
        route = routes.get(event.get("routeKey", ""))
        if route is None:
            return _json(404, {"error": "not found"})
        try:
            return route(event)
        except ValidationError as exc:
            return _json(
                422, {"error": "invalid request", "details": exc.errors(include_url=False)}
            )
        except json.JSONDecodeError:
            return _json(400, {"error": "body must be JSON"})

    def submit_alert(self, event: dict[str, Any]) -> Response:
        """Validate, then hand off to the triage Lambda asynchronously."""
        alert = Alert.model_validate_json(event.get("body") or "{}")
        report_id = f"rpt-{uuid4().hex[:12]}"
        self.start_triage(alert, report_id)
        return _json(202, {"report_id": report_id, "alert_id": alert.alert_id})

    def receive_jira(self, event: dict[str, Any]) -> Response:
        """Index a Jira issue webhook."""
        webhook = JiraWebhook.model_validate_json(event.get("body") or "{}")
        return self._publish(webhook.to_incident())

    def receive_servicenow(self, event: dict[str, Any]) -> Response:
        """Index a ServiceNow incident record."""
        record = ServiceNowIncident.model_validate_json(event.get("body") or "{}")
        return self._publish(record.to_incident())

    def _publish(self, incident: HistoricalIncident) -> Response:
        self.publish_ticket(incident)
        return _json(202, {"incident_id": incident.incident_id})

    def get_report(self, event: dict[str, Any]) -> Response:
        """Return the report JSON, including its computed ``status``."""
        report = self.reports.get(event["pathParameters"]["report_id"])
        if report is None:
            return _json(404, {"error": "report not found (triage may still be running)"})
        return _json(200, report.model_dump(mode="json"))

    def list_reports(self, _event: dict[str, Any]) -> Response:
        """The Analyst UI's report list."""
        reports = self.reports.recent(_REPORT_LIST_SIZE)
        return _json(
            200, {"reports": [ReportSummary.of(r).model_dump(mode="json") for r in reports]}
        )

    def get_report_with_feedback(self, event: dict[str, Any]) -> Response:
        """One report for the Analyst UI, with the feedback already given on it."""
        report_id = event["pathParameters"]["report_id"]
        report = self.reports.get(report_id)
        if report is None:
            return _json(404, {"error": "report not found"})
        feedback = [f.model_dump(mode="json") for f in self.feedback.for_report(report_id)]
        return _json(200, {"report": report.model_dump(mode="json"), "feedback": feedback})

    def post_feedback(self, event: dict[str, Any]) -> Response:
        """Store feedback; it feeds the evaluation dataset (continuous improvement)."""
        report_id = event["pathParameters"]["report_id"]
        if self.reports.get(report_id) is None:
            return _json(404, {"error": "report not found"})
        body = json.loads(event.get("body") or "{}")
        caller = _caller(event)
        feedback = Feedback.model_validate(
            {**body, "report_id": report_id, "user": caller or body.get("user", "anonymous")}
        )
        self.feedback.add(feedback)
        return _json(201, {"ok": True})


def handler(event: dict[str, Any], context: object) -> Response:  # pragma: no cover
    """Lambda entry point."""
    del context
    import os  # noqa: PLC0415

    import boto3  # noqa: PLC0415

    from incident_agent.config import Settings  # noqa: PLC0415
    from incident_agent.storage.repositories import (  # noqa: PLC0415
        DynamoFeedbackStore,
        DynamoReportStore,
    )

    settings = Settings.from_env()
    dynamo = boto3.resource("dynamodb", region_name=settings.region)
    lambda_client = boto3.client("lambda", region_name=settings.region)
    triage_fn = os.environ["IA_TRIAGE_FUNCTION"]

    def start(alert: Alert, report_id: str) -> None:
        lambda_client.invoke(
            FunctionName=triage_fn,
            InvocationType="Event",
            Payload=json.dumps({"alert": alert.model_dump(mode="json"), "report_id": report_id}),
        )

    s3 = boto3.client("s3", region_name=settings.region)

    def publish(incident: HistoricalIncident) -> None:
        # The upload triggers the ingest function, which indexes it and upserts the table.
        s3.put_object(
            Bucket=settings.knowledge_bucket,
            Key=f"knowledge/{incident.incident_id}.json",
            Body=incident.model_dump_json().encode(),
        )

    api = Api(
        reports=DynamoReportStore(dynamo.Table(settings.reports_table)),
        feedback=DynamoFeedbackStore(dynamo.Table(settings.feedback_table)),
        start_triage=start,
        publish_ticket=publish,
    )
    return api.handle(event)
