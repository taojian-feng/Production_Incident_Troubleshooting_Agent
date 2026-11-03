"""Lambda: HTTP API for the analyst UI.

Routes (API Gateway HTTP API, payload v2, IAM-authorised):

* ``POST /alerts``                  submit an alert; triage runs async -> 202 + report id
* ``GET  /reports/{report_id}``     fetch a report
* ``POST /reports/{report_id}/feedback``  thumbs up/down + optional correction
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ValidationError

from incident_agent.domain.models import Alert, Feedback

if TYPE_CHECKING:
    from collections.abc import Callable

    from incident_agent.storage.repositories import FeedbackStore, ReportStore

logger = logging.getLogger()

Response = dict[str, Any]


def _json(status: int, body: object) -> Response:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body, default=str),
    }


@dataclass(slots=True)
class Api:
    """Pure request -> response logic; AWS clients are injected."""

    reports: ReportStore
    feedback: FeedbackStore
    start_triage: Callable[[Alert, str], None]

    def handle(self, event: dict[str, Any]) -> Response:
        """Dispatch on ``routeKey`` via a dict (O(1), no if/elif ladder)."""
        routes: dict[str, Callable[[dict[str, Any]], Response]] = {
            "POST /alerts": self.submit_alert,
            "GET /reports/{report_id}": self.get_report,
            "POST /reports/{report_id}/feedback": self.post_feedback,
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

    def get_report(self, event: dict[str, Any]) -> Response:
        """Return the report JSON, including its computed ``status``."""
        report = self.reports.get(event["pathParameters"]["report_id"])
        if report is None:
            return _json(404, {"error": "report not found (triage may still be running)"})
        return _json(200, report.model_dump(mode="json"))

    def post_feedback(self, event: dict[str, Any]) -> Response:
        """Store feedback; it feeds the evaluation dataset (continuous improvement)."""
        report_id = event["pathParameters"]["report_id"]
        if self.reports.get(report_id) is None:
            return _json(404, {"error": "report not found"})
        body = json.loads(event.get("body") or "{}")
        caller = event.get("requestContext", {}).get("authorizer", {}).get("iam", {}).get("userArn")
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

    api = Api(
        reports=DynamoReportStore(dynamo.Table(settings.reports_table)),
        feedback=DynamoFeedbackStore(dynamo.Table(settings.feedback_table)),
        start_triage=start,
    )
    return api.handle(event)
