"""Ticket feeds: other systems' payloads become the shared incident model."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from incident_agent.domain import Feedback, Severity
from incident_agent.domain.models import HistoricalIncident, JiraWebhook, ServiceNowIncident
from incident_agent.handlers.api import Api
from incident_agent.storage.repositories import InMemoryReportStore

TICKETS = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "tickets"


def _payload(name: str) -> str:
    return (TICKETS / name).read_text()


def test_resolved_jira_issue_becomes_an_incident() -> None:
    incident = JiraWebhook.model_validate_json(_payload("jira-issue-resolved.json")).to_incident()
    assert incident.incident_id == "PAY-1432"
    assert incident.service == "payments-api"
    assert incident.severity is Severity.SEV2
    assert "unique constraint" in incident.root_cause
    assert "idempotent" in incident.resolution
    assert incident.opened_at == datetime(2026, 10, 20, 14, 2, tzinfo=UTC)
    assert incident.resolved_at == datetime(2026, 10, 20, 15, 10, tzinfo=UTC)
    assert incident.tags == ("synthetic", "refunds")


def test_open_servicenow_incident_becomes_an_incident_with_no_resolution() -> None:
    record = ServiceNowIncident.model_validate_json(_payload("servicenow-incident-open.json"))
    incident = record.to_incident()
    assert incident.incident_id == "INC0051001"
    assert incident.service == "ledger-service"
    assert incident.severity is Severity.SEV1
    assert incident.resolved_at is None
    assert incident.resolution == ""
    assert incident.opened_at == datetime(2026, 11, 4, 6, 10, tzinfo=UTC)


def test_unknown_fields_from_other_systems_are_ignored() -> None:
    data = json.loads(_payload("servicenow-incident-open.json"))
    data["sys_mod_count"] = "7"
    assert ServiceNowIncident.model_validate(data).number == "INC0051001"


def test_jira_issue_without_a_component_is_rejected() -> None:
    data = json.loads(_payload("jira-issue-resolved.json"))
    data["issue"]["fields"]["components"] = []
    with pytest.raises(ValidationError):
        JiraWebhook.model_validate(data)


class NoFeedback:
    def add(self, feedback: Feedback) -> None:
        raise AssertionError(feedback)

    def for_report(self, report_id: str) -> list[Feedback]:
        raise AssertionError(report_id)


def _api() -> tuple[Api, list[HistoricalIncident]]:
    published: list[HistoricalIncident] = []
    api = Api(
        InMemoryReportStore(),
        NoFeedback(),
        lambda _alert, _report_id: None,
        published.append,
    )
    return api, published


@pytest.mark.parametrize(
    ("route", "filename", "incident_id"),
    [
        ("POST /tickets/jira", "jira-issue-resolved.json", "PAY-1432"),
        ("POST /tickets/servicenow", "servicenow-incident-open.json", "INC0051001"),
    ],
)
def test_webhook_routes_publish_the_ticket(route: str, filename: str, incident_id: str) -> None:
    api, published = _api()
    response = api.handle({"routeKey": route, "body": _payload(filename)})
    assert response["statusCode"] == 202
    assert json.loads(response["body"]) == {"incident_id": incident_id}
    assert [i.incident_id for i in published] == [incident_id]


def test_invalid_webhook_is_rejected_and_publishes_nothing() -> None:
    api, published = _api()
    response = api.handle({"routeKey": "POST /tickets/jira", "body": "{}"})
    assert response["statusCode"] == 422
    assert published == []
