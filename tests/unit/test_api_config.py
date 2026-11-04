from __future__ import annotations

import json
from typing import Any

import pytest

from incident_agent.app import Components
from incident_agent.config import Settings
from incident_agent.domain import Alert, Feedback, ModelTier
from incident_agent.handlers.api import Api
from incident_agent.handlers.triage import parse_alert, triage
from incident_agent.storage.repositories import InMemoryReportStore


class ListFeedback:
    def __init__(self) -> None:
        self.items: list[Feedback] = []

    def add(self, feedback: Feedback) -> None:
        self.items.append(feedback)

    def for_report(self, report_id: str) -> list[Feedback]:
        return [f for f in self.items if f.report_id == report_id]


@pytest.fixture
def api() -> tuple[Api, list[tuple[Alert, str]], ListFeedback]:
    started: list[tuple[Alert, str]] = []
    feedback = ListFeedback()
    return (
        Api(InMemoryReportStore(), feedback, lambda a, r: started.append((a, r))),
        started,
        feedback,
    )


def _event(route: str, body: object = None, report_id: str | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {"routeKey": route}
    if body is not None:
        event["body"] = body if isinstance(body, str) else json.dumps(body)
    if report_id:
        event["pathParameters"] = {"report_id": report_id}
    return event


class TestApi:
    def test_submit_alert_starts_async_triage(self, api: tuple[Api, list[Any], Any]) -> None:
        handler, started, _ = api
        resp = handler.handle(_event("POST /alerts", {"service": "payments-api", "title": "504s"}))
        assert resp["statusCode"] == 202
        report_id = json.loads(resp["body"])["report_id"]
        assert started[0][1] == report_id

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            ({"service": "s"}, 422),
            ("{not json", 422),
            ({"service": "s", "title": "t", "x": 1}, 422),
        ],
    )
    def test_invalid_bodies(self, api: tuple[Api, Any, Any], body: object, status: int) -> None:
        handler, _, _ = api
        assert handler.handle(_event("POST /alerts", body))["statusCode"] == status

    def test_unknown_route(self, api: tuple[Api, Any, Any]) -> None:
        assert api[0].handle(_event("DELETE /everything"))["statusCode"] == 404

    async def test_get_report_and_feedback(
        self, api: tuple[Api, Any, ListFeedback], local: Components, payments_alert: Alert
    ) -> None:
        handler, _, feedback = api
        assert (
            handler.handle(_event("GET /reports/{report_id}", report_id="nope"))["statusCode"]
            == 404
        )

        report = await triage(payments_alert, local, handler.reports, report_id="rpt-fixed")
        got = handler.handle(_event("GET /reports/{report_id}", report_id="rpt-fixed"))
        assert got["statusCode"] == 200
        assert json.loads(got["body"])["status"] == report.status.value

        resp = handler.handle(
            _event(
                "POST /reports/{report_id}/feedback",
                {"rating": "down", "correction": ""},
                report_id="rpt-fixed",
            )
        )
        assert resp["statusCode"] == 201
        assert feedback.items[0].correction is None  # blank normalised to None
        bad = handler.handle(
            _event("POST /reports/{report_id}/feedback", {"rating": "meh"}, report_id="rpt-fixed")
        )
        assert bad["statusCode"] == 422
        missing = handler.handle(
            _event("POST /reports/{report_id}/feedback", {"rating": "up"}, report_id="x")
        )
        assert missing["statusCode"] == 404

    def test_parse_alert_from_api_payload(self) -> None:
        alert = parse_alert({"alert": {"service": "s", "title": "t"}})
        assert alert.service == "s"


class TestSettings:
    def test_key_secret_selects_claude_api_and_defaults_to_none(self) -> None:
        required = {
            "IA_KNOWLEDGE_BUCKET": "b",
            "IA_INCIDENTS_TABLE": "i",
            "IA_REPORTS_TABLE": "r",
            "IA_FEEDBACK_TABLE": "f",
        }
        assert Settings.from_env(required).anthropic_key_secret is None
        env = {**required, "IA_ANTHROPIC_KEY_SECRET": "incident-agent/key"}
        assert Settings.from_env(env).anthropic_key_secret == env["IA_ANTHROPIC_KEY_SECRET"]

    def test_from_env(self) -> None:
        env = {
            "IA_KNOWLEDGE_BUCKET": "b",
            "IA_INCIDENTS_TABLE": "i",
            "IA_REPORTS_TABLE": "r",
            "IA_FEEDBACK_TABLE": "f",
            "IA_MODEL_HEAVY": "us.anthropic.claude-opus-5-5",
            "IA_MAX_USD_PER_RUN": "1.5",
        }
        settings = Settings.from_env(env)
        assert settings.model_overrides == {ModelTier.HEAVY: "us.anthropic.claude-opus-5-5"}
        assert settings.max_usd_per_run == 1.5
        assert settings.notify_topic_arn is None

    def test_missing_required_names_the_variable(self) -> None:
        with pytest.raises(KeyError, match="IA_KNOWLEDGE_BUCKET"):
            Settings.from_env({})
