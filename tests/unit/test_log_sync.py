"""Log sync: first-seen error patterns become small knowledge documents."""

from __future__ import annotations

import base64
import gzip
import json
from datetime import UTC, datetime

from incident_agent.domain.values import LogEvent
from incident_agent.handlers.log_sync import parse_subscription_event, pattern_documents
from incident_agent.ingestion.loaders import parse_markdown

NOW = datetime(2026, 11, 4, tzinfo=UTC)


def _event(message: str) -> LogEvent:
    return LogEvent(event_id="1", timestamp=NOW, log_group="/app/orders-api", message=message)


def _delivery(payload: dict[str, object]) -> dict[str, dict[str, str]]:
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    return {"awslogs": {"data": encoded}}


def test_subscription_delivery_is_decoded() -> None:
    events = parse_subscription_event(
        _delivery(
            {
                "messageType": "DATA_MESSAGE",
                "logGroup": "/app/orders-api",
                "logStream": "task-1",
                "logEvents": [{"id": "e1", "timestamp": 1793750400000, "message": "ERROR boom"}],
            }
        )
    )
    assert [(e.event_id, e.log_group, e.message) for e in events] == [
        ("e1", "/app/orders-api", "ERROR boom")
    ]


def test_control_messages_hold_no_events() -> None:
    assert parse_subscription_event(_delivery({"messageType": "CONTROL_MESSAGE"})) == []


def test_repeats_of_a_pattern_produce_one_document_and_non_errors_none() -> None:
    documents = pattern_documents(
        [
            _event("ERROR timeout after 3012ms"),
            _event("ERROR timeout after 2999ms"),
            _event("INFO all good"),
            _event("ERROR connection refused"),
        ],
        "/app/",
    )
    assert len(documents) == 2
    assert all(d.key.startswith("knowledge/log-orders-api-") for d in documents)


def test_same_pattern_always_gets_the_same_key() -> None:
    first = pattern_documents([_event("ERROR timeout after 3012ms")], "/app/")
    second = pattern_documents([_event("ERROR timeout after 9ms")], "/app/")
    assert first[0].key == second[0].key


def test_document_is_valid_knowledge() -> None:
    document = pattern_documents([_event("ERROR timeout after 3012ms")], "/app/")[0]
    parsed = parse_markdown("log-1", document.body)
    assert parsed.service == "orders-api"
    assert parsed.doc_type == "log_pattern"
    assert "timeout" in parsed.body
