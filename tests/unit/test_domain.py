from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from incident_agent.domain import (
    AgentState,
    Alert,
    Chunk,
    Evidence,
    EvidenceSource,
    HistoricalIncident,
    LogEvent,
    ReportStatus,
    RetrievalMode,
    Severity,
    StepKind,
    TokenUsage,
    TriageReport,
    TriageStep,
)

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)

CW_EVENT = {
    "version": "0",
    "id": "c4c1c1c9-6542-e61b-6ef0-8c4d36933a92",
    "detail-type": "CloudWatch Alarm State Change",
    "source": "aws.cloudwatch",
    "time": "2026-11-03T16:25:00Z",
    "detail": {
        "alarmName": "payments-api-5xx-rate",
        "state": {"value": "ALARM", "reason": "Threshold Crossed: 1 datapoint [7.1] > 3.0"},
        "configuration": {
            "description": "service=payments-api\nseverity=SEV1\nlog_group=/app/payments-api",
            "metrics": [
                {"metricStat": {"metric": {"dimensions": {"ServiceName": "payments-api"}}}}
            ],
        },
    },
}


class TestAlert:
    def test_from_cloudwatch_alarm_reads_tags_from_description(self) -> None:
        alert = Alert.from_cloudwatch_alarm(CW_EVENT)
        assert alert.service == "payments-api"
        assert alert.severity is Severity.SEV1
        assert alert.log_group == "/app/payments-api"
        assert alert.source == "cloudwatch"
        assert alert.triggered_at == datetime(2026, 11, 3, 16, 25, tzinfo=UTC)
        assert alert.dimensions == {"ServiceName": "payments-api"}

    def test_is_immutable(self, payments_alert: Alert) -> None:
        with pytest.raises(ValidationError):
            payments_alert.service = "other"  # type: ignore[misc]

    def test_rejects_unknown_fields_and_naive_datetimes(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Alert.model_validate({"service": "s", "title": "t", "sevirity": "SEV1"})
        with pytest.raises(ValidationError, match="timezone"):
            Alert(service="s", title="t", triggered_at=datetime(2026, 1, 1))  # noqa: DTZ001

    def test_model_copy_derives_new_object(self, payments_alert: Alert) -> None:
        changed = payments_alert.model_copy(update={"severity": Severity.SEV2})
        assert changed.severity is Severity.SEV2
        assert payments_alert.severity is Severity.SEV1


class TestHistoricalIncident:
    def test_resolved_before_opened_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="resolved_at"):
            HistoricalIncident(
                incident_id="INC0000001",
                service="s",
                title="t",
                summary="x",
                opened_at=NOW,
                resolved_at=NOW.replace(hour=1),
            )

    @pytest.mark.parametrize("bad_id", ["INC123", "inc0000001", "123"])
    def test_ticket_id_format(self, bad_id: str) -> None:
        with pytest.raises(ValidationError):
            HistoricalIncident(
                incident_id=bad_id, service="s", title="t", summary="", opened_at=NOW
            )

    def test_to_document(self, incidents: list[HistoricalIncident]) -> None:
        doc = incidents[0].to_document()
        assert doc.doc_type == "incident"
        assert doc.doc_id == incidents[0].incident_id
        assert "## Root cause" in doc.body


def _step(*, verified: bool) -> TriageStep:
    return TriageStep(number=1, kind=StepKind.SUMMARY, content="x", citations=(), verified=verified)


class TestReportStatus:
    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ((True, True), ReportStatus.VALIDATED),
            ((True, False), ReportStatus.NEEDS_REVIEW),
            ((), ReportStatus.NEEDS_REVIEW),
        ],
    )
    def test_status_is_computed(
        self, payments_alert: Alert, flags: tuple[bool, ...], expected: ReportStatus
    ) -> None:
        report = TriageReport(
            alert=payments_alert,
            retrieval_mode=RetrievalMode.HYBRID,
            steps=tuple(_step(verified=f) for f in flags),
            evidence=(),
        )
        assert report.status is expected
        assert report.model_dump()["status"] == expected


class TestValueObjects:
    def test_log_signature_masks_volatile_tokens(self) -> None:
        a = LogEvent("1", NOW, "/g", "timeout after 3012ms req=6513270e")
        b = LogEvent("2", NOW, "/g", "timeout after 99ms req=1818e811")
        assert a.signature == b.signature == "timeout after <*> req=<*>"

    def test_log_events_are_hashable_for_set_dedupe(self) -> None:
        event = LogEvent("1", NOW, "/g", "boom")
        assert len({event, LogEvent("1", NOW, "/g", "boom")}) == 1

    def test_chunk_metadata_is_read_only_and_chunk_hashable(self) -> None:
        source = {"title": "T"}
        chunk = Chunk("c1", "d1", "text", source)
        source["title"] = "mutated"  # caller's dict is copied, not aliased
        assert chunk.metadata["title"] == "T"
        with pytest.raises(TypeError):
            chunk.metadata["title"] = "x"  # type: ignore[index]
        assert hash(chunk) == hash(Chunk("c1", "d1", "text", {"other": "meta"}))

    def test_token_usage_adds(self) -> None:
        total = TokenUsage(10, 5) + TokenUsage(1, 2)
        assert (total.input_tokens, total.output_tokens, total.total) == (11, 7, 18)

    def test_slots_reduce_per_instance_memory(self) -> None:
        assert not hasattr(LogEvent("1", NOW, "/g", "m"), "__dict__")


class TestStateCopySemantics:
    def _evidence(self, n: int) -> Evidence:
        return Evidence(
            evidence_id=f"E{n}", source=EvidenceSource.LOGS, ref=f"log:{n}", title="t", snippet="s"
        )

    def test_shallow_copy_aliases_nested_lists(self, payments_alert: Alert) -> None:
        """The bug fork() exists to prevent."""
        state = AgentState(alert=payments_alert)
        shallow = copy.copy(state)
        shallow.evidence.append(self._evidence(1))
        assert state.evidence, "shallow copy shares the list with the original"

    def test_fork_is_independent(self, payments_alert: Alert) -> None:
        state = AgentState(alert=payments_alert)
        branch = state.fork()
        branch.evidence.append(self._evidence(1))
        branch.notes["k"] = "v"
        assert state.evidence == []
        assert state.notes == {}

    def test_merge_deduplicates_by_ref_and_keeps_order(self, payments_alert: Alert) -> None:
        state = AgentState(alert=payments_alert, evidence=[self._evidence(1)])
        a, b = state.fork(), state.fork()
        a.evidence.append(self._evidence(2))
        b.evidence.extend([self._evidence(2), self._evidence(3)])
        b.errors.append("tickets: down")
        state.merge_evidence(a, b)
        assert [e.ref for e in state.evidence] == ["log:1", "log:2", "log:3"]
        assert state.errors == ["tickets: down"]

    def test_snapshot_is_detached(self, payments_alert: Alert) -> None:
        state = AgentState(alert=payments_alert)
        snap = state.snapshot()
        state.notes["later"] = "x"
        assert snap["notes"] == {}
