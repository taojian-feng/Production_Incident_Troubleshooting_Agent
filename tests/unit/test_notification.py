"""The report email: readable, complete, and always accepted by SNS."""

from __future__ import annotations

from incident_agent.domain import Alert, Severity
from incident_agent.domain.enums import EvidenceSource, RetrievalMode, StepKind
from incident_agent.domain.models import CostSummary, Evidence, TriageReport, TriageStep
from incident_agent.handlers.notification import format_report_email


def _step(number: int, kind: StepKind, content: str, *, verified: bool = True) -> TriageStep:
    return TriageStep(
        number=number,
        kind=kind,
        content=content,
        citations=("E1",),
        verified=verified,
        review_reason=None if verified else "cites unknown evidence E9",
    )


def _report(*steps: TriageStep, title: str = "orders-api 5xx") -> TriageReport:
    return TriageReport(
        report_id="rpt-1",
        alert=Alert(service="orders-api", title=title, severity=Severity.SEV2),
        retrieval_mode=RetrievalMode.HYBRID,
        steps=steps,
        evidence=(
            Evidence(
                evidence_id="E1",
                source=EvidenceSource.KNOWLEDGE,
                ref="runbook#0",
                title="Runbook: cache timeouts",
                snippet="details",
            ),
        ),
        cost=CostSummary(usd=0.034),
        latency_ms=12600.0,
    )


def test_body_has_findings_actions_evidence_and_run_details() -> None:
    subject, body = format_report_email(
        _report(
            _step(1, StepKind.SUMMARY, "Cache is timing out."),
            _step(2, StepKind.PRECEDENT, "INC0050101 matches."),
            _step(3, StepKind.RECOMMENDATION, "Restart orders-api."),
        )
    )
    assert subject == "[SEV2] Triage ready: orders-api 5xx"
    assert body.index("SUMMARY") < body.index("SIMILAR PAST") < body.index("RECOMMENDED ACTIONS")
    assert "1. Restart orders-api.\n   Sources: E1" in body
    assert "E1 (knowledge): Runbook: cache timeouts" in body
    assert "Report: rpt-1" in body
    assert "Run: 13s, $0.03" in body
    assert not body.startswith("{")


def test_actions_are_renumbered_when_the_model_already_numbered_them() -> None:
    _, body = format_report_email(
        _report(
            _step(1, StepKind.RECOMMENDATION, "1. Check logs."),
            _step(2, StepKind.RECOMMENDATION, "2) Roll back."),
        )
    )
    assert "1. Check logs." in body
    assert "2. Roll back." in body
    assert "1. 1." not in body


def test_unverified_steps_are_flagged_in_subject_and_body() -> None:
    subject, body = format_report_email(
        _report(_step(1, StepKind.SUMMARY, "Unsure.", verified=False))
    )
    assert subject.startswith("[SEV2] Needs review:")
    assert "Status: NEEDS REVIEW" in body
    assert "NEEDS REVIEW: cites unknown evidence E9" in body


def test_subject_is_always_valid_for_sns() -> None:
    subject, _ = format_report_email(_report(title="café down\nsecond line " + "x" * 200))
    assert subject.isascii()
    assert "\n" not in subject
    assert len(subject) < 100
