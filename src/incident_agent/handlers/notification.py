"""Plain text report email for on-call engineers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from incident_agent.domain.enums import EvidenceSource, ReportStatus, StepKind

if TYPE_CHECKING:
    from incident_agent.domain.models import Evidence, TriageReport, TriageStep

_SNS_SUBJECT_LIMIT = 100
_LOG_SNIPPET_CHARS = 160
_LEADING_NUMBER = re.compile(r"^\s*\d+[.)]\s+")
_HEADINGS = (
    (StepKind.SUMMARY, "SUMMARY"),
    (StepKind.PRECEDENT, "SIMILAR PAST INCIDENTS"),
    (StepKind.RECOMMENDATION, "RECOMMENDED ACTIONS"),
)


def _subject(report: TriageReport) -> str:
    """SNS rejects non-ASCII text, line breaks and subjects of 100 characters or more."""
    label = "Needs review" if report.status is ReportStatus.NEEDS_REVIEW else "Triage ready"
    text = f"[{report.alert.severity}] {label}: {report.alert.title}"
    ascii_text = text.encode("ascii", "replace").decode("ascii")
    return " ".join(ascii_text.split())[: _SNS_SUBJECT_LIMIT - 1]


def _status_line(report: TriageReport) -> str:
    if report.status is ReportStatus.VALIDATED:
        return "Status: Validated. Every step cites verified evidence."
    return "Status: NEEDS REVIEW. Check the flagged steps against the evidence before acting."


def _step_lines(step: TriageStep, number: int | None) -> list[str]:
    text = _LEADING_NUMBER.sub("", step.content) if number is not None else step.content
    prefix = f"{number}. " if number is not None else ""
    lines = [f"{prefix}{text}"]
    if step.citations:
        lines.append(f"   Sources: {', '.join(step.citations)}")
    if not step.verified:
        reason = step.review_reason or "not tied to verified evidence"
        lines.append(f"   NEEDS REVIEW: {reason}")
    return lines


def _section(heading: str, steps: list[TriageStep], *, numbered: bool) -> list[str]:
    lines = [heading]
    for position, step in enumerate(steps, start=1):
        lines += [*_step_lines(step, position if numbered else None), ""]
    return lines


def _evidence_line(evidence: Evidence) -> str:
    """Log titles are only counts, so they also show which error was counted."""
    title = evidence.title.rstrip(" >")
    if evidence.source is EvidenceSource.LOGS:
        title = f"{title}: {evidence.snippet[:_LOG_SNIPPET_CHARS]}"
    return f"{evidence.evidence_id} ({evidence.source}): {title}"


def format_report_email(report: TriageReport) -> tuple[str, str]:
    """Return ``(subject, body)``: findings first, then evidence, then run details."""
    alert = report.alert
    lines = [
        f"{alert.severity} on {alert.service}: {alert.title}",
        _status_line(report),
        "",
    ]
    for kind, heading in _HEADINGS:
        steps = [step for step in report.steps if step.kind is kind]
        if steps:
            lines += _section(heading, steps, numbered=kind is StepKind.RECOMMENDATION)
    if report.evidence:
        lines.append("EVIDENCE")
        lines += [_evidence_line(e) for e in report.evidence]
        lines.append("")
    lines += [
        f"Report: {report.report_id} (GET /reports/{report.report_id} returns the full report)",
        f"Run: {report.latency_ms / 1000:.0f}s, ${report.cost.usd:.2f}",
    ]
    return _subject(report), "\n".join(lines)
