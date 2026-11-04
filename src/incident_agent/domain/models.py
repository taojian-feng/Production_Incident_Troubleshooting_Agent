"""Boundary models: pydantic v2.

Anything that enters the system from the outside world (EventBridge events,
API Gateway bodies, LLM JSON output, DynamoDB items) or leaves it (reports,
API responses) is a pydantic model. That gives us:

* parsing + validation with precise error messages at the edge,
* JSON schema for free (used to constrain the LLM's structured output),
* ``frozen=True`` so a validated object cannot drift after validation,
* ``extra="forbid"`` so typos in payloads fail loudly instead of being dropped.

Inside the core we pass these objects around by reference; they are immutable,
so sharing is safe. Use ``model_copy(update=...)`` to derive a changed copy.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

from incident_agent.domain.enums import (
    EvidenceSource,
    ReportStatus,
    RetrievalMode,
    Severity,
    StepKind,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceId = Annotated[str, StringConstraints(pattern=r"^E\d+$")]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    """Base for all boundary models: immutable, strict about unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- inputs


class Alert(FrozenModel):
    """A production alert, normalised from whichever monitoring tool raised it."""

    alert_id: NonEmptyStr = Field(default_factory=lambda: f"alrt-{uuid4().hex[:12]}")
    source: Literal["cloudwatch", "api", "pagerduty"] = "api"
    service: NonEmptyStr
    title: NonEmptyStr
    description: str = ""
    severity: Severity = Severity.SEV3
    triggered_at: AwareDatetime = Field(default_factory=_utcnow)
    log_group: str | None = None
    dimensions: dict[str, str] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Title and description as one search string."""
        return f"{self.title}\n{self.description}".strip()

    @classmethod
    def from_cloudwatch_alarm(cls, event: dict[str, Any]) -> Self:
        """Parse an EventBridge ``CloudWatch Alarm State Change`` event.

        Alarm tags are not in the event, so service/severity/log group are read
        from the alarm *description*, which our IaC writes as ``key=value``
        lines (see ``infra/stacks``). Unknown keys are ignored.
        """
        detail = event["detail"]
        config = detail.get("configuration", {})
        description: str = config.get("description") or ""
        meta = dict(re.findall(r"^\s*(\w+)\s*=\s*(.+?)\s*$", description, flags=re.MULTILINE))
        metrics = config.get("metrics") or [{}]
        dims = metrics[0].get("metricStat", {}).get("metric", {}).get("dimensions", {}) or {}
        return cls(
            alert_id=f"cw-{event.get('id', uuid4().hex)}",
            source="cloudwatch",
            service=meta.get("service", dims.get("ServiceName", "unknown")),
            title=detail.get("alarmName", "CloudWatch alarm"),
            description=detail.get("state", {}).get("reason", ""),
            severity=Severity(meta.get("severity", Severity.SEV3)),
            triggered_at=datetime.fromisoformat(event.get("time", _utcnow().isoformat())),
            log_group=meta.get("log_group"),
            dimensions={str(k): str(v) for k, v in dims.items()},
        )


class SourceDocument(FrozenModel):
    """An unstructured document to ingest: runbook, post-mortem, architecture doc."""

    doc_id: NonEmptyStr
    title: NonEmptyStr
    body: str
    doc_type: Literal["runbook", "postmortem", "architecture", "incident"] = "runbook"
    service: str | None = None
    uri: str | None = None


class HistoricalIncident(FrozenModel):
    """A resolved ticket mirrored from ServiceNow / Jira."""

    incident_id: Annotated[str, StringConstraints(pattern=r"^(INC\d{7}|[A-Z][A-Z0-9]+-\d+)$")]
    service: NonEmptyStr
    title: NonEmptyStr
    summary: str
    root_cause: str = ""
    resolution: str = ""
    severity: Severity = Severity.SEV3
    opened_at: AwareDatetime
    resolved_at: AwareDatetime | None = None
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _resolved_after_opened(self) -> Self:
        if self.resolved_at is not None and self.resolved_at < self.opened_at:
            msg = "resolved_at must not be before opened_at"
            raise ValueError(msg)
        return self

    def to_document(self) -> SourceDocument:
        """Render as a document so incidents are also searchable semantically."""
        body = (
            f"# {self.incident_id}: {self.title}\n\n## Summary\n{self.summary}\n\n"
            f"## Root cause\n{self.root_cause}\n\n## Resolution\n{self.resolution}\n"
        )
        return SourceDocument(
            doc_id=self.incident_id,
            title=f"{self.incident_id}: {self.title}",
            body=body,
            doc_type="incident",
            service=self.service,
        )


# --------------------------------------------------------------------------- evidence & report


class Evidence(FrozenModel):
    """A citable fact. Every triage step must reference these by ``evidence_id``."""

    evidence_id: EvidenceId
    source: EvidenceSource
    ref: NonEmptyStr = Field(description="Stable source id: log event id, ticket id, chunk id.")
    title: str
    snippet: str
    score: float = 0.0
    uri: str | None = None


class LLMTriageStep(FrozenModel):
    """Shape the reasoning model must return (used as its JSON schema)."""

    kind: StepKind
    content: NonEmptyStr
    citations: list[EvidenceId] = Field(default_factory=list)


class LLMTriageOutput(FrozenModel):
    """Root object of the reasoning model's structured output."""

    steps: list[LLMTriageStep] = Field(min_length=1)


class LLMGroundingVerdict(FrozenModel):
    """Whether one step's cited sources support it."""

    step_number: int
    supported: bool
    reason: str


class LLMGroundingOutput(FrozenModel):
    """Root object of the grounding model's structured output."""

    verdicts: list[LLMGroundingVerdict]


class TriageStep(FrozenModel):
    """A validated step in the final report."""

    number: int = Field(ge=1)
    kind: StepKind
    content: str
    citations: tuple[EvidenceId, ...]
    verified: bool
    review_reason: str | None = None


class CostSummary(FrozenModel):
    """FinOps roll-up attached to every report."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, float] = Field(default_factory=dict)


class TriageReport(FrozenModel):
    """What the analyst UI renders: ordered steps, evidence, cost, status."""

    report_id: NonEmptyStr = Field(default_factory=lambda: f"rpt-{uuid4().hex[:12]}")
    alert: Alert
    retrieval_mode: RetrievalMode
    steps: tuple[TriageStep, ...]
    evidence: tuple[Evidence, ...]
    cost: CostSummary = CostSummary()
    latency_ms: float = 0.0
    created_at: AwareDatetime = Field(default_factory=_utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> ReportStatus:
        """NEEDS_REVIEW as soon as any step is not tied to verified evidence."""
        if self.steps and all(step.verified for step in self.steps):
            return ReportStatus.VALIDATED
        return ReportStatus.NEEDS_REVIEW


class Feedback(FrozenModel):
    """Analyst thumbs up/down (and optional correction) on a report."""

    report_id: NonEmptyStr
    rating: Literal["up", "down"]
    correction: str | None = None
    user: NonEmptyStr = "anonymous"
    created_at: AwareDatetime = Field(default_factory=_utcnow)

    @field_validator("correction")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None
