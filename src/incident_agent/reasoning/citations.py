"""Enforced citation validation (design section 3).

Deterministic checks run *after* the model answers; the model is never trusted
to grade itself. A step is ``verified`` only if:

1. it cites at least one evidence id,
2. every cited id exists in the evidence actually shown to the model,
3. every concrete identifier the step mentions (ticket ids, error codes)
   appears in the alert or in at least one of the evidence items it cites.

Failing any check flags the step (and therefore the report) for human review.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from incident_agent.domain.enums import StepKind
from incident_agent.domain.models import TriageStep
from incident_agent.retrieval.text import extract_error_codes, extract_ticket_ids

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from incident_agent.domain.models import Evidence, LLMTriageStep

STEP_ORDER: tuple[StepKind, ...] = (StepKind.SUMMARY, StepKind.PRECEDENT, StepKind.RECOMMENDATION)


def _check(
    step: LLMTriageStep, evidence_by_id: Mapping[str, Evidence], alert_text: str
) -> str | None:
    """Return why ``step`` is not grounded, or ``None`` if it is."""
    if not step.citations:
        return "no citations"
    unknown = [c for c in step.citations if c not in evidence_by_id]
    if unknown:
        return f"cites unknown evidence {unknown}"
    cited_text = " ".join(
        f"{evidence_by_id[c].ref} {evidence_by_id[c].title} {evidence_by_id[c].snippet}"
        for c in step.citations
    )
    claimed = set(extract_ticket_ids(step.content)) | set(extract_error_codes(step.content))
    unsupported = sorted(i for i in claimed if i not in cited_text and i not in alert_text)
    if unsupported:
        return f"identifiers not found in cited evidence: {unsupported}"
    return None


def validate_steps(
    steps: Sequence[LLMTriageStep], evidence: Sequence[Evidence], alert_text: str = ""
) -> list[TriageStep]:
    """Order steps canonically, number them, and attach verification results.

    ``sorted`` is stable, so multiple recommendations keep the model's order.
    """
    evidence_by_id = {e.evidence_id: e for e in evidence}
    ordered = sorted(steps, key=lambda s: STEP_ORDER.index(s.kind))
    validated: list[TriageStep] = []
    for number, step in enumerate(ordered, start=1):
        reason = _check(step, evidence_by_id, alert_text)
        validated.append(
            TriageStep(
                number=number,
                kind=step.kind,
                content=step.content,
                citations=tuple(dict.fromkeys(step.citations)),
                verified=reason is None,
                review_reason=reason,
            )
        )
    present = {s.kind for s in validated}
    for kind in STEP_ORDER:
        if kind not in present:
            validated.append(
                TriageStep(
                    number=len(validated) + 1,
                    kind=kind,
                    content=f"No {kind.value} step was produced.",
                    citations=(),
                    verified=False,
                    review_reason="missing required step",
                )
            )
    return validated
